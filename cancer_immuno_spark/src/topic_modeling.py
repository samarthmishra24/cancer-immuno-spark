"""
Branch A of the pipeline: unsupervised topic modeling over the MeSH-filtered
PubMed corpus produced by pubmed_parser.py.

Stages (tokenize -> stop-word removal -> TF-IDF ->
inspect top IDF terms -- on a 15-record "sample" with a hardcoded CSV and
ad-hoc "year"/"abstract" columns). 
It reads the actual pubmed_filtered Parquet that pubmed_parser.py
writes (columns: pmid, title, abstract, pub_year, journal, mesh_terms),
and extends the feasibility script's TF-IDF step with the two stages DA1
always planned but never committed: LDA topic modeling (swept over a few
values of k, since Spark MLlib has no built-in coherence metric -- see
config.py / README) and K-means clustering on L2-normalized TF-IDF vectors
(normalized first so Spark's default Euclidean-distance KMeans behaves like
cosine similarity, the appropriate metric for text vectors).

Run with:
  spark-submit src/topic_modeling.py \
      --input output/pubmed_filtered --output output/topics \
      --lda-ks 5,10,15 --final-k 10 --kmeans-k 10
"""

import argparse
import os
import shutil
import sys

from pyspark.sql import functions as F
from pyspark.ml.feature import (
    Tokenizer,
    StopWordsRemover,
    CountVectorizer,
    IDF,
    Normalizer,
)
from pyspark.ml.clustering import LDA, KMeans

sys.path.append(os.path.dirname(__file__))
from config import get_spark  # noqa: E402


def build_arg_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="Parquet dir from pubmed_parser.py")
    p.add_argument("--output", required=True, help="Output directory for CSV/Parquet results")
    p.add_argument(
        "--lda-ks",
        default="5,10,15",
        help="Comma-separated list of topic counts to sweep for LDA (default: 5,10,15)",
    )
    p.add_argument(
        "--final-k",
        type=int,
        default=10,
        help="Topic count used for the LDA model whose per-paper assignments get written out",
    )
    p.add_argument(
        "--kmeans-k",
        type=int,
        default=10,
        help="Cluster count for K-means on the L2-normalized TF-IDF vectors",
    )
    p.add_argument("--vocab-size", type=int, default=5000, help="Max vocabulary size for CountVectorizer")
    p.add_argument("--min-df", type=float, default=2.0, help="Minimum document frequency for a term to enter the vocabulary")
    return p


def write_csv(df, output_path):
    """Write a Spark DataFrame as a single CSV file (same pattern as join_analysis.py)."""
    temp_dir = output_path + "_tmp"

    df.coalesce(1).write.mode("overwrite").option("header", True).csv(temp_dir)

    part_file = next(
        name for name in os.listdir(temp_dir)
        if name.startswith("part-") and name.endswith(".csv")
    )

    os.replace(os.path.join(temp_dir, part_file), output_path)
    shutil.rmtree(temp_dir)


def tokenize_corpus(df):
    """
    Clean + tokenize the abstract column, then remove stop words.
    Cleaning step (lowercase, strip everything but letters/digits/hyphens/whitespace) but runs it through
    Spark ML's Tokenizer/StopWordsRemover Estimators, matching the pattern
    already used for TF-IDF, rather than the feasibility script's manual
    regexp_replace + split.
    """
    cleaned = df.withColumn(
        "clean_abstract",
        F.regexp_replace(F.lower(F.col("abstract")), r"[^a-z0-9\s-]", ""),
    ).filter(F.col("clean_abstract").isNotNull() & (F.col("clean_abstract") != ""))

    tokenizer = Tokenizer(inputCol="clean_abstract", outputCol="raw_tokens")
    tokenized = tokenizer.transform(cleaned)

    remover = StopWordsRemover(inputCol="raw_tokens", outputCol="filtered_tokens")
    return remover.transform(tokenized)


def build_tfidf(df, vocab_size, min_df):
    """
    CountVectorizer + IDF over filtered_tokens, producing a `features`
    column of TF-IDF vectors. Returns the fitted CountVectorizer model too
    (needed to map vocabulary indices back to real words for reporting).
    """
    cv = CountVectorizer(
        inputCol="filtered_tokens",
        outputCol="tf",
        vocabSize=vocab_size,
        minDF=min_df,
    )
    cv_model = cv.fit(df)
    tf_df = cv_model.transform(df)

    idf = IDF(inputCol="tf", outputCol="features")
    idf_model = idf.fit(tf_df)
    tfidf_df = idf_model.transform(tf_df)

    return tfidf_df, cv_model


def lda_sweep(tfidf_df, cv_model, k_values, max_terms_per_topic=10):
    """
    Fit LDA at each k in k_values. Spark MLlib's LDA exposes no topic
    coherence metric, only logLikelihood/logPerplexity. We report both, plus each topic's
    top terms, so a human can sanity-check interpretability manually
    alongside the numeric scores.

    Returns a list of dicts: {k, log_likelihood, log_perplexity, topics}
    where topics is a list of (topic_id, [top terms]).
    """
    vocab = cv_model.vocabulary
    results = []

    for k in k_values:
        lda = LDA(featuresCol="features", k=k, maxIter=20, seed=42)
        model = lda.fit(tfidf_df)

        log_likelihood = model.logLikelihood(tfidf_df)
        log_perplexity = model.logPerplexity(tfidf_df)

        topics = model.describeTopics(maxTermsPerTopic=max_terms_per_topic).collect()
        topic_terms = []
        for row in topics:
            terms = [vocab[i] for i in row["termIndices"] if i < len(vocab)]
            topic_terms.append((row["topic"], terms))

        print(f"[topic_modeling] LDA k={k}: logLikelihood={log_likelihood:.2f}, logPerplexity={log_perplexity:.4f}")
        for topic_id, terms in topic_terms:
            print(f"  topic {topic_id}: {', '.join(terms)}")

        results.append(
            {
                "k": k,
                "log_likelihood": log_likelihood,
                "log_perplexity": log_perplexity,
                "topics": topic_terms,
            }
        )

    return results


def fit_final_lda(spark, tfidf_df, cv_model, k):
    """Fit one LDA model at the chosen k and return per-paper topic assignments."""
    lda = LDA(featuresCol="features", k=k, maxIter=20, seed=42)
    model = lda.fit(tfidf_df)

    transformed = model.transform(tfidf_df)  # adds `topicDistribution` per row

    # argmax(topicDistribution) -> dominant topic per paper, via a small UDF
    from pyspark.sql.types import IntegerType

    argmax_udf = F.udf(lambda v: int(v.argmax()), IntegerType())
    assignments = transformed.withColumn("dominant_topic", argmax_udf(F.col("topicDistribution")))

    return assignments.select("pmid", "title", "pub_year", "dominant_topic")


def run_kmeans(tfidf_df, k):
    """
    L2-normalize the TF-IDF vectors first (Normalizer, p=2), since Spark's
    KMeans minimizes Euclidean distance by default and normalized Euclidean
    distance ranks the same as cosine similarity -- the appropriate metric
    for text vectors.
    """
    normalizer = Normalizer(inputCol="features", outputCol="norm_features", p=2.0)
    normalized = normalizer.transform(tfidf_df)

    kmeans = KMeans(featuresCol="norm_features", k=k, seed=42)
    model = kmeans.fit(normalized)
    clustered = model.transform(normalized)

    return clustered.select("pmid", "title", "pub_year", F.col("prediction").alias("cluster"))


def main():
    args = build_arg_parser().parse_args()
    k_values = [int(k.strip()) for k in args.lda_ks.split(",") if k.strip()]

    spark = get_spark("cancer-immunology-topic-modeling")

    df = spark.read.parquet(args.input)
    df = df.filter(F.col("abstract").isNotNull())

    total = df.count()
    print(f"[topic_modeling] {total} filtered papers with a non-null abstract loaded from {args.input}")

    tokenized = tokenize_corpus(df)
    tfidf_df, cv_model = build_tfidf(tokenized, args.vocab_size, args.min_df)
    tfidf_df.cache()

    print(f"[topic_modeling] Vocabulary size: {len(cv_model.vocabulary)}")

    # --- LDA: sweep k, report logLikelihood/logPerplexity + top words ----
    sweep_results = lda_sweep(tfidf_df, cv_model, k_values)

    metrics_rows = [(r["k"], float(r["log_likelihood"]), float(r["log_perplexity"])) for r in sweep_results]
    metrics_df = spark.createDataFrame(metrics_rows, ["k", "log_likelihood", "log_perplexity"])
    write_csv(metrics_df, os.path.join(args.output, "lda_sweep_metrics.csv"))

    topic_rows = [
        (r["k"], topic_id, ", ".join(terms))
        for r in sweep_results
        for topic_id, terms in r["topics"]
    ]
    topics_df = spark.createDataFrame(topic_rows, ["k", "topic_id", "top_terms"])
    write_csv(topics_df, os.path.join(args.output, "lda_sweep_topics.csv"))

    # --- Final LDA fit: per-paper dominant topic at --final-k -------------
    assignments = fit_final_lda(spark, tfidf_df, cv_model, args.final_k)
    assignments.write.mode("overwrite").parquet(os.path.join(args.output, "paper_topics.parquet"))
    print(f"[topic_modeling] Wrote per-paper topic assignments (k={args.final_k}) to paper_topics.parquet")

    # --- K-means on L2-normalized TF-IDF vectors ---------------------------
    clusters = run_kmeans(tfidf_df, args.kmeans_k)
    clusters.write.mode("overwrite").parquet(os.path.join(args.output, "paper_clusters.parquet"))
    print(f"[topic_modeling] Wrote K-means cluster assignments (k={args.kmeans_k}) to paper_clusters.parquet")

    print(f"[topic_modeling] All Branch A stages (tokenize -> TF-IDF -> LDA sweep -> final LDA -> K-means) "
          f"executed successfully. Output in {args.output}")

    spark.stop()


if __name__ == "__main__":
    main()
