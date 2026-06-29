from __future__ import annotations

import argparse
import logging
from datetime import UTC, datetime

import numpy as np
import pandas as pd
from google.cloud import bigquery, storage

PROJECT_ID = "amex-credit-risk-ml"
BQ_LOCATION = "US"
BASELINE_TABLE = "amex-credit-risk-ml.amex_ml.train_features"
CURRENT_TABLE = "amex-credit-risk-ml.amex_ml.train_features"
METRICS_TABLE = "amex-credit-risk-ml.amex_ml.drift_metrics"
OUTPUT_URI = "gs://amex-credit-risk-ml-data/monitoring/drift_report.csv"

ID_COL = "customer_ID"
TARGET_COL = "target"
PSI_THRESHOLD = 0.2

LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", default=PROJECT_ID)
    parser.add_argument("--bq-location", default=BQ_LOCATION)
    parser.add_argument("--baseline-table", default=BASELINE_TABLE)
    parser.add_argument("--current-table", default=CURRENT_TABLE)
    parser.add_argument("--metrics-table", default=METRICS_TABLE)
    parser.add_argument("--output-uri", default=OUTPUT_URI)
    parser.add_argument("--psi-threshold", type=float, default=PSI_THRESHOLD)
    parser.add_argument("--bins", type=int, default=10)
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--baseline-start-date", default=None)
    parser.add_argument("--baseline-end-date", default=None)
    parser.add_argument("--current-start-date", default=None)
    parser.add_argument("--current-end-date", default=None)
    return parser.parse_args()


def read_table(
    client: bigquery.Client,
    table: str,
    max_rows: int | None,
) -> pd.DataFrame:
    query = f"SELECT * FROM `{table}`"
    if max_rows:
        query += f" LIMIT {max_rows}"

    LOGGER.info("Reading %s", table)
    return client.query(query).result().to_dataframe()


def calculate_psi(
    baseline: pd.Series,
    current: pd.Series,
    bins: int,
) -> float:
    baseline = baseline.dropna()
    current = current.dropna()
    if baseline.empty or current.empty:
        return float("nan")

    quantiles = np.linspace(0, 1, bins + 1)
    edges = np.unique(np.quantile(baseline, quantiles))
    if len(edges) < 2:
        return 0.0

    baseline_counts, _ = np.histogram(baseline, bins=edges)
    current_counts, _ = np.histogram(current, bins=edges)

    baseline_pct = baseline_counts / max(baseline_counts.sum(), 1)
    current_pct = current_counts / max(current_counts.sum(), 1)

    baseline_pct = np.where(baseline_pct == 0, 0.0001, baseline_pct)
    current_pct = np.where(current_pct == 0, 0.0001, current_pct)

    return float(
        np.sum((current_pct - baseline_pct) * np.log(current_pct / baseline_pct))
    )


def build_drift_report(
    baseline_df: pd.DataFrame,
    current_df: pd.DataFrame,
    bins: int,
    threshold: float,
    baseline_start_date: str | None = None,
    baseline_end_date: str | None = None,
    current_start_date: str | None = None,
    current_end_date: str | None = None,
) -> pd.DataFrame:
    excluded = {ID_COL, TARGET_COL}
    numeric_columns = [
        column
        for column in baseline_df.select_dtypes(include=[np.number]).columns
        if column not in excluded and column in current_df.columns
    ]

    report_time = datetime.now(UTC).isoformat()
    rows = []
    for column in numeric_columns:
        psi = calculate_psi(baseline_df[column], current_df[column], bins)
        rows.append(
            {
                "feature_name": column,
                "psi": psi,
                "drifted": bool(psi > threshold) if not np.isnan(psi) else False,
                "threshold": threshold,
                "baseline_rows": int(baseline_df[column].notna().sum()),
                "current_rows": int(current_df[column].notna().sum()),
                "baseline_start_date": baseline_start_date,
                "baseline_end_date": baseline_end_date,
                "current_start_date": current_start_date,
                "current_end_date": current_end_date,
                "computed_at": report_time,
            }
        )

    return pd.DataFrame(rows).sort_values("psi", ascending=False)


def write_gcs_csv(df: pd.DataFrame, output_uri: str) -> None:
    if not output_uri.startswith("gs://"):
        raise ValueError("--output-uri must be a GCS URI.")

    bucket_name, blob_name = output_uri.removeprefix("gs://").split("/", 1)
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    bucket.blob(blob_name).upload_from_string(
        df.to_csv(index=False),
        content_type="text/csv",
    )


def save_report(
    client: bigquery.Client,
    report: pd.DataFrame,
    metrics_table: str,
    output_uri: str,
) -> None:
    LOGGER.info("Writing drift report to %s", output_uri)
    write_gcs_csv(report, output_uri)

    LOGGER.info("Writing drift metrics to %s", metrics_table)
    job_config = bigquery.LoadJobConfig(
        write_disposition=bigquery.WriteDisposition.WRITE_APPEND,
    )
    client.load_table_from_dataframe(
        report,
        metrics_table,
        job_config=job_config,
    ).result()


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    args = parse_args()
    client = bigquery.Client(project=args.project, location=args.bq_location)

    baseline_df = read_table(client, args.baseline_table, args.max_rows)
    current_df = read_table(client, args.current_table, args.max_rows)
    report = build_drift_report(
        baseline_df,
        current_df,
        bins=args.bins,
        threshold=args.psi_threshold,
        baseline_start_date=args.baseline_start_date,
        baseline_end_date=args.baseline_end_date,
        current_start_date=args.current_start_date,
        current_end_date=args.current_end_date,
    )
    save_report(client, report, args.metrics_table, args.output_uri)

    drifted_count = int(report["drifted"].sum())
    LOGGER.info("Detected drift in %d features", drifted_count)


if __name__ == "__main__":
    main()
