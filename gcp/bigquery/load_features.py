from __future__ import annotations

import argparse
import logging

from google.cloud import bigquery
from google.cloud.exceptions import NotFound

PROJECT_ID = "amex-credit-risk-ml"
LOCATION = "US"
DATASET_ID = "amex_ml"
TABLE_ID = "train_features"
SOURCE_URI = "gs://amex-credit-risk-ml-data/processed/v1/train_features/*.parquet"

LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", default=PROJECT_ID)
    parser.add_argument("--location", default=LOCATION)
    parser.add_argument("--dataset", default=DATASET_ID)
    parser.add_argument("--table", default=TABLE_ID)
    parser.add_argument("--source-uri", default=SOURCE_URI)
    parser.add_argument(
        "--write-disposition",
        default=bigquery.WriteDisposition.WRITE_TRUNCATE,
        choices=[
            bigquery.WriteDisposition.WRITE_TRUNCATE,
            bigquery.WriteDisposition.WRITE_APPEND,
            bigquery.WriteDisposition.WRITE_EMPTY,
        ],
    )
    return parser.parse_args()


def ensure_dataset(
    client: bigquery.Client,
    dataset_id: str,
    location: str,
) -> bigquery.Dataset:
    dataset_ref = bigquery.DatasetReference(client.project, dataset_id)

    try:
        return client.get_dataset(dataset_ref)
    except NotFound:
        dataset = bigquery.Dataset(dataset_ref)
        dataset.location = location
        return client.create_dataset(dataset)


def load_parquet_to_bigquery(args: argparse.Namespace) -> None:
    client = bigquery.Client(project=args.project, location=args.location)
    dataset = ensure_dataset(client, args.dataset, args.location)
    table_ref = dataset.table(args.table)

    job_config = bigquery.LoadJobConfig(
        source_format=bigquery.SourceFormat.PARQUET,
        write_disposition=args.write_disposition,
    )

    LOGGER.info("Loading %s into %s", args.source_uri, table_ref)
    load_job = client.load_table_from_uri(
        args.source_uri,
        table_ref,
        job_config=job_config,
        location=args.location,
    )
    load_job.result()

    table = client.get_table(table_ref)
    LOGGER.info(
        "Loaded %d rows and %d columns into %s",
        table.num_rows,
        len(table.schema),
        table.full_table_id,
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    load_parquet_to_bigquery(parse_args())


if __name__ == "__main__":
    main()
