import argparse
import logging

from pyspark.sql import SparkSession
from pyspark.sql import functions as F

from amex_default.config import DATE_COL, ID_COL
from amex_default.features_spark import build_customer_features

LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--labels", default=None)
    parser.add_argument("--output", required=True)
    parser.add_argument("--start-date", default=None)
    parser.add_argument("--end-date", default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    args = parse_args()

    spark = SparkSession.builder.appName("amex-build-features").getOrCreate()

    try:
        LOGGER.info("Reading preprocessed statements from %s", args.input)
        statements = spark.read.parquet(args.input)

        if args.start_date or args.end_date:
            statements = statements.withColumn(DATE_COL, F.to_date(F.col(DATE_COL)))
            if args.start_date:
                LOGGER.info("Filtering statements from %s", args.start_date)
                statements = statements.where(F.col(DATE_COL) >= F.lit(args.start_date))
            if args.end_date:
                LOGGER.info("Filtering statements through %s", args.end_date)
                statements = statements.where(F.col(DATE_COL) <= F.lit(args.end_date))

        LOGGER.info("Building customer-level features")
        features = build_customer_features(statements)

        if args.labels:
            LOGGER.info("Reading labels from %s", args.labels)
            labels = spark.read.csv(args.labels, header=True, inferSchema=True)

            LOGGER.info("Joining labels")
            features = features.join(labels, on=ID_COL, how="left")
        else:
            LOGGER.info("No labels path provided; writing unlabeled features")

        mode = "overwrite" if args.overwrite else "errorifexists"

        LOGGER.info("Writing features to %s", args.output)
        (features.write.mode(mode).option("compression", "snappy").parquet(args.output))

    finally:
        spark.stop()


if __name__ == "__main__":
    main()
