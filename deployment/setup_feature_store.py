from __future__ import annotations

import argparse
import logging

from google.api_core import exceptions
from google.cloud.aiplatform_v1 import (
    CreateFeatureOnlineStoreRequest,
    CreateFeatureViewRequest,
    FeatureOnlineStore,
    FeatureOnlineStoreAdminServiceClient,
    FeatureView,
    GetFeatureOnlineStoreRequest,
    GetFeatureViewRequest,
    SyncFeatureViewRequest,
)

from gcp.config import (
    CUSTOMER_FEATURES_TABLE,
    FEATURE_STORE_NAME,
    FEATURE_VIEW_NAME,
    PROJECT_ID,
    REGION,
)

LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", default=PROJECT_ID)
    parser.add_argument("--location", default=REGION)
    parser.add_argument("--feature-store-name", default=FEATURE_STORE_NAME)
    parser.add_argument("--feature-view-name", default=FEATURE_VIEW_NAME)
    parser.add_argument("--source-table", default=CUSTOMER_FEATURES_TABLE)
    parser.add_argument("--entity-id-column", default="customer_ID")
    parser.add_argument("--skip-sync", action="store_true")
    return parser.parse_args()


def get_client(location: str) -> FeatureOnlineStoreAdminServiceClient:
    return FeatureOnlineStoreAdminServiceClient(
        client_options={"api_endpoint": f"{location}-aiplatform.googleapis.com"}
    )


def ensure_online_store(
    client: FeatureOnlineStoreAdminServiceClient,
    project: str,
    location: str,
    feature_store_name: str,
) -> str:
    store_resource = client.feature_online_store_path(
        project=project,
        location=location,
        feature_online_store=feature_store_name,
    )
    try:
        client.get_feature_online_store(
            request=GetFeatureOnlineStoreRequest(name=store_resource)
        )
        LOGGER.info("Using existing Feature Online Store: %s", store_resource)
        return store_resource
    except exceptions.NotFound:
        pass

    parent = client.common_location_path(project, location)
    operation = client.create_feature_online_store(
        request=CreateFeatureOnlineStoreRequest(
            parent=parent,
            feature_online_store_id=feature_store_name,
            feature_online_store=FeatureOnlineStore(
                optimized=FeatureOnlineStore.Optimized()
            ),
        )
    )
    result = operation.result()
    LOGGER.info("Created Feature Online Store: %s", result.name)
    return result.name


def ensure_feature_view(
    client: FeatureOnlineStoreAdminServiceClient,
    project: str,
    location: str,
    feature_store_name: str,
    feature_view_name: str,
    source_table: str,
    entity_id_column: str,
) -> str:
    view_resource = client.feature_view_path(
        project=project,
        location=location,
        feature_online_store=feature_store_name,
        feature_view=feature_view_name,
    )
    try:
        client.get_feature_view(request=GetFeatureViewRequest(name=view_resource))
        LOGGER.info("Using existing Feature View: %s", view_resource)
        return view_resource
    except exceptions.NotFound:
        pass

    parent = client.feature_online_store_path(
        project=project,
        location=location,
        feature_online_store=feature_store_name,
    )
    operation = client.create_feature_view(
        request=CreateFeatureViewRequest(
            parent=parent,
            feature_view_id=feature_view_name,
            feature_view=FeatureView(
                big_query_source=FeatureView.BigQuerySource(
                    uri=f"bq://{source_table}",
                    entity_id_columns=[entity_id_column],
                ),
            ),
            run_sync_immediately=True,
        )
    )
    result = operation.result()
    LOGGER.info("Created Feature View: %s", result.name)
    return result.name


def sync_feature_view(
    client: FeatureOnlineStoreAdminServiceClient,
    feature_view_resource: str,
) -> str:
    response = client.sync_feature_view(
        request=SyncFeatureViewRequest(feature_view=feature_view_resource)
    )
    LOGGER.info("Started Feature View sync: %s", response.feature_view_sync)
    return response.feature_view_sync


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    args = parse_args()
    client = get_client(args.location)
    ensure_online_store(
        client=client,
        project=args.project,
        location=args.location,
        feature_store_name=args.feature_store_name,
    )
    feature_view_resource = ensure_feature_view(
        client=client,
        project=args.project,
        location=args.location,
        feature_store_name=args.feature_store_name,
        feature_view_name=args.feature_view_name,
        source_table=args.source_table,
        entity_id_column=args.entity_id_column,
    )
    if not args.skip_sync:
        sync_feature_view(client, feature_view_resource)


if __name__ == "__main__":
    main()
