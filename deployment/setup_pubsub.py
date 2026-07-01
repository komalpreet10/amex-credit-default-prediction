from __future__ import annotations

import argparse
import logging
import os
import subprocess

from google.api_core.exceptions import AlreadyExists, NotFound
from google.cloud import monitoring_v3, pubsub_v1
from google.protobuf.duration_pb2 import Duration
from google.protobuf.field_mask_pb2 import FieldMask

from gcp.config import (
    PIPELINE_TRIGGER_TOPIC,
    PROJECT_ID,
    STATEMENT_ACK_DEADLINE_SECONDS,
    STATEMENT_DLQ_ALERT_ALIGNMENT_SECONDS,
    STATEMENT_DLQ_ALERT_DISPLAY_NAME,
    STATEMENT_DLQ_ALERT_THRESHOLD,
    STATEMENT_DLQ_MAX_DELIVERY_ATTEMPTS,
    STATEMENT_DLQ_TOPIC,
    STATEMENT_RETRY_MAX_BACKOFF_SECONDS,
    STATEMENT_RETRY_MIN_BACKOFF_SECONDS,
    STATEMENT_SUBSCRIPTION,
    STATEMENT_TOPIC,
)

LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", default=PROJECT_ID)
    parser.add_argument(
        "--notification-channel",
        default=os.environ.get("MONITORING_NOTIFICATION_CHANNEL"),
        help="Existing Cloud Monitoring notification channel resource name.",
    )
    parser.add_argument(
        "--alert-email",
        default=os.environ.get("ALERT_EMAIL"),
        help="Email address used to create/find a Cloud Monitoring email channel.",
    )
    return parser.parse_args()


def run(command: list[str], check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(command, check=check, text=True, capture_output=True)


def project_number(project: str) -> str:
    result = run(
        [
            "gcloud",
            "projects",
            "describe",
            project,
            "--format=value(projectNumber)",
        ]
    )
    return result.stdout.strip()


def ensure_topic(
    publisher: pubsub_v1.PublisherClient,
    project: str,
    topic: str,
) -> str:
    topic_path = publisher.topic_path(project, topic)
    try:
        publisher.get_topic(request={"topic": topic_path})
        LOGGER.info("Topic already exists: %s", topic_path)
    except NotFound:
        publisher.create_topic(request={"name": topic_path})
        LOGGER.info("Created topic: %s", topic_path)
    return topic_path


def grant_topic_publisher(project: str, topic: str, member: str) -> None:
    run(
        [
            "gcloud",
            "pubsub",
            "topics",
            "add-iam-policy-binding",
            topic,
            "--project",
            project,
            "--member",
            member,
            "--role",
            "roles/pubsub.publisher",
            "--quiet",
        ]
    )
    LOGGER.info("Granted roles/pubsub.publisher on %s to %s", topic, member)


def grant_subscription_subscriber(project: str, subscription: str, member: str) -> None:
    run(
        [
            "gcloud",
            "pubsub",
            "subscriptions",
            "add-iam-policy-binding",
            subscription,
            "--project",
            project,
            "--member",
            member,
            "--role",
            "roles/pubsub.subscriber",
            "--quiet",
        ]
    )
    LOGGER.info("Granted roles/pubsub.subscriber on %s to %s", subscription, member)


def subscription_policies(
    dlq_topic_path: str,
) -> tuple[pubsub_v1.types.DeadLetterPolicy, pubsub_v1.types.RetryPolicy]:
    dead_letter_policy = pubsub_v1.types.DeadLetterPolicy(
        dead_letter_topic=dlq_topic_path,
        max_delivery_attempts=STATEMENT_DLQ_MAX_DELIVERY_ATTEMPTS,
    )
    retry_policy = pubsub_v1.types.RetryPolicy(
        minimum_backoff=Duration(seconds=STATEMENT_RETRY_MIN_BACKOFF_SECONDS),
        maximum_backoff=Duration(seconds=STATEMENT_RETRY_MAX_BACKOFF_SECONDS),
    )
    return dead_letter_policy, retry_policy


def ensure_subscription(
    subscriber: pubsub_v1.SubscriberClient,
    project: str,
    subscription: str,
    topic_path: str,
    dlq_topic_path: str,
) -> str:
    subscription_path = subscriber.subscription_path(project, subscription)
    dead_letter_policy, retry_policy = subscription_policies(dlq_topic_path)

    request = {
        "name": subscription_path,
        "topic": topic_path,
        "ack_deadline_seconds": STATEMENT_ACK_DEADLINE_SECONDS,
        "dead_letter_policy": dead_letter_policy,
        "retry_policy": retry_policy,
    }
    try:
        subscriber.create_subscription(request=request)
        LOGGER.info(
            "Created subscription with DLQ/retry policies: %s", subscription_path
        )
    except AlreadyExists:
        existing = subscriber.get_subscription(
            request={"subscription": subscription_path}
        )
        existing.dead_letter_policy = dead_letter_policy
        existing.retry_policy = retry_policy
        existing.ack_deadline_seconds = STATEMENT_ACK_DEADLINE_SECONDS
        subscriber.update_subscription(
            request={
                "subscription": existing,
                "update_mask": FieldMask(
                    paths=[
                        "dead_letter_policy",
                        "retry_policy",
                        "ack_deadline_seconds",
                    ]
                ),
            }
        )
        LOGGER.info(
            "Updated subscription with DLQ/retry policies: %s", subscription_path
        )

    return subscription_path


def find_email_channel(
    client: monitoring_v3.NotificationChannelServiceClient,
    parent: str,
    email: str,
) -> str | None:
    for channel in client.list_notification_channels(request={"name": parent}):
        if channel.type_ != "email":
            continue
        if channel.labels.get("email_address") == email:
            return channel.name
    return None


def ensure_email_channel(project: str, email: str) -> str:
    client = monitoring_v3.NotificationChannelServiceClient()
    parent = f"projects/{project}"
    existing = find_email_channel(client, parent, email)
    if existing:
        LOGGER.info("Using existing email notification channel: %s", existing)
        return existing

    channel = monitoring_v3.NotificationChannel(
        display_name=f"AMEX alerts - {email}",
        type_="email",
        labels={"email_address": email},
        enabled=True,
    )
    response = client.create_notification_channel(
        request={"name": parent, "notification_channel": channel}
    )
    LOGGER.info("Created email notification channel: %s", response.name)
    return response.name


def ensure_dlq_alert_policy(
    project: str,
    dlq_topic: str,
    notification_channel: str,
) -> None:
    client = monitoring_v3.AlertPolicyServiceClient()
    parent = f"projects/{project}"
    for policy in client.list_alert_policies(request={"name": parent}):
        if policy.display_name == STATEMENT_DLQ_ALERT_DISPLAY_NAME:
            LOGGER.info("Alert policy already exists: %s", policy.name)
            return

    metric_filter = (
        'metric.type="pubsub.googleapis.com/topic/send_message_operation_count" '
        'AND resource.type="pubsub_topic" '
        f'AND resource.label."topic_id"="{dlq_topic}"'
    )
    aggregation = monitoring_v3.Aggregation(
        alignment_period=Duration(seconds=STATEMENT_DLQ_ALERT_ALIGNMENT_SECONDS),
        per_series_aligner=monitoring_v3.Aggregation.Aligner.ALIGN_DELTA,
        cross_series_reducer=monitoring_v3.Aggregation.Reducer.REDUCE_SUM,
    )
    condition = monitoring_v3.AlertPolicy.Condition(
        display_name="DLQ received messages",
        condition_threshold=monitoring_v3.AlertPolicy.Condition.MetricThreshold(
            filter=metric_filter,
            comparison=(
                monitoring_v3.AlertPolicy.Condition.MetricThreshold.ComparisonType.COMPARISON_GT
            ),
            threshold_value=STATEMENT_DLQ_ALERT_THRESHOLD,
            duration=Duration(seconds=0),
            aggregations=[aggregation],
        ),
    )
    policy = monitoring_v3.AlertPolicy(
        display_name=STATEMENT_DLQ_ALERT_DISPLAY_NAME,
        combiner=monitoring_v3.AlertPolicy.ConditionCombinerType.OR,
        conditions=[condition],
        notification_channels=[notification_channel],
        enabled=True,
    )
    response = client.create_alert_policy(
        request={"name": parent, "alert_policy": policy}
    )
    LOGGER.info("Created DLQ alert policy: %s", response.name)


def resolve_notification_channel(args: argparse.Namespace) -> str:
    if args.notification_channel:
        return args.notification_channel
    if args.alert_email:
        return ensure_email_channel(args.project, args.alert_email)
    raise RuntimeError(
        "Set MONITORING_NOTIFICATION_CHANNEL or ALERT_EMAIL to create the DLQ alert."
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    args = parse_args()

    publisher = pubsub_v1.PublisherClient()
    subscriber = pubsub_v1.SubscriberClient()

    statement_topic_path = ensure_topic(publisher, args.project, STATEMENT_TOPIC)
    dlq_topic_path = ensure_topic(publisher, args.project, STATEMENT_DLQ_TOPIC)

    pubsub_service_account = (
        f"serviceAccount:service-{project_number(args.project)}"
        "@gcp-sa-pubsub.iam.gserviceaccount.com"
    )
    grant_topic_publisher(args.project, STATEMENT_DLQ_TOPIC, pubsub_service_account)

    ensure_subscription(
        subscriber=subscriber,
        project=args.project,
        subscription=STATEMENT_SUBSCRIPTION,
        topic_path=statement_topic_path,
        dlq_topic_path=dlq_topic_path,
    )
    grant_subscription_subscriber(
        args.project,
        STATEMENT_SUBSCRIPTION,
        pubsub_service_account,
    )

    ensure_topic(publisher, args.project, PIPELINE_TRIGGER_TOPIC)
    notification_channel = resolve_notification_channel(args)
    ensure_dlq_alert_policy(args.project, STATEMENT_DLQ_TOPIC, notification_channel)


if __name__ == "__main__":
    main()
