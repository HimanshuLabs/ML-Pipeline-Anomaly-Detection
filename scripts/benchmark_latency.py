"""Latency smoke benchmark for Project 4 online anomaly inference.

This script sends repeated POST requests to the FastAPI /predict endpoint and
measures client-observed latency. It intentionally uses the active model feature
contract from the local Project 4 codebase so the benchmark payload stays aligned
with the currently deployed model version.

Example:
    python scripts/benchmark_latency.py \
      --url http://localhost:8004/predict \
      --requests 100
"""

from __future__ import annotations

import argparse
import math
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import httpx


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"

if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from anomaly_detection.online_inference import OnlineInferenceService  # noqa: E402


DEFAULT_LATENCY_BUDGET_MS = 200.0


class LatencyBenchmarkError(RuntimeError):
    """Raised when the latency benchmark cannot run safely."""


def percentile(values: list[float], percentile_value: float) -> float:
    """Return nearest-rank percentile for a non-empty latency list."""

    if not values:
        raise LatencyBenchmarkError("cannot calculate percentile for empty values")

    if percentile_value <= 0 or percentile_value > 100:
        raise LatencyBenchmarkError("percentile must be in the range (0, 100]")

    ordered_values = sorted(values)
    rank = math.ceil((percentile_value / 100.0) * len(ordered_values))
    return ordered_values[max(rank - 1, 0)]


def build_model_ready_payload(
    service: OnlineInferenceService,
    *,
    entity_id: str,
) -> dict[str, float | str]:
    """Build a valid model-ready payload for the active model feature contract."""

    if not service.feature_names:
        raise LatencyBenchmarkError("active model exposes no feature names")

    payload: dict[str, float | str] = {
        feature_name: 1.0
        for feature_name in service.feature_names
    }
    payload["entity_id"] = entity_id
    return payload


def run_warmup_requests(
    client: httpx.Client,
    *,
    url: str,
    service: OnlineInferenceService,
    warmup_requests: int,
) -> None:
    """Run warmup requests so startup noise does not dominate the benchmark."""

    for index in range(warmup_requests):
        entity_id = f"latency_warmup_{index:04d}"
        payload = build_model_ready_payload(service, entity_id=entity_id)

        response = client.post(
            url,
            json={
                "entity_id": entity_id,
                "feature_payload": payload,
            },
        )

        if response.status_code != 200:
            raise LatencyBenchmarkError(
                "warmup request failed: "
                f"status_code={response.status_code}, body={response.text}"
            )


def run_measured_requests(
    client: httpx.Client,
    *,
    url: str,
    service: OnlineInferenceService,
    request_count: int,
) -> tuple[list[float], list[float]]:
    """Run measured benchmark requests and return client/API latency lists."""

    client_latencies_ms: list[float] = []
    api_latencies_ms: list[float] = []

    for index in range(request_count):
        entity_id = f"latency_user_{index:04d}"
        payload = build_model_ready_payload(service, entity_id=entity_id)

        started_at = time.perf_counter()
        response = client.post(
            url,
            json={
                "entity_id": entity_id,
                "feature_payload": payload,
            },
        )
        elapsed_ms = (time.perf_counter() - started_at) * 1000.0

        if response.status_code != 200:
            raise LatencyBenchmarkError(
                "measured request failed: "
                f"index={index}, status_code={response.status_code}, "
                f"body={response.text}"
            )

        body: dict[str, Any] = response.json()
        api_latency_ms = body.get("latency_ms")

        if not isinstance(api_latency_ms, int | float):
            raise LatencyBenchmarkError(
                f"response missing numeric latency_ms: {body}"
            )

        client_latencies_ms.append(elapsed_ms)
        api_latencies_ms.append(float(api_latency_ms))

    return client_latencies_ms, api_latencies_ms


def format_bool(value: bool) -> str:
    """Render booleans in shell-friendly lowercase."""

    return "true" if value else "false"


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(
        description="Benchmark Project 4 online inference latency.",
    )
    parser.add_argument(
        "--url",
        required=True,
        help="Full /predict endpoint URL, for example http://localhost:8004/predict.",
    )
    parser.add_argument(
        "--requests",
        type=int,
        default=100,
        help="Number of measured requests to send.",
    )
    parser.add_argument(
        "--warmup-requests",
        type=int,
        default=10,
        help="Number of warmup requests sent before measurement.",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=5.0,
        help="HTTP client timeout per request.",
    )
    parser.add_argument(
        "--budget-ms",
        type=float,
        default=DEFAULT_LATENCY_BUDGET_MS,
        help="p95 latency budget in milliseconds.",
    )
    return parser.parse_args()


def main() -> int:
    """Run the latency benchmark."""

    args = parse_args()

    if args.requests <= 0:
        raise LatencyBenchmarkError("--requests must be greater than zero")

    if args.warmup_requests < 0:
        raise LatencyBenchmarkError("--warmup-requests cannot be negative")

    if args.budget_ms <= 0:
        raise LatencyBenchmarkError("--budget-ms must be greater than zero")

    service = OnlineInferenceService()

    with httpx.Client(timeout=args.timeout_seconds) as client:
        run_warmup_requests(
            client,
            url=args.url,
            service=service,
            warmup_requests=args.warmup_requests,
        )
        client_latencies_ms, api_latencies_ms = run_measured_requests(
            client,
            url=args.url,
            service=service,
            request_count=args.requests,
        )

    p50_latency_ms = percentile(client_latencies_ms, 50)
    p95_latency_ms = percentile(client_latencies_ms, 95)
    max_latency_ms = max(client_latencies_ms)
    mean_latency_ms = statistics.fmean(client_latencies_ms)
    api_p95_latency_ms = percentile(api_latencies_ms, 95)

    pass_latency_budget = p95_latency_ms < args.budget_ms

    print(f"request_count={args.requests}")
    print(f"warmup_request_count={args.warmup_requests}")
    print(f"model_version={service.artifacts.model_version}")
    print(f"feature_count={len(service.feature_names)}")
    print(f"budget_ms={args.budget_ms:.3f}")
    print(f"p50_latency_ms={p50_latency_ms:.3f}")
    print(f"p95_latency_ms={p95_latency_ms:.3f}")
    print(f"max_latency_ms={max_latency_ms:.3f}")
    print(f"mean_latency_ms={mean_latency_ms:.3f}")
    print(f"api_p95_latency_ms={api_p95_latency_ms:.3f}")
    print(f"pass_latency_budget={format_bool(pass_latency_budget)}")

    return 0 if pass_latency_budget else 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except LatencyBenchmarkError as exc:
        print(f"benchmark_error={exc}", file=sys.stderr)
        raise SystemExit(1)
