from __future__ import annotations

import time

from fastapi import APIRouter, Request, Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest


metrics_router = APIRouter()

HTTP_REQUESTS = Counter(
    "qgen_http_requests_total",
    "Total HTTP requests",
    labelnames=("method", "route", "status"),
)
HTTP_LATENCY = Histogram(
    "qgen_http_request_latency_seconds",
    "HTTP request latency",
    labelnames=("method", "route"),
    buckets=(0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
)


def _route_template(request: Request) -> str:
    route = request.scope.get("route")
    if route and getattr(route, "path", None):
        return str(route.path)
    return request.url.path


def observe_request(request: Request, duration_s: float) -> None:
    route = _route_template(request)
    method = request.method
    status = str(getattr(request.state, "status_code", 200))
    HTTP_REQUESTS.labels(method=method, route=route, status=status).inc()
    HTTP_LATENCY.labels(method=method, route=route).observe(duration_s)


@metrics_router.get("/metrics")
def metrics() -> Response:
    start = time.perf_counter()
    payload = generate_latest()
    # keep the endpoint itself from skewing latency histograms (it still records counters)
    _ = time.perf_counter() - start
    return Response(content=payload, media_type=CONTENT_TYPE_LATEST)


