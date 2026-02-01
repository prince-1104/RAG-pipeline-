from __future__ import annotations

import logging
import os
import sys

import structlog


def configure_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(message)s",
        stream=sys.stdout,
    )
    structlog.configure(
        processors=[
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.add_log_level,
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer(),
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )


log = structlog.get_logger()


def get_process_rss_mb() -> float | None:
    try:
        import resource

        rss_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        if sys.platform == "darwin":
            rss_kb = rss_kb / 1024
        return rss_kb / 1024
    except Exception:
        pass

    try:
        with open("/proc/self/statm", "r", encoding="utf-8") as handle:
            parts = handle.read().strip().split()
        if len(parts) >= 2:
            rss_pages = int(parts[1])
            page_size = os.sysconf("SC_PAGE_SIZE")
            return (rss_pages * page_size) / (1024 * 1024)
    except Exception:
        return None

    return None


