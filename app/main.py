from __future__ import annotations

import os
import time

import uvicorn

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import ORJSONResponse
from fastapi.middleware.cors import CORSMiddleware
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware

from app.api.v1.middleware.api_key_auth import APIKeyMiddleware
from app.api.v1.router import router as v1_router
from app.db.session import init_db
from app.observability.logging import configure_logging, log
from app.observability.metrics import metrics_router, observe_request
from app.observability.rate_limit import limiter
from app.settings import get_settings


def create_app() -> FastAPI:
    settings = get_settings()
    configure_logging(settings.log_level)

    limiter.default_limits = [f"{settings.requests_per_minute}/minute"]

    # Use JSONResponse as default instead of ORJSONResponse to avoid serialization issues
    from fastapi.responses import JSONResponse
    app = FastAPI(
        title=settings.service_name,
        version="1.0.0",
        default_response_class=JSONResponse,
        openapi_url="/openapi.json",
    )

    # CORS middleware for niyukti platform integration
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # In production, specify exact origins
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.state.limiter = limiter
    app.add_middleware(SlowAPIMiddleware)
    
    # API Key authentication middleware (optional, enabled if API_KEY is set)
    app.add_middleware(APIKeyMiddleware)

    @app.exception_handler(RateLimitExceeded)
    async def _rate_limit_handler(request: Request, exc: RateLimitExceeded):
        return ORJSONResponse(status_code=429, content={"error": {"code": "RATE_LIMITED", "message": str(exc)}})

    @app.exception_handler(HTTPException)
    async def _http_exception_handler(request: Request, exc: HTTPException):
        """Ensure HTTPExceptions return proper status codes and details"""
        # Ensure detail is serializable - convert to string if needed
        try:
            detail = exc.detail
            if not isinstance(detail, (str, dict, list)):
                detail = str(detail)
            # Limit length to avoid serialization issues
            if isinstance(detail, str) and len(detail) > 1000:
                detail = detail[:1000] + "..."
        except Exception:
            detail = "internal_server_error"
        
        # Use JSONResponse for better compatibility - with absolute minimal content
        from fastapi.responses import JSONResponse
        try:
            return JSONResponse(status_code=exc.status_code, content={"detail": detail})
        except Exception:
            # Ultimate fallback - return plain text if JSON fails
            from fastapi.responses import Response
            return Response(
                status_code=exc.status_code,
                content=f"Error: {str(detail)[:200]}",
                media_type="text/plain"
            )

    @app.exception_handler(Exception)
    async def _global_exception_handler(request: Request, exc: Exception):
        """Catch any unhandled exceptions and return proper error response"""
        import traceback
        error_trace = traceback.format_exc()
        try:
            log.exception("unhandled_exception", path=request.url.path, method=request.method, error=str(exc), error_type=type(exc).__name__, traceback=error_trace)
        except Exception:
            pass
        
        # Use JSONResponse for better compatibility - with absolute minimal content
        from fastapi.responses import JSONResponse
        try:
            error_msg = str(exc)[:500]  # Limit length
            error_type = type(exc).__name__
            return JSONResponse(
                status_code=500,
                content={"detail": f"internal_server_error: {error_msg}", "error_type": error_type}
            )
        except Exception:
            # Ultimate fallback - return plain text if JSON fails
            from fastapi.responses import Response
            try:
                error_msg = str(exc)[:200]
            except Exception:
                error_msg = "unknown_error"
            return Response(
                status_code=500,
                content=f"Error: {error_msg}",
                media_type="text/plain"
            )

    @app.exception_handler(RequestValidationError)
    async def _validation_error_handler(request: Request, exc: RequestValidationError):
        from fastapi.responses import JSONResponse
        return JSONResponse(status_code=422, content={"detail": exc.errors()})

    @app.middleware("http")
    async def _metrics_mw(request: Request, call_next):
        start = time.perf_counter()
        try:
            response = await call_next(request)
            request.state.status_code = response.status_code
            return response
        except Exception as e:
            request.state.status_code = 500
            import sys
            import traceback
            error_trace = traceback.format_exc()
            print("=" * 80, file=sys.stderr)
            print(f"MIDDLEWARE EXCEPTION: {type(e).__name__}: {str(e)}", file=sys.stderr)
            print(error_trace, file=sys.stderr)
            sys.stderr.flush()
            raise
        finally:
            observe_request(request, time.perf_counter() - start)

    @app.on_event("startup")
    def _startup() -> None:
        init_db()
        log.info("startup_complete", env=settings.env, provider=settings.llm_provider)

    app.include_router(metrics_router, include_in_schema=False)
    app.include_router(v1_router, prefix="/v1")

    # Runtime RAG API: POST /rag/generate/questions (Qdrant + LLM)
    from app.api.v1.routes.rag import router as rag_router
    app.include_router(rag_router, prefix="/rag", tags=["rag"])

    # Niyukti compatibility endpoint at /api/v1 (matches their expected path)
    from app.api.v1.routes.niyukti_compat import router as niyukti_compat_router
    app.include_router(niyukti_compat_router, prefix="/api/v1", tags=["niyukti-compat"])

    return app


app = create_app()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=port,
    )
