from __future__ import annotations

from fastapi import APIRouter

from app.api.v1.routes.admin import router as admin_router
from app.api.v1.routes.corpora import router as corpora_router
from app.api.v1.routes.generation import router as generation_router
from app.api.v1.routes.health import router as health_router
from app.api.v1.routes.ingest import router as ingest_router
from app.api.v1.routes.jobs import router as jobs_router
from app.api.v1.routes.niyukti_compat import router as niyukti_compat_router
from app.api.v1.routes.school import router as school_router


router = APIRouter()
router.include_router(health_router, tags=["health"])
router.include_router(corpora_router, prefix="/corpora", tags=["corpora"])
router.include_router(jobs_router, prefix="/jobs", tags=["jobs"])
router.include_router(generation_router, prefix="/question-generation", tags=["question-generation"])
router.include_router(ingest_router, prefix="/ingest", tags=["ingest"])
router.include_router(admin_router, prefix="/admin", tags=["admin"])
# Niyukti compatibility endpoint - matches expected API structure
router.include_router(niyukti_compat_router, tags=["niyukti-compat"])
# School self-service question generation (multitenant, scoped by X-Tenant-ID)
router.include_router(school_router, prefix="/school")


