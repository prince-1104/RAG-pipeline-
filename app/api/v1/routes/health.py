from __future__ import annotations

from fastapi import APIRouter

from app.settings import get_settings


router = APIRouter()


@router.get("/healthz")
@router.get("/health")  # Alias for convenience
def healthz():
    s = get_settings()
    # Include minimal debug info (safe: do not expose secrets)
    from app.llm.factory import get_llm_client
    llm_client_type = None
    try:
        client = get_llm_client()
        llm_client_type = type(client).__name__
    except Exception:
        pass
    
    gemini_tail = None
    if s.gemini_api_key:
        gemini_tail = s.gemini_api_key[-4:]
    
    openrouter_tail = None
    if s.openrouter_api_key:
        openrouter_tail = s.openrouter_api_key[-4:]
    
    # Include AgentBill status for diagnostics
    agentbill_status = None
    try:
        from app.observability.agentbill import get_agentbill_status
        agentbill_status = get_agentbill_status()
    except Exception:
        pass
    
    return {
        "status": "ok",
        "service": s.service_name,
        "env": s.env,
        "llm_provider": s.llm_provider,
        "llm_client_type": llm_client_type,
        "gemini_model": s.gemini_model,
        "gemini_api_key_tail": gemini_tail,
        "openrouter_model": s.openrouter_model,
        "openrouter_api_key_tail": openrouter_tail,
        "agentbill": agentbill_status,
        "health_file": __file__,
    }


