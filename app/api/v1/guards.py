from __future__ import annotations

from fastapi import HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import Corpus


async def assert_corpus_owner(corpus_id: str, tenant_id: str, session: AsyncSession) -> Corpus:
    """Verify the corpus exists and belongs to the given tenant."""
    corpus = await session.get(Corpus, corpus_id)
    if not corpus:
        raise HTTPException(status_code=404, detail="Corpus not found")
    if corpus.tenant_id != tenant_id:
        raise HTTPException(status_code=403, detail="This corpus does not belong to your tenant")
    return corpus
