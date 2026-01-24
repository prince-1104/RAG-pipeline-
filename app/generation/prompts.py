from __future__ import annotations

from app.api.v1.schemas.generation import Blueprint
from app.retrieval.retriever import RetrievalResult


def build_system_prompt() -> str:
    return (
        "You are an exam question writer for CBSE.\n"
        "Hard rules:\n"
        "- Use ONLY the provided CONTEXT chunks.\n"
        "- If context is insufficient, output zero questions and add an item to rejected with reason=INSUFFICIENT_CONTEXT.\n"
        "- Every question MUST include >=1 citation that points to a chunk/page range and includes a verbatim excerpt.\n"
        "- Output MUST be valid JSON matching the EXACT schema provided. No markdown, no extra text, NO additional fields.\n"
        "- Do NOT include fields like 'chapter' or 'correct_option' at the question level - use 'tags.chapter' and 'answer' instead.\n"
        "- Do not invent facts, formulas, definitions, names, or numbers.\n"
        "- Generate EXACTLY the number of questions specified in the blueprint 'count' field.\n"
        "- Distribute difficulty levels according to 'difficulty_mix' (easy, medium, hard counts).\n"
        "- Only use question types listed in 'question_types' array.\n"
        "- For MCQ questions, provide exactly 4 options in the 'options' array as SIMPLE STRINGS.\n"
        "  Example: \"options\": [\"Text of option A\", \"Text of option B\", \"Text of option C\", \"Text of option D\"]\n"
        "  DO NOT use objects like {\"id\": \"opt1\", \"text\": \"...\"} - the schema requires plain strings.\n"
        "- Ensure questions cover the specified chapters and topics from the blueprint.\n"
        "- Allowed fields per question: id, qtype, difficulty, language, stem, options, answer, explanation, tags, citations.\n"
        "- Do NOT add any other fields - they will cause validation errors.\n"
    )


def build_user_prompt(*, blueprint: Blueprint, retrieval: RetrievalResult, document_uri_by_id: dict[str, str]) -> str:
    """
    We include citations metadata inside the context so the model can point to them deterministically.
    """
    ctx_lines: list[str] = []
    for r in retrieval.chunks:
        c = r.chunk
        uri = document_uri_by_id.get(c.document_id, "unknown")
        # Optimize: Reduce text length to speed up processing (1500 chars instead of 2000)
        # Excerpt is already included, so we don't need full 2000 chars
        chunk_text = c.text[:1500] if len(c.text) > 1500 else c.text
        excerpt = c.text[:200].replace("\n", " ")  # Shorter excerpt (200 instead of 240)
        ctx_lines.extend(
            [
                "----",
                f"CHUNK_ID: {c.id}",
                f"DOCUMENT_ID: {c.document_id}",
                f"STORAGE_URI: {uri}",
                f"PAGES: {c.page_start}-{c.page_end}",
                f"CHAPTER: {c.chapter}",
                f"EXCERPT: {excerpt}",
                f"TEXT: {chunk_text}",
            ]
        )

    # Build explicit requirements summary
    total_count = blueprint.count
    difficulty_mix = blueprint.difficulty_mix
    qtypes = blueprint.question_types
    chapters = blueprint.constraints.chapters
    
    requirements = [
        f"Generate EXACTLY {total_count} questions.",
        f"Question types allowed: {', '.join(qtypes)}.",
        f"Difficulty distribution: {difficulty_mix.easy} easy, {difficulty_mix.medium} medium, {difficulty_mix.hard} hard.",
        f"Chapters to cover: {', '.join(chapters)}.",
        "Each question must:",
        "  - Have a unique 'id' (UUID string)",
        "  - Match one of the allowed question_types",
        "  - Have difficulty matching the distribution above",
        "  - Include at least one citation with chunk_id, document_id, storage_uri, page_start, page_end, and excerpt",
        "  - For MCQ: provide exactly 4 options in 'options' array as SIMPLE STRINGS (not objects)",
        "    Example: \"options\": [\"Option A text\", \"Option B text\", \"Option C text\", \"Option D text\"]",
        "    DO NOT use objects like {\"id\": \"opt1\", \"text\": \"...\"} - use plain strings only",
        "  - For short/long: provide 'answer' and 'explanation'",
    ]
    
    return "\n".join(
        [
            "BLUEPRINT_JSON:",
            blueprint.model_dump_json(),
            "",
            "REQUIREMENTS:",
            "\n".join(requirements),
            "",
            "CONTEXT:",
            "\n".join(ctx_lines),
            "",
            "TASK:",
            f"Generate exactly {total_count} questions following the requirements above.",
            "Ensure the difficulty mix matches: {easy} easy, {medium} medium, {hard} hard.".format(
                easy=difficulty_mix.easy, medium=difficulty_mix.medium, hard=difficulty_mix.hard
            ),
            "Return JSON with keys: questions (array), rejected (array).",
            "If you cannot generate enough questions, add rejected items with reason=INSUFFICIENT_CONTEXT.",
        ]
    )


