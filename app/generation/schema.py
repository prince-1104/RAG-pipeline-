from __future__ import annotations

QUESTION_SET_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["questions", "rejected"],
    "properties": {
        "questions": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["id", "qtype", "difficulty", "language", "stem", "citations", "tags"],
                "properties": {
                    "id": {"type": "string"},
                    "qtype": {"type": "string", "enum": ["mcq", "short", "long", "case_based", "assertion_reason"]},
                    "difficulty": {"type": "string", "enum": ["easy", "medium", "hard"]},
                    "language": {"type": "string", "enum": ["en", "hi"]},
                    "stem": {"type": "string", "minLength": 10},
                    "options": {"type": ["array", "null"], "items": {"type": "string"}},
                    "answer": {"type": ["string", "null"]},
                    "explanation": {"type": ["string", "null"]},
                    "tags": {"type": "object"},
                    "citations": {
                        "type": "array",
                        "minItems": 1,
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "required": ["chunk_id", "document_id", "storage_uri", "page_start", "page_end", "excerpt"],
                            "properties": {
                                "chunk_id": {"type": "string"},
                                "document_id": {"type": "string"},
                                "storage_uri": {"type": "string"},
                                "page_start": {"type": "integer", "minimum": 1},
                                "page_end": {"type": "integer", "minimum": 1},
                                "excerpt": {"type": "string", "minLength": 10},
                            },
                        },
                    },
                },
            },
        },
        "rejected": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": True,
                "required": ["reason"],
                "properties": {"reason": {"type": "string"}},
            },
        },
    },
}


