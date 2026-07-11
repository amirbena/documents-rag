"""Confirms every Settings field alias documented in .env.example actually exists on Settings.

Guards against documentation drift — a stale or invented environment variable name in
.env.example (e.g. a rename that wasn't reflected everywhere) would otherwise go unnoticed until
someone hit it in practice.
"""

from pathlib import Path

from app.core.config import Settings

_ENV_EXAMPLE_PATH = Path(__file__).resolve().parents[1] / ".env.example"


def _env_example_keys() -> set[str]:
    keys = set()
    for line in _ENV_EXAMPLE_PATH.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        keys.add(stripped.split("=", 1)[0])
    return keys


def _settings_alias_keys() -> set[str]:
    return {field.alias for field in Settings.model_fields.values() if field.alias is not None}


def test_every_env_example_key_is_a_real_settings_alias() -> None:
    """Every KEY= line in .env.example must correspond to a real Settings field alias."""
    unknown_keys = _env_example_keys() - _settings_alias_keys()

    assert not unknown_keys, f"Unknown/stale env vars in .env.example: {sorted(unknown_keys)}"


def test_bge_m3_is_referenced_as_the_default_embedding_model_in_env_example() -> None:
    """.env.example must document the same default embedding model Settings actually uses."""
    settings = Settings()
    env_example_text = _ENV_EXAMPLE_PATH.read_text(encoding="utf-8")

    assert f"OLLAMA_EMBEDDING_MODEL={settings.ollama_embedding_model}" in env_example_text
