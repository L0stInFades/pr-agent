"""Maps MOSAICO's env-var contract onto pr-agent's Dynaconf settings and registers
the Langfuse litellm callback. Every function is a no-op unless the corresponding
MOSAICO env var is set: importing or calling apply_mosaico_env() with no MOSAICO env
present changes nothing."""
import os

from pr_agent.config_loader import get_settings
from pr_agent.log import get_logger

ENV_API_BASE = "API_BASE"
ENV_API_KEY = "API_KEY"
ENV_MODEL_NAME = "MODEL_NAME"
ENV_MODEL_MAX_TOKENS = "MODEL_MAX_TOKENS"
ENV_LANGFUSE_HOST = "LANGFUSE_HOST"
# Token budget for a MOSAICO model not in pr-agent's built-in MAX_TOKENS table.
DEFAULT_CUSTOM_MODEL_MAX_TOKENS = 32000
MINIMAX_M3_MODEL = "MiniMax-M3"
MINIMAX_M3_MAX_TOKENS = 1000000
MINIMAX_API_BASE_MARKERS = ("api.minimax.io", "api.minimaxi.com")
ENV_LANGFUSE_PUBLIC_KEY = "LANGFUSE_PUBLIC_KEY"
ENV_LANGFUSE_SECRET_KEY = "LANGFUSE_SECRET_KEY"
LANGFUSE_CALLBACK_NAME = "langfuse_otel"
LEGACY_LANGFUSE_CALLBACK = "langfuse"


def langfuse_env_present() -> bool:
    return bool(
        os.getenv(ENV_LANGFUSE_HOST)
        and os.getenv(ENV_LANGFUSE_PUBLIC_KEY)
        and os.getenv(ENV_LANGFUSE_SECRET_KEY)
    )


def _is_minimax_api_base(api_base: str | None) -> bool:
    return bool(api_base and any(marker in api_base for marker in MINIMAX_API_BASE_MARKERS))


def _normalize_model_name(model_name: str, api_base: str | None) -> str:
    if "/" in model_name:
        return model_name
    if _is_minimax_api_base(api_base) and model_name.startswith("MiniMax-"):
        return f"minimax/{model_name}"
    return f"openai/{model_name}"


def _is_minimax_m3_model(model: str) -> bool:
    return model.endswith(f"/{MINIMAX_M3_MODEL}")


def apply_mosaico_env() -> None:
    """Idempotent. Call once at MOSAICO-server startup, BEFORE LiteLLMAIHandler() is
    constructed. Does nothing when MOSAICO env is absent."""
    settings = get_settings()
    api_base = os.getenv(ENV_API_BASE)
    api_key = os.getenv(ENV_API_KEY)
    model_name = os.getenv(ENV_MODEL_NAME)

    if api_base:
        settings.set("OPENAI.API_BASE", api_base)
        if _is_minimax_api_base(api_base):
            settings.set("MINIMAX.API_BASE", api_base)
    if api_key:
        settings.set("OPENAI.KEY", api_key)
        if _is_minimax_api_base(api_base):
            settings.set("MINIMAX.KEY", api_key)
    if model_name:
        model = _normalize_model_name(model_name, api_base)
        settings.set("CONFIG.MODEL", model)
        settings.set("CONFIG.FALLBACK_MODELS", [])
        # MOSAICO models are not in pr-agent's built-in MAX_TOKENS table; declare a budget
        # so reviews don't fail with "not defined in MAX_TOKENS". Overridable via env.
        max_tokens_env = os.getenv(ENV_MODEL_MAX_TOKENS)
        default_max_tokens = (
            MINIMAX_M3_MAX_TOKENS if _is_minimax_m3_model(model) else DEFAULT_CUSTOM_MODEL_MAX_TOKENS
        )
        try:
            custom_max_tokens = int(max_tokens_env) if max_tokens_env else default_max_tokens
        except ValueError:
            custom_max_tokens = default_max_tokens
        if custom_max_tokens <= 0:
            get_logger().warning(
                f"MOSAICO: MODEL_MAX_TOKENS={custom_max_tokens!r} is non-positive; "
                f"falling back to {default_max_tokens}."
            )
            custom_max_tokens = default_max_tokens
        settings.set("CONFIG.CUSTOM_MODEL_MAX_TOKENS", custom_max_tokens)
        if _is_minimax_m3_model(model):
            settings.set("CONFIG.MAX_MODEL_TOKENS", custom_max_tokens)

    if langfuse_env_present():
        _register_langfuse_callback(settings)
    else:
        get_logger().info("MOSAICO: Langfuse env not fully set; LLM-call tracing disabled.")


def _register_langfuse_callback(settings) -> None:
    for key in ("LITELLM.SUCCESS_CALLBACK", "LITELLM.FAILURE_CALLBACK"):
        # Drop the legacy 'langfuse' callback (incompatible with langfuse 3.x -> sdk_integration
        # TypeError) and ensure exactly one 'langfuse_otel'.
        current = [c for c in (settings.get(key, []) or []) if c != LEGACY_LANGFUSE_CALLBACK]
        if LANGFUSE_CALLBACK_NAME not in current:
            current.append(LANGFUSE_CALLBACK_NAME)
        settings.set(key, current)
    settings.set("LITELLM.ENABLE_CALLBACKS", True)
    get_logger().info("MOSAICO: registered Langfuse litellm callback.")
