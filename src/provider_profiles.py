"""Explicit provider profiles for teacher and student runtime isolation."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Mapping, Optional

@dataclass(frozen=True)
class ProviderProfile:
    profile_id: str
    provider: str
    default_model: Optional[str]
    model_env: Optional[str]
    vlm_provider: str
    default_vlm_model: Optional[str] = None
    vlm_model_env: Optional[str] = None
    llm_wire_api: str = "chat_completions"
    vlm_wire_api: str = "chat_completions"

@dataclass(frozen=True)
class ResolvedProviderSettings:
    profile_id: Optional[str]
    provider: str
    model_name: str
    vlm_provider: str
    vlm_model: str
    llm_wire_api: Optional[str]
    vlm_wire_api: Optional[str]

PROVIDER_PROFILES = {
    "teacher-gemini": ProviderProfile(
        profile_id="teacher-gemini",
        provider="gemini",
        default_model="gemini-3.5-flash",
        model_env=None,
        vlm_provider="gemini",
        default_vlm_model="gemini-3.5-flash",
        llm_wire_api="interactions",
        vlm_wire_api="interactions",
    ),
    "student-qwen-local": ProviderProfile(
        profile_id="student-qwen-local",
        provider="lmdeploy",
        default_model="/gsdata/home/wza/models/Qwen3-VL-8B-Thinking",
        model_env="LMDEPLOY_MODEL",
        vlm_provider="lmdeploy",
        llm_wire_api="chat_completions",
        vlm_wire_api="chat_completions",
    ),
    "student-qwen-api": ProviderProfile(
        profile_id="student-qwen-api",
        provider="qwen",
        default_model=None,
        model_env="QWEN_API_MODEL",
        vlm_provider="qwen",
        vlm_model_env="QWEN_API_VISION_MODEL",
        llm_wire_api="chat_completions",
        vlm_wire_api="chat_completions",
    ),
}
PROFILE_IDS = tuple(PROVIDER_PROFILES)

def _clean(value: Optional[str]) -> Optional[str]:
    rendered = str(value or "").strip()
    return rendered or None

def _profile_model(
    profile: ProviderProfile,
    environ: Mapping[str, str],
    *,
    vision: bool = False,
    fallback: Optional[str] = None,
 ) -> str:
    env_name = profile.vlm_model_env if vision else profile.model_env
    default = profile.default_vlm_model if vision else profile.default_model
    value = _clean(environ.get(env_name, "")) if env_name else None
    value = value or _clean(default) or _clean(fallback)
    if not value:
        role = "vision model" if vision else "model"
        requirement = env_name or "an explicit profile model"
        raise ValueError(
            f"provider profile {profile.profile_id!r} requires {role} via {requirement}"
        )
    return value

def resolve_provider_settings(
    *,
    profile_id: Optional[str] = None,
    provider: Optional[str] = None,
    model_name: Optional[str] = None,
    vlm_provider: Optional[str] = None,
    vlm_model: Optional[str] = None,
    llm_wire_api: Optional[str] = None,
    vlm_wire_api: Optional[str] = None,
    environ: Optional[Mapping[str, str]] = None,
 ) -> ResolvedProviderSettings:
    """Resolve one profile or the legacy loose settings, never both."""

    normalized_profile = _clean(profile_id)
    if normalized_profile:
        conflicts = [
            name
            for name, value in {
                "provider": provider,
                "model_name": model_name,
                "vlm_provider": vlm_provider,
                "vlm_model": vlm_model,
                "llm_wire_api": llm_wire_api,
                "vlm_wire_api": vlm_wire_api,
            }.items()
            if value is not None
        ]
        if conflicts:
            raise ValueError(
                "provider profile cannot be combined with loose overrides: "
                + ", ".join(conflicts)
            )
        try:
            profile = PROVIDER_PROFILES[normalized_profile]
        except KeyError as exc:
            raise ValueError(
                f"unknown provider profile {normalized_profile!r}; "
                f"expected one of {', '.join(PROFILE_IDS)}"
            ) from exc
        active_env = environ if environ is not None else os.environ
        resolved_model = _profile_model(profile, active_env)
        resolved_vlm_model = _profile_model(
            profile,
            active_env,
            vision=True,
            fallback=resolved_model,
        )
        return ResolvedProviderSettings(
            profile_id=profile.profile_id,
            provider=profile.provider,
            model_name=resolved_model,
            vlm_provider=profile.vlm_provider,
            vlm_model=resolved_vlm_model,
            llm_wire_api=profile.llm_wire_api,
            vlm_wire_api=profile.vlm_wire_api,
        )

    resolved_provider = _clean(provider) or "gemini"
    resolved_model = _clean(model_name) or "gemini-3.5-flash"
    return ResolvedProviderSettings(
        profile_id=None,
        provider=resolved_provider.lower(),
        model_name=resolved_model,
        vlm_provider=(_clean(vlm_provider) or resolved_provider).lower(),
        vlm_model=_clean(vlm_model) or resolved_model,
        llm_wire_api=_clean(llm_wire_api),
        vlm_wire_api=_clean(vlm_wire_api),
    )
