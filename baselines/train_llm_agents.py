"""
baselines/train_llm_agents.py  —  LLM Agent Utilities
=====================================================
Utilities for LLM-based control agents including:
  - Prompt template loading
  - State-to-dict conversion with semantic naming
  - JSON extraction from LLM outputs
  - LLMPolicyAgent class for hardware and macro control
"""
from __future__ import annotations

import json
import os
import re
import urllib.request
import warnings
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import numpy as np
import yaml

from c2g_env import C2GFastEnv


# ──────────────────────────────────────────────────────────────────
# LLM Agent Helper Functions
# ──────────────────────────────────────────────────────────────────

def extract_json(
    text: str,
    field_order: list[str] | None = None,
    bounds: dict[str, tuple[float, float]] | None = None,
    previous_by_field: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Extract and sanitize JSON fields with optional per-field fallback.

    If ``field_order`` is provided, each field is converted to a finite float.
    Invalid values (including NaN/Inf) reuse ``previous_by_field[field]`` when
    available; otherwise a ValueError is raised to terminate the episode cleanly.
    """
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    raw: dict[str, Any] = {}
    if match:
        try:
            raw = json.loads(match.group(0))
        except json.JSONDecodeError:
            pass

    if field_order is None:
        return raw

    if not raw and previous_by_field:
        warnings.warn(
            "LLM returned no parseable JSON; falling back to previous action values.",
            RuntimeWarning,
            stacklevel=3,
        )

    out: dict[str, float] = {}
    for field in field_order:
        parsed = safe_float_or_none(raw.get(field))
        if parsed is None:
            if previous_by_field is not None and field in previous_by_field:
                parsed = float(previous_by_field[field])
            else:
                raise ValueError(
                    f"LLM field '{field}' is missing/invalid and no previous value is available."
                )

        low, high = bounds[field] if bounds and field in bounds else (-np.inf, np.inf)
        out[field] = float(np.clip(parsed, low, high))

    return out


def safe_float_or_none(value: Any) -> float | None:
    """Safely convert value to float; return None for invalid/non-finite values."""
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if np.isfinite(out) else None


def obs_to_dict(obs: np.ndarray, state_names: list[str]) -> dict[str, float]:
    """Convert observation array to semantic state dictionary."""
    arr = np.asarray(obs, dtype=np.float32).reshape(-1)
    obs_dict: dict[str, float] = {}
    for idx, value in enumerate(arr):
        key = state_names[idx] if idx < len(state_names) else f"obs_{idx}"
        obs_dict[key] = float(value)
    return obs_dict


def build_prompt(
    mode_prompts: dict[str, str],
    state_dict: dict[str, float],
    scenario: str,
    env_context: dict[str, Any] | None = None,
) -> str:
    """Build full prompt from system and user prompts with state context.

    ``env_context`` is an optional dict of environment-derived values (e.g.
    ``committed_mw_max``, ``dr_baseline_mw``) injected as extra format variables
    into the user prompt template.
    """
    system_prompt = mode_prompts["system"]
    fmt_kwargs: dict[str, Any] = {
        "scenario": scenario,
        "state_json": json.dumps(state_dict),
    }
    if env_context:
        fmt_kwargs.update(env_context)
    user_prompt = mode_prompts["user"].format(**fmt_kwargs)
    return f"{system_prompt}\n\n{user_prompt}"


def generate_structured(
    client: Any,
    model_name: str,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    field_order: list[str] | None = None,
    bounds: dict[str, tuple[float, float]] | None = None,
    previous_by_field: dict[str, float] | None = None,
    enable_thinking: bool = True,
) -> dict[str, Any]:
    """Generate structured JSON response from LLM via vLLM server (OpenAI-compatible API).

    vLLM has no per-request thinking-budget parameter. We cap total output tokens
    (max_new_tokens) as an indirect limit: the model fills <think> first, so
    max_new_tokens ≈ thinking_budget + json_headroom .
    When enable_thinking=False the <think> block is suppressed; max_new_tokens
    can then be set much smaller (~128 tokens for JSON-only output).
    """
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "user", "content": prompt}
            ],
            max_tokens=max(1, int(max_new_tokens)),
            temperature=max(0.0, float(temperature)),
            extra_body={
                "chat_template_kwargs": {
                    "enable_thinking": enable_thinking,
                    "thinking_budget": 12000,
                }
            },
        )
        choice = response.choices[0]
        if choice.finish_reason == "length":
            warnings.warn(
                f"LLM response was truncated (finish_reason='length'). "
                "Increase --llm-max-new-tokens to fit thinking + JSON output.",
                RuntimeWarning,
                stacklevel=2,
            )
        text = choice.message.content or ""
        print(f"[LLM RAW] {text}", flush=True)
        # Strip <think>…</think> blocks emitted by reasoning models (e.g. Qwen3, QwQ)
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
        
    except Exception as exc:
        warnings.warn(
            f"vLLM server call failed: {exc}",
            RuntimeWarning,
            stacklevel=2,
        )
        text = ""
    return extract_json(
        text,
        field_order=field_order,
        bounds=bounds,
        previous_by_field=previous_by_field,
    )


def load_prompt_templates(template_path: str | Path) -> dict[str, dict[str, str]]:
    """Load system and user prompts for hardware and macro agents."""
    path = Path(template_path)
    if not path.is_absolute():
        path = Path(__file__).resolve().parent.parent / path
    if not path.exists():
        raise FileNotFoundError(f"Prompt template file not found: {path}")

    with open(path, encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}

    # Load hardware prompts
    hw_system = str(data.get("hardware_system_prompt", "")).strip()
    hw_user = str(data.get("hardware_user_prompt", "")).strip()
    if not hw_system or not hw_user:
        raise ValueError(
            "Prompt template YAML must define non-empty 'hardware_system_prompt' "
            "and 'hardware_user_prompt' keys."
        )

    # Load macro prompts
    macro_system = str(data.get("macro_system_prompt", "")).strip()
    macro_user = str(data.get("macro_user_prompt", "")).strip()
    if not macro_system or not macro_user:
        raise ValueError(
            "Prompt template YAML must define non-empty 'macro_system_prompt' "
            "and 'macro_user_prompt' keys."
        )

    return {
        "hardware": {"system": hw_system, "user": hw_user},
        "macro": {"system": macro_system, "user": macro_user},
    }


def probe_api_base(api_base: str, timeout: float = 5.0) -> None:
    """Verify the API endpoint is reachable by hitting /models.

    Raises ``ConnectionError`` with a human-readable message on failure.
    """
    url = api_base.rstrip("/") + "/models"
    try:
        with urllib.request.urlopen(url, timeout=timeout):  # noqa: S310
            pass
    except Exception as exc:
        raise ConnectionError(
            f"Cannot reach LLM API at {url!r}. "
            "Check that the server is running and --llm-api-base is correct.\n"
            f"  ({type(exc).__name__}: {exc})"
        ) from exc


def validate_llm_model_id(model_id: str) -> str:
    """
    Validate and normalise a model identifier.

    Accepted forms:
      - Hugging Face repo id:  ``org/model``
      - Hugging Face URL:      ``https://huggingface.co/org/model``
      - Ollama-style tag:      ``name:tag``  (e.g. ``qwen3:4b``)
      - Bare name:             ``modelname``  (passed through as-is)

    Returns the model id string as the backend expects it.
    """
    model_ref = str(model_id).strip()
    if not model_ref:
        raise ValueError("--llm-model-id must be non-empty.")

    if " " in model_ref:
        raise ValueError("--llm-model-id must not contain spaces.")

    # Only parse as URL if it contains "://" (avoids treating "name:tag" as a scheme)
    if "://" in model_ref:
        parsed = urlparse(model_ref)
        if parsed.scheme not in {"http", "https"}:
            raise ValueError("--llm-model-id URL scheme must be http or https.")
        if parsed.netloc not in {"huggingface.co", "www.huggingface.co"}:
            raise ValueError("--llm-model-id URL must point to huggingface.co.")
        repo_id = parsed.path.strip("/")
        if repo_id.count("/") != 1:
            raise ValueError(
                "--llm-model-id URL must contain exactly one path segment (e.g. org/model)."
            )
        return repo_id

    # Bare id: org/model, name:tag, or plain name — pass through as-is
    return model_ref


# ──────────────────────────────────────────────────────────────────
# LLM Policy Agent
# ──────────────────────────────────────────────────────────────────


class _BaseLLMPolicyAgent:
    """Shared client/prompt plumbing for LLM-driven policy agents."""

    uses_env_context = True

    def __init__(
        self,
        model_id: str,
        prompts: dict[str, dict[str, str]],
        state_names: list[str],
        max_new_tokens: int = 12000,
        temperature: float = 0.0,
        api_base: str = "http://localhost:8000/v1",
        enable_thinking: bool = True,
    ):
        probe_api_base(api_base)

        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ImportError(
                "openai is required for llm agent. Install with: pip install openai"
            ) from exc

        self._client = OpenAI(
            api_key=os.environ.get("HF_TOKEN") or os.environ.get("OPENAI_API_KEY") or "not-needed",
            base_url=api_base,
        )
        self._model_name = model_id
        self._prompts = prompts
        self._state_names = state_names
        self._max_new_tokens = int(max_new_tokens)
        self._temperature = float(temperature)
        self._enable_thinking: dict[str, bool] = {
            "hardware": enable_thinking,
            "macro": enable_thinking,
        }
        self._previous_action: np.ndarray | None = None

    def _generate_payload(
        self,
        obs: np.ndarray,
        scenario: str,
        mode: str,
        field_order: list[str],
        bounds: dict[str, tuple[float, float]],
        previous_by_field: dict[str, float] | None,
        env_context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        obs_dict = obs_to_dict(obs, self._state_names)
        prompt = build_prompt(self._prompts[mode], obs_dict, scenario, env_context=env_context)
        return generate_structured(
            self._client,
            self._model_name,
            prompt,
            self._max_new_tokens,
            self._temperature,
            field_order=field_order,
            bounds=bounds,
            previous_by_field=previous_by_field,
            enable_thinking=self._enable_thinking.get(mode, True),
        )


class HardwareLLMPolicyAgent(_BaseLLMPolicyAgent):
    """Hardware-level LLM policy that emits 4-D low-level actions."""

    def predict(
        self,
        obs: np.ndarray,
        deterministic: bool = True,
        env: C2GFastEnv | None = None,
        scenario: str = "default",
    ):
        previous_by_field = None
        if self._previous_action is not None:
            previous_by_field = {
                "throttle_batch": float(self._previous_action[0]),
                "pump_speed_A": float(self._previous_action[1]),
                "hvac_effort": float(self._previous_action[2]),
                "bess_dispatch": float(self._previous_action[3]),
            }

        env_context: dict[str, Any] | None = None
        if env is not None:
            bess_p_max = float(getattr(env._bess, "P_MAX_MW", 5.0))
            committed_mw_max = float(env._scfg.get("committed_mw_max", 30.0))
            obs_dict = obs_to_dict(obs, self._state_names)
            committed_mw_norm = obs_dict.get("committed_mw_norm", 0.0)
            committed_mw = round(committed_mw_norm * committed_mw_max, 3)
            regd_signal = obs_dict.get("regd_signal", 0.0)
            p_flex_nom_norm = obs_dict.get("p_flex_nom_norm", 0.0)
            T_max = round(max(obs_dict.get("temp_A_norm", 0.0), obs_dict.get("temp_B_norm", 0.0)), 4)
            target_kw = round(regd_signal * committed_mw * 1000, 1)
            p_flex_nom_kw = round(p_flex_nom_norm * 250000, 1)
            d_baseline = float(np.clip(regd_signal * committed_mw / bess_p_max if bess_p_max > 0 else 0.0, -1.0, 1.0))
            bess_soc = obs_dict.get("bess_soc", 0.5)
            if bess_soc < 0.15 and d_baseline > 0:
                d_baseline = 0.0
            if bess_soc > 0.80 and d_baseline < 0:
                d_baseline = 0.0
            env_context = {
                "committed_mw_max":       committed_mw_max,
                "bess_p_max_mw":          bess_p_max,
                "committed_mw":           committed_mw,
                "T_max":                  T_max,
                "target_kw":              target_kw,
                "p_flex_nom_kw":          p_flex_nom_kw,
                "backlog_increment":      round(p_flex_nom_norm * 2.78, 4),
                "bess_dispatch_baseline": round(d_baseline, 4),
            }

        try:
            payload = self._generate_payload(
                obs=obs,
                scenario=scenario,
                mode="hardware",
                field_order=["throttle_batch", "pump_speed_A", "hvac_effort", "bess_dispatch"],
                bounds={
                    "throttle_batch": (0.0, 1.0),
                    "pump_speed_A": (0.0, 1.0),
                    "hvac_effort": (0.0, 1.0),
                    "bess_dispatch": (-1.0, 1.0),
                },
                previous_by_field=previous_by_field,
                env_context=env_context,
            )
        except ValueError as exc:
            raise RuntimeError(
                f"LLM hardware agent failed to produce a valid action at step 0 "
                f"(no previous action to fall back on): {exc}"
            ) from exc

        action = np.array([
            float(payload["throttle_batch"]),
            float(payload["pump_speed_A"]),
            float(payload["hvac_effort"]),
            float(payload["bess_dispatch"]),
        ], dtype=np.float32)
        self._previous_action = action
        return action, None


class MacroLLMPolicyAgent(_BaseLLMPolicyAgent):
    """Macro-level LLM policy that emits commitment and bidding actions."""

    def predict(
        self,
        obs: np.ndarray,
        deterministic: bool = True,
        env: C2GFastEnv | None = None,
        scenario: str = "default",
    ):
        previous_by_field = None
        if self._previous_action is not None:
            previous_by_field = {
                "commit_norm": float(self._previous_action[0]),
                "bid_price": float(self._previous_action[1] * 100.0),
            }

        env_context: dict[str, Any] | None = None
        if env is not None:
            bess_p_max = float(getattr(env._bess, "P_MAX_MW", 5.0))
            committed_mw_max = float(getattr(env, "_committed_max_mw", 30.0))
            dr_baseline_mw = float(getattr(env, "_dr_baseline_mw", 5.0))
            obs_dict = obs_to_dict(obs, self._state_names)
            rmcp_norm = obs_dict.get("rmcp_norm", 0.0)
            grid_load_norm = obs_dict.get("grid_load_norm", 0.0)
            rmcp_usd = round(100.0 * rmcp_norm, 2)
            T_max_macro = round(max(obs_dict.get("temp_A_norm", 0.0), obs_dict.get("temp_B_norm", 0.0)), 4)
            freq_dev_norm = obs_dict.get("freq_dev_norm", 0.0)
            v_pcc_pu = obs_dict.get("v_pcc_pu", 1.0)
            # Warm-start baseline (mirrors WARM-START rules in system prompt)
            if grid_load_norm > 0.7:
                c0 = 0.80
            elif grid_load_norm > 0.4:
                c0 = 0.50
            else:
                c0 = 0.20
            # Safety overrides
            headroom = 1.0 - T_max_macro
            if headroom < 0.10:
                c0 = min(c0, 0.30)
            if freq_dev_norm < -0.3:
                c0 = min(1.0, c0 + 0.2)
            if v_pcc_pu < 0.96:
                c0 = min(c0, 0.40)
            p0 = float(np.clip(40.0 * rmcp_norm, 0.0, 100.0))
            commit_norm_prev = float(self._previous_action[0]) if self._previous_action is not None else 0.0
            env_context = {
                "committed_mw_max":  committed_mw_max,
                "dr_baseline_mw":    dr_baseline_mw,
                "bess_p_max_mw":     bess_p_max,
                "commit_norm_prev":  commit_norm_prev,
                "rmcp_usd":          rmcp_usd,
                "T_max":             T_max_macro,
                "commit_norm_0":     round(c0, 2),
                "bid_price_0":       round(p0, 1),
            }

        try:
            payload = self._generate_payload(
                obs=obs,
                scenario=scenario,
                mode="macro",
                field_order=["commit_norm", "bid_price"],
                bounds={
                    "commit_norm": (0.0, 1.0),
                    "bid_price": (0.0, 100.0),
                },
                previous_by_field=previous_by_field,
                env_context=env_context,
            )
        except ValueError as exc:
            raise RuntimeError(
                f"LLM macro agent failed to produce a valid action at step 0 "
                f"(no previous action to fall back on): {exc}"
            ) from exc

        commit_norm = float(payload["commit_norm"])
        bid_price_norm = float(payload["bid_price"]) / 100.0

        if env is not None:
            max_commit_mw = float(getattr(env, "_committed_max_mw", 15.0))
            env.committed_mw = commit_norm * max_commit_mw

        action = np.array([commit_norm, bid_price_norm], dtype=np.float32)
        self._previous_action = action
        return action, None


class LLMPolicyAgent:
    """Backward-compatible wrapper over dedicated hardware/macro LLM classes."""

    uses_env_context = True

    def __init__(
        self,
        model_id: str,
        mode: str,
        prompts: dict[str, dict[str, str]],
        state_names: list[str],
        max_new_tokens: int = 12000,
        temperature: float = 0.0,
        api_base: str = "http://localhost:8000/v1",
        enable_thinking: bool = True,
    ):
        if mode == "hardware":
            self._delegate = HardwareLLMPolicyAgent(
                model_id=model_id,
                prompts=prompts,
                state_names=state_names,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                api_base=api_base,
                enable_thinking=enable_thinking,
            )
        elif mode == "macro":
            self._delegate = MacroLLMPolicyAgent(
                model_id=model_id,
                prompts=prompts,
                state_names=state_names,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                api_base=api_base,
                enable_thinking=enable_thinking,
            )
        else:
            raise ValueError(f"Invalid LLM mode '{mode}'. Expected 'hardware' or 'macro'.")

    def predict(
        self,
        obs: np.ndarray,
        deterministic: bool = True,
        env: C2GFastEnv | None = None,
        scenario: str = "default",
    ):
        return self._delegate.predict(obs, deterministic=deterministic, env=env, scenario=scenario)
