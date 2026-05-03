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
from collections import deque
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
    # Provide defaults for ICRL placeholders so non-ICRL templates don't raise KeyError
    fmt_kwargs.setdefault("icrl_context", "(no past attempts yet)")
    fmt_kwargs.setdefault("icrl_instruction", "")
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
            extra_body={"chat_template_kwargs": {"enable_thinking": enable_thinking}},
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
        "icrl": {
            "hardware_attempt":   str(data.get("hardware_icrl_attempt",   "")).strip(),
            "macro_attempt":      str(data.get("macro_icrl_attempt",      "")).strip(),
            "hardware_explore":   str(data.get("hardware_icrl_explore",   "")).strip(),
            "hardware_exploit":   str(data.get("hardware_icrl_exploit",   "")).strip(),
            "hardware_autonomous":str(data.get("hardware_icrl_autonomous","")).strip(),
            "macro_explore":      str(data.get("macro_icrl_explore",      "")).strip(),
            "macro_exploit":      str(data.get("macro_icrl_exploit",      "")).strip(),
            "macro_autonomous":   str(data.get("macro_icrl_autonomous",   "")).strip(),
        },
    }


def probe_api_base(api_base: str, timeout: float = 5.0) -> str:
    """Verify the vLLM API is live and return the name of the first served model.

    Raises ``ConnectionError`` when the server is unreachable or returns an
    unexpected response.
    """
    import json as _json
    url = api_base.rstrip("/") + "/models"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:  # noqa: S310
            body = _json.loads(resp.read())
    except Exception as exc:
        raise ConnectionError(
            f"vLLM server not live at {url!r}. "
            "Start the server before running the benchmark.\n"
            f"  ({type(exc).__name__}: {exc})"
        ) from exc

    models = body.get("data")
    if not isinstance(models, list) or not models:
        raise ConnectionError(
            f"vLLM server at {url!r} returned an unexpected response "
            f"(expected {{\"data\": [...]}}, got: {body!r})."
        )
    return models[0]["id"]


# ──────────────────────────────────────────────────────────────────
# LLM Policy Agent
# ──────────────────────────────────────────────────────────────────


class _BaseLLMPolicyAgent:
    """Shared client/prompt plumbing for LLM-driven policy agents."""

    uses_env_context = True
    _agent_mode: str = ""  # "hardware" or "macro"; set by subclasses

    def __init__(
        self,
        prompts: dict[str, dict[str, str]],
        state_names: list[str],
        max_new_tokens: int = 256,
        temperature: float = 0.0,
        api_base: str = "http://localhost:8000/v1",
        enable_thinking: bool = True,
        context_num_steps: int = 25,
        icrl_mode: str = "autonomous",
    ):
        self._model_name = probe_api_base(api_base)

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
        self._prompts = prompts
        self._state_names = state_names
        self._max_new_tokens = int(max_new_tokens)
        self._temperature = float(temperature)
        self._enable_thinking: dict[str, bool] = {
            "hardware": enable_thinking,
            "macro": enable_thinking,
        }
        self._previous_action: np.ndarray | None = None

        # ── ICRL buffer ───────────────────────────────────────────────
        # Sliding window of the last context_num_steps (state, action, reward) triples.
        # deque(maxlen=N) automatically evicts the oldest entry when full.
        # maxlen=None means unlimited (context_num_steps=0).
        _maxlen: int | None = int(context_num_steps) if context_num_steps > 0 else None
        self._icrl_buffer: deque[dict[str, Any]] = deque(maxlen=_maxlen)
        _icrl = prompts.get("icrl", {})
        _m = self._agent_mode
        self._icrl_attempt_template:    str | None = _icrl.get(f"{_m}_attempt",    "") or None
        self._icrl_explore_template:    str | None = _icrl.get(f"{_m}_explore",    "") or None
        self._icrl_exploit_template:    str | None = _icrl.get(f"{_m}_exploit",    "") or None
        self._icrl_autonomous_template: str | None = _icrl.get(f"{_m}_autonomous", "") or None
        _valid_modes = ("autonomous", "preset", "exploit")
        if icrl_mode not in _valid_modes:
            raise ValueError(f"icrl_mode must be one of {_valid_modes}, got '{icrl_mode}'")
        self._icrl_mode: str = icrl_mode
        self._icrl_step: int = 0          # monotonic counter of env steps (incremented in push_reward)
        # Pending slot: filled by predict(), consumed by push_reward()
        self._pending_obs: np.ndarray | None = None
        self._pending_action_dict: dict[str, float] | None = None

    # ------------------------------------------------------------------
    # ICRL helpers
    # ------------------------------------------------------------------

    def push_reward(self, reward: float) -> None:
        """Record the reward received for the last predict() call.

        Must be called by the runner after every env.step().  All steps are
        counted and stored regardless of reward sign.
        """
        if self._pending_obs is None or self._pending_action_dict is None:
            return
        self._icrl_step += 1
        if self._icrl_attempt_template:
            state_dict = obs_to_dict(self._pending_obs, self._state_names)
            self._icrl_buffer.append({
                "step":        self._icrl_step,
                "reward":      round(float(reward), 3),
                "state_json":  json.dumps({k: round(v, 3) for k, v in state_dict.items()}),
                "action_json": json.dumps({k: round(v, 4) for k, v in self._pending_action_dict.items()}),
            })
        self._pending_obs = None
        self._pending_action_dict = None

    def _format_icrl_context(self) -> str:
        """Render the ICRL buffer into a prompt string.

        Each entry is formatted with the attempt template using the placeholders
        {step}, {reward}, {state_json}, {action_json}.  Entries are ordered
        oldest-first (t-N, …, t-1) so the most recent attempt is last.
        Returns a placeholder string when the buffer is empty.
        """
        if not self._icrl_buffer or not self._icrl_attempt_template:
            return "(no past attempts yet)"
        parts = [self._icrl_attempt_template.format(**entry).strip()
                 for entry in self._icrl_buffer]
        return "\n".join(parts)

    def _resolve_icrl_instruction(self) -> str:
        """Return the instruction string for the current step based on icrl_mode.

        - 'exploit'    : always use the exploitation instruction.
        - 'autonomous' : always use the combined explore-or-exploit instruction.
        - 'preset'     : alternate every step — even _icrl_step → explore, odd → exploit.
        """
        if self._icrl_mode == "exploit":
            return self._icrl_exploit_template or ""
        if self._icrl_mode == "autonomous":
            return self._icrl_autonomous_template or ""
        # preset: alternate per paper (K odd → exploit, K even → explore).
        # _icrl_step counts completed steps (0-indexed), so step 0 is K=1 (odd → exploit).
        if self._icrl_step % 2 == 0:
            return self._icrl_exploit_template or ""
        return self._icrl_explore_template or ""

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

    _agent_mode = "hardware"

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
            from c2g_env.obs_indices import Fast as _F
            bess_p_max        = float(getattr(env._bess, "P_MAX_MW", 5.0))
            committed_mw_max  = float(env._scfg.get("committed_mw_max", 30.0))
            # Derived obs values
            committed_mw_norm = float(obs[_F.COMMITTED])
            regd_signal       = float(obs[_F.REGD])
            bess_soc          = float(obs[_F.SOC])
            temp_a_norm       = float(obs[_F.TEMP_A])
            temp_b_norm       = float(obs[_F.TEMP_B])
            p_flex_nom_norm   = float(obs[_F.P_FLEX])
            committed_mw      = round(committed_mw_norm * committed_mw_max, 4)
            T_max             = round(max(temp_a_norm, temp_b_norm), 4)
            target_kw         = round(regd_signal * committed_mw * 1000.0, 2)
            p_flex_nom_kw     = round(p_flex_nom_norm * 250_000.0, 2)
            backlog_increment = round(p_flex_nom_norm * 2.78, 4)
            # BESS baseline with SOC ramp guards (mirrors system-prompt rules)
            if abs(regd_signal) >= 0.10:
                bess_base = float(np.clip(regd_signal * committed_mw / bess_p_max, -1.0, 1.0))
            else:
                bess_base = 0.0
            if bess_soc < 0.15 and bess_base > 0:
                bess_base *= max(0.0, (bess_soc - 0.10) / 0.05)
            if bess_soc > 0.80 and bess_base < 0:
                bess_base *= max(0.0, (0.95 - bess_soc) / 0.15)
            env_context = {
                "committed_mw_max":      committed_mw_max,
                "bess_p_max_mw":         bess_p_max,
                "committed_mw":          committed_mw,
                "T_max":                 T_max,
                "target_kw":             target_kw,
                "p_flex_nom_kw":         p_flex_nom_kw,
                "backlog_increment":     backlog_increment,
                "bess_dispatch_baseline": round(bess_base, 4),
            }

        # Inject ICRL context (formatted past attempts) into env_context
        if env_context is None:
            env_context = {}
        env_context["icrl_context"] = self._format_icrl_context()
        env_context["icrl_instruction"] = self._resolve_icrl_instruction()

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
        # Save for ICRL buffer; push_reward() will complete the entry
        self._pending_obs = obs.copy()
        self._pending_action_dict = {
            "throttle_batch": round(float(payload["throttle_batch"]), 4),
            "pump_speed_A":   round(float(payload["pump_speed_A"]),   4),
            "hvac_effort":    round(float(payload["hvac_effort"]),    4),
            "bess_dispatch":  round(float(payload["bess_dispatch"]),  4),
        }
        return action, None


class MacroLLMPolicyAgent(_BaseLLMPolicyAgent):
    """Macro-level LLM policy that emits commitment and bidding actions."""

    _agent_mode = "macro"

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
            bess_p_max = float(getattr(env._fast_env._bess, "P_MAX_MW", 5.0))
            from c2g_env.obs_indices import Macro as _M
            rmcp_norm       = float(obs[_M.RMCP]) if len(obs) > _M.RMCP else 0.25
            load_norm       = float(obs[_M.GRID_LOAD])
            hroom_A         = float(obs[_M.HEADROOM_A])
            hroom_B         = float(obs[_M.HEADROOM_B])
            freq_dev        = float(obs[_M.FREQ_DEV])
            v_pcc           = float(obs[_M.VPCC])
            temp_a          = float(obs[_M.TEMP_A])
            temp_b          = float(obs[_M.TEMP_B])
            committed_mw_max = float(getattr(env, "_committed_max_mw", 30.0))
            T_max           = round(max(temp_a, temp_b), 4)
            rmcp_usd        = round(rmcp_norm * 100.0, 2)
            # Replicate rule_based_macro baseline (commit_norm_0, bid_price_0)
            if load_norm > 0.7:
                commit_norm_0 = 0.80
            elif load_norm > 0.4:
                commit_norm_0 = 0.50
            else:
                commit_norm_0 = 0.20
            if min(hroom_A, hroom_B) < 0.10:
                commit_norm_0 = min(commit_norm_0, 0.30)
            if freq_dev < -0.3:
                commit_norm_0 = min(1.0, commit_norm_0 + 0.2)
            if v_pcc < 0.96:
                commit_norm_0 = min(commit_norm_0, 0.40)
            bid_price_0 = round(float(np.clip(40.0 * rmcp_norm, 0.0, 100.0)), 2)
            commit_norm_prev = round(
                float(self._previous_action[0]) if self._previous_action is not None else commit_norm_0, 4
            )
            env_context = {
                "committed_mw_max": committed_mw_max,
                "dr_baseline_mw":   float(getattr(env, "_dr_baseline_mw", 5.0)),
                "bess_p_max_mw":    bess_p_max,
                "rmcp_usd":         rmcp_usd,
                "T_max":            T_max,
                "commit_norm_0":    round(commit_norm_0, 4),
                "commit_norm_prev": commit_norm_prev,
                "bid_price_0":      bid_price_0,
            }

        # Inject ICRL context into env_context
        if env_context is None:
            env_context = {}
        env_context["icrl_context"] = self._format_icrl_context()
        env_context["icrl_instruction"] = self._resolve_icrl_instruction()

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
        # Save for ICRL buffer; push_reward() will complete the entry
        self._pending_obs = obs.copy()
        self._pending_action_dict = {
            "commit_norm": round(commit_norm, 4),
            "bid_price":   round(float(payload["bid_price"]), 2),
        }
        return action, None


class LLMPolicyAgent:
    """Backward-compatible wrapper over dedicated hardware/macro LLM classes."""

    uses_env_context = True

    def __init__(
        self,
        mode: str,
        prompts: dict[str, dict[str, str]],
        state_names: list[str],
        max_new_tokens: int = 256,
        temperature: float = 0.0,
        api_base: str = "http://localhost:8000/v1",
        enable_thinking: bool = True,
        context_num_steps: int = 25,
        icrl_mode: str = "autonomous",
    ):
        _shared_kwargs = dict(
            prompts=prompts,
            state_names=state_names,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            api_base=api_base,
            enable_thinking=enable_thinking,
            context_num_steps=context_num_steps,
            icrl_mode=icrl_mode,
        )
        if mode == "hardware":
            self._delegate = HardwareLLMPolicyAgent(**_shared_kwargs)
        elif mode == "macro":
            self._delegate = MacroLLMPolicyAgent(**_shared_kwargs)
        else:
            raise ValueError(f"Invalid LLM mode '{mode}'. Expected 'hardware' or 'macro'.")

    def push_reward(self, reward: float) -> None:
        """Forward reward to the delegate's ICRL buffer."""
        self._delegate.push_reward(reward)

    def predict(
        self,
        obs: np.ndarray,
        deterministic: bool = True,
        env: C2GFastEnv | None = None,
        scenario: str = "default",
        **kwargs,
    ):
        return self._delegate.predict(obs, deterministic=deterministic, env=env, scenario=scenario)
