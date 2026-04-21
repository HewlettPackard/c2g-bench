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
import warnings
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import numpy as np
import yaml

from c2g_env import C2GFastEnv

# Suppress transformers and tokenizers warnings
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore", category=UserWarning, module="transformers")
warnings.filterwarnings("ignore", category=FutureWarning, module="transformers")


# ──────────────────────────────────────────────────────────────────
# LLM Agent Helper Functions
# ──────────────────────────────────────────────────────────────────

def extract_json(text: str) -> dict[str, Any]:
    """Extract JSON object from LLM-generated text."""
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        return {}
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return {}


def _get_default_hardware_action() -> np.ndarray:
    """Return hardware fallback action from ablation defaults when available.

    Order: [throttle_batch, pump_speed_A, hvac_effort, bess_dispatch].
    """
    try:
        from c2g_env.experiments.action_ablation_env import ActionAblationFastEnv

        d = ActionAblationFastEnv.ABLATION_DEFAULTS
        return np.array(
            [
                float(d["throttle_batch"]),
                float(d["pump_speed_A"]),
                float(d["hvac_effort"]),
                float(d["bess_dispatch"]),
            ],
            dtype=np.float32,
        )
    except Exception:
        # Fallback if ablation env import fails for any reason.
        return np.array([1.0, 1.0, 1.0, 0.0], dtype=np.float32)


def _get_default_macro_action() -> np.ndarray:
    """Return safe default action for macro mode (moderate bidding).
    
    Action components:
    - commit_norm=0.5: moderate commitment level
    - bid_price_norm=0.5: moderate bidding price
    """
    return np.array([0.5, 0.5], dtype=np.float32)


def safe_float(value: Any, default: float) -> float:
    """Safely convert value to float with fallback default."""
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if np.isfinite(out) else default


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
    mode: str,
) -> str:
    """Build full prompt from system and user prompts with state context."""
    system_prompt = mode_prompts["system"]
    user_prompt = mode_prompts["user"].format(scenario=scenario, state_json=json.dumps(state_dict))
    return f"{system_prompt}\n\n{user_prompt}"


def generate_structured(
    generator: Any,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
) -> dict[str, Any]:
    """Generate structured JSON response from LLM via text-generation pipeline."""
    do_sample = temperature > 0.0
    kwargs: dict[str, Any] = {
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "return_full_text": False,
        "truncation": True,
    }
    if do_sample:
        kwargs["temperature"] = temperature

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        outputs = generator(prompt, **kwargs)
    
    text = outputs[0].get("generated_text", "") if outputs else ""
    return extract_json(text)


def load_prompt_templates(template_path: str | Path) -> dict[str, dict[str, str]]:
    """Load system and user prompts for hardware and macro agents."""
    path = Path(template_path)
    if not path.is_absolute():
        path = Path(__file__).resolve().parent.parent / path
    if not path.exists():
        raise FileNotFoundError(f"Prompt template file not found: {path}")

    with open(path) as fh:
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


def validate_llm_model_id(model_id: str) -> str:
    """
    Validate that model_id is either:
      1) a local filesystem path to a loadable HF model directory, or
      2) a Hugging Face Hub repo id / URL for a remote model.

    Returns a normalized identifier usable by transformers.pipeline.
    """
    model_ref = str(model_id).strip()
    if not model_ref:
        raise ValueError("--llm-model-id must be a non-empty local path or Hugging Face model id/URL.")

    local_path = Path(model_ref).expanduser().resolve()
    if local_path.exists() and local_path.is_dir():
        try:
            from transformers import AutoConfig
        except ImportError as exc:
            raise ImportError(
                "transformers is required for llm_policy validation. Install with: pip install transformers"
            ) from exc

        try:
            AutoConfig.from_pretrained(str(local_path), local_files_only=True)
        except Exception as exc:
            raise ValueError(
                f"llm_model_id points to an existing directory but it is not a loadable HF model: {local_path}"
            ) from exc
        return str(local_path)

    # Normalize Hugging Face model URL to repo_id when applicable.
    parsed = urlparse(model_ref)
    repo_id = model_ref
    if parsed.scheme in {"http", "https"}:
        if parsed.netloc not in {"huggingface.co", "www.huggingface.co"}:
            raise ValueError(
                "--llm-model-id URL must point to huggingface.co for remote model loading."
            )
        path = parsed.path.strip("/")
        if not path:
            raise ValueError("--llm-model-id URL is missing a model repo path.")
        parts = path.split("/")
        if parts[0] == "models":
            parts = parts[1:]
        if len(parts) < 2:
            raise ValueError(
                "--llm-model-id URL must be a model repo URL like https://huggingface.co/<org>/<model>."
            )
        repo_id = "/".join(parts[:2])

    try:
        from huggingface_hub import HfApi
        from huggingface_hub.errors import HfHubHTTPError
        from huggingface_hub.utils import HFValidationError, validate_repo_id
    except ImportError as exc:
        raise ImportError(
            "huggingface_hub is required for llm_policy remote id validation. "
            "Install with: pip install huggingface_hub"
        ) from exc

    try:
        validate_repo_id(repo_id)
    except HFValidationError as exc:
        raise ValueError(
            "--llm-model-id must be a valid local path or Hugging Face repo id/URL (e.g. org/model or https://huggingface.co/org/model)."
        ) from exc

    try:
        HfApi().model_info(repo_id)
    except HfHubHTTPError as exc:
        raise ValueError(
            f"Remote Hugging Face model repo not accessible: {repo_id}. "
            "Check repo id, visibility, token permissions, and network access."
        ) from exc

    return repo_id


# ──────────────────────────────────────────────────────────────────
# LLM Policy Agent
# ──────────────────────────────────────────────────────────────────

def _load_committed_mw_from_config() -> dict[str, float]:
    """Load committed_mw values from scenario config files."""
    scenarios = ["default", "scenario_a", "scenario_b", "scenario_c"]
    committed_mw = {}
    
    for scenario in scenarios:
        try:
            config_path = Path(__file__).resolve().parent.parent / "conf" / "scenario" / f"{scenario}.yaml"
            if config_path.exists():
                with open(config_path) as fh:
                    data = yaml.safe_load(fh) or {}
                committed_mw[scenario] = float(data.get("committed_mw", 15.0))
            else:
                # Fallback if config not found
                committed_mw[scenario] = 15.0
        except Exception:
            # Fallback to defaults on any error
            committed_mw[scenario] = 15.0
    
    return committed_mw


COMMIT_MW = _load_committed_mw_from_config()


class LLMPolicyAgent:
    """LLM-driven policy via transformers text-generation backend."""

    uses_env_context = True

    def __init__(
        self,
        model_id: str,
        mode: str,
        prompts: dict[str, dict[str, str]],
        state_names: list[str],
        max_new_tokens: int = 256,
        temperature: float = 0.0,
        committed_mw: dict[str, float] | None = None,
    ):
        """
        Initialize LLM policy agent.
        
        Parameters
        ----------
        model_id : str
            HuggingFace model ID (e.g., "gpt2", "meta-llama/Llama-2-7b-chat")
        mode : str
            "hardware" for 4-D low-level control, "macro" for 2-D commitment control
        prompts : dict[str, dict[str, str]]
            System and user prompts for each mode: {"hardware": {"system": "...", "user": "..."}, ...}
        state_names : list[str]
            Semantic names for each state dimension (replaces index-based naming)
        max_new_tokens : int
            Max tokens in LLM response
        temperature : float
            Sampling temperature (0 = deterministic)
        committed_mw : dict[str, float] | None
            Max commitment MW by scenario
        """
        if mode not in {"hardware", "macro"}:
            raise ValueError(f"Invalid LLM mode '{mode}'. Expected 'hardware' or 'macro'.")

        try:
            from transformers import pipeline
        except ImportError as exc:
            raise ImportError(
                "transformers is required for llm agent. Install with: pip install transformers"
            ) from exc

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self._generator = pipeline("text-generation", model=model_id)
        
        self._mode = mode
        self._max_new_tokens = int(max_new_tokens)
        self._temperature = float(temperature)
        self._committed_mw = committed_mw or COMMIT_MW
        self._prompts = prompts  # {"hardware": {"system": "...", "user": "..."}, "macro": {...}}
        self._state_names = state_names
        self._previous_action = None  # Track last action for fallback
        self.algo_name = f"llm_policy_{mode}"

    def predict(
        self,
        obs: np.ndarray,
        deterministic: bool = True,
        env: C2GFastEnv | None = None,
        scenario: str = "default",
    ):
        """
        Predict action from observation using LLM.
        
        Parameters
        ----------
        obs : np.ndarray
            Observation vector (17-D for hardware, 2-D for macro)
        deterministic : bool
            Ignored (use temperature in init instead)
        env : object | None
            Environment (C2GFastEnv for hardware, C2GMacroEnv for macro) for context
        scenario : str
            Scenario name for prompt formatting
        
        Returns
        -------
        action : np.ndarray
            4-D action vector for hardware mode, 2-D for macro mode
        info : None
        """
        # Convert observation to semantic state dict
        obs_dict = obs_to_dict(obs, self._state_names)
        
        # Build prompt from system + user templates
        prompt = build_prompt(
            self._prompts[self._mode],
            obs_dict,
            scenario,
            self._mode,
        )
        
        # Generate structured response from LLM
        payload = generate_structured(
            self._generator,
            prompt,
            self._max_new_tokens,
            self._temperature,
        )

        if self._mode == "hardware":
            # Check if payload is empty (JSON extraction failed)
            if not payload:
                # Use previous action if available, otherwise use default full cooling
                if self._previous_action is not None:
                    warnings.warn(
                        f"LLM did not return valid JSON for hardware mode in scenario {scenario}. "
                        "Using previous action.",
                        RuntimeWarning,
                        stacklevel=2,
                    )
                    return self._previous_action, None
                else:
                    warnings.warn(
                        f"LLM did not return valid JSON for hardware mode in scenario {scenario} at step 0. "
                        "Using default action (throttle=1.0, pump=1.0, hvac=1.0, bess=0.0).",
                        RuntimeWarning,
                        stacklevel=2,
                    )
                    action = _get_default_hardware_action()
                    self._previous_action = action
                    return action, None
            
            action = np.array([
                np.clip(safe_float(payload.get("throttle_batch"), 1.0), 0.0, 1.0),
                np.clip(safe_float(payload.get("pump_speed_A"), 1.0), 0.0, 1.0),
                np.clip(safe_float(payload.get("hvac_effort"), 1.0), 0.0, 1.0),
                np.clip(safe_float(payload.get("bess_dispatch"), 0.0), -1.0, 1.0),
            ], dtype=np.float32)
            self._previous_action = action
            return action, None

        # Macro mode: return 2-D action (commitment_mw, bid_price)
        if not payload:
            # Use previous action if available, otherwise use default moderate bidding
            if self._previous_action is not None:
                warnings.warn(
                    f"LLM did not return valid JSON for macro mode in scenario {scenario}. "
                    "Using previous action.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                return self._previous_action, None
            else:
                warnings.warn(
                    f"LLM did not return valid JSON for macro mode in scenario {scenario} at step 0. "
                    "Using default action (commit_norm=0.5, bid_price_norm=0.5).",
                    RuntimeWarning,
                    stacklevel=2,
                )
                action = _get_default_macro_action()
                self._previous_action = action
                return action, None
        
        commit_norm = np.clip(safe_float(payload.get("commit_norm"), 0.5), 0.0, 1.0)
        bid_price = np.clip(safe_float(payload.get("bid_price"), 50.0), 0.0, 100.0)
        
        if env is not None:
            max_commit_mw = float(self._committed_mw.get(scenario, 15.0))
            env.committed_mw = commit_norm * max_commit_mw

        action = np.array([commit_norm, bid_price / 100.0], dtype=np.float32)
        self._previous_action = action
        return action, None
