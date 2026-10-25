"""Launch-time LLM param probe (2026-10-25, plan: docs/internal-notes/
20261025-reasoning-effort-knob-plan.md §3.4).

Param-agnostic by design: the probe never knows WHICH params it validates —
it fires one minimal real request per stage using that stage's exact
production param assembly, so any future knob is probed for free and the
probe can never drift from what production sends.

Semantics:
- Runs only when the caller supplies at least one stage (i.e. an effort knob
  is set). Zero network, zero cost for the default config.
- 200 -> log verified.
- 400 (deterministic config error) -> raise SystemExit with the provider's
  message verbatim (MiniMax's lists the valid values — self-healing).
- 429/5xx/timeout/network -> WARN and continue: provider weather must not
  block startup; the runtime 400-abort remains as the backstop.
"""

import logging
from typing import Any, Callable, Dict, List, NamedTuple

from openai import BadRequestError

from neatmem.utils.llm_client import complete_chat

logger = logging.getLogger(__name__)

# Probe timeout is transport-level (never sent to the API), so it may
# legitimately differ from a stage's production timeout without drift.
_PROBE_TIMEOUT_S = 30.0


class ProbeStage(NamedTuple):
    label: str          # e.g. "rerank", "query_rewrite"
    client: Any         # OpenAI-compatible client for this stage
    model: str
    params: Dict[str, Any]  # the stage's exact production request params


def run_launch_probe(stages: List[ProbeStage]) -> int:
    """Fire one minimal call per stage. Returns the number verified.

    Raises SystemExit on a deterministic 400 so the server refuses to start
    with the provider's own error message.
    """
    verified = 0
    for stage in stages:
        label = f"{stage.label}(model={stage.model})"
        try:
            complete_chat(
                stage.client,
                stage.model,
                [{"role": "user", "content": "ping"}],
                **stage.params,
                timeout=_PROBE_TIMEOUT_S,
            )
        except BadRequestError as e:
            raise SystemExit(
                f"ERROR: launch probe rejected the LLM params for {label}.\n"
                f"  Provider says: {e}\n"
                "  Fix or unset the offending *_REASONING_EFFORT knob. "
                "(Validated by plan §3.4 launch probe.)"
            )
        except Exception as e:  # transient: 429/5xx/timeout/network
            logger.warning(
                "Launch probe for %s could not verify params (transient: %s); "
                "starting anyway — runtime errors will surface per call.",
                label, e,
            )
            continue
        logger.info("Launch probe verified LLM params for %s", label)
        verified += 1
    return verified
