from typing import Optional

from memory.history import StepVerdict
from memory.signature import normalize_url
from models.action_models import AgentAction
from models.orchestration_models import ActionResult

VERIFIED = "verified"
UNVERIFIED = "unverified"
NOT_APPLICABLE = "not_applicable"

# Nothing observable to check: waiting changes nothing by design, and whether
# the goal is met is GoalVerifier's question, not this one.
UNVERIFIABLE_ACTIONS = {"wait", "done"}

# Actions whose entire purpose is to land on a different URL.
NAVIGATION_ACTIONS = {"navigate", "back"}


class StepVerifier:
    """
    Decides whether an executed step actually did something.

    Deterministic on purpose. GoalVerifier asks an LLM once per completion
    claim; this runs on every execution, so an LLM call per step would treble
    cost and latency for a question that URL and viewport comparison already
    answer.

    ActionResult.success only means Playwright did not raise. This asks the
    separate question a memory layer should gate on: did the world observably
    change? Only status == VERIFIED should be trusted as a successful action.
    """

    def needs_view_signature(self, action: AgentAction, result: ActionResult) -> bool:
        """
        Whether a post-action DOM extraction is worth its cost.

        Re-extracting is the expensive part of verification, so it is skipped
        wherever the answer is already known: unverifiable actions, actions
        that failed outright, extractions (judged on their returned value),
        and anything that already changed the URL.
        """
        if action.action in UNVERIFIABLE_ACTIONS:
            return False
        if not result.success:
            return False
        if action.action == "extract":
            return False
        if action.action in NAVIGATION_ACTIONS:
            # Judged on the URL alone, so the viewport is never consulted.
            return False
        return not self._url_changed(result)

    def verify(
        self,
        action: AgentAction,
        result: ActionResult,
        signature_before: Optional[str] = None,
        signature_after: Optional[str] = None,
    ) -> StepVerdict:
        name = action.action

        def verdict(status: str, evidence: str) -> StepVerdict:
            return StepVerdict(
                status=status,
                evidence=evidence,
                signature_before=signature_before,
                signature_after=signature_after,
            )

        if name in UNVERIFIABLE_ACTIONS:
            return verdict(NOT_APPLICABLE, "not_verifiable")

        if not result.success:
            return verdict(UNVERIFIED, "execution_failed")

        if name == "extract":
            text = "" if result.value is None else str(result.value).strip()
            if text:
                return verdict(VERIFIED, "value_extracted")
            return verdict(UNVERIFIED, "empty_extraction")

        if self._url_changed(result):
            return verdict(VERIFIED, "url_changed")

        if name in NAVIGATION_ACTIONS:
            # Navigating and landing on the same URL moved nothing.
            return verdict(UNVERIFIED, "no_observable_change")

        if signature_before and signature_after:
            if signature_before != signature_after:
                return verdict(VERIFIED, "view_changed")
            return verdict(UNVERIFIED, "no_observable_change")

        return verdict(UNVERIFIED, "no_evidence_captured")

    @staticmethod
    def _url_changed(result: ActionResult) -> bool:
        if not (result.url_before and result.url_after):
            return False
        return normalize_url(result.url_before) != normalize_url(result.url_after)
