import shutil
import tempfile
import unittest

from agents.goal_verifier import GoalVerifier
from agents.recovery_policy import RecoveryPolicy
from agents.step_verifier import StepVerifier
from browser.actions import BrowserExecutor
from context.context_builder import ContextBuilder
from memory.graph import NavigationGraph
from memory.history import MemoryState
from memory.signature import (
    descriptor_for_target, element_descriptor, normalize_url, page_key,
    target_for_descriptor, view_signature,
)
from models.action_models import AgentAction
from models.orchestration_models import ActionResult


class FakeMouse:
    def __init__(self):
        self.wheel_calls = []

    async def wheel(self, x, y):
        self.wheel_calls.append((x, y))


class FakePage:
    def __init__(self):
        self.url = "https://example.com"
        self.mouse = FakeMouse()
        self.frames = []


class FakeLLMClient:
    def __init__(self, response):
        self.response = response

    async def generate_json(self, system_prompt, user_prompt):
        return self.response


class RecoveryPolicyTests(unittest.TestCase):
    def test_failed_locator_recovers_with_scroll(self):
        policy = RecoveryPolicy()
        result = ActionResult(
            success=False,
            action="click",
            target="pw-id-1",
            error="No element found",
            recovery_hint="refresh_dom_or_use_vision",
        )

        action = policy.from_result(result)

        self.assertEqual(action.action, "scroll")
        self.assertEqual(action.value, "down")


class BrowserExecutorTests(unittest.IsolatedAsyncioTestCase):
    async def test_page_level_scroll_returns_structured_result(self):
        page = FakePage()
        executor = BrowserExecutor(page)

        result = await executor.execute(AgentAction(action="scroll", value="up"))

        self.assertTrue(result.success)
        self.assertEqual(result.action, "scroll")
        self.assertEqual(result.metadata["direction"], "up")
        self.assertEqual(page.mouse.wheel_calls, [(0, -600)])

    async def test_missing_target_returns_recovery_hint(self):
        executor = BrowserExecutor(FakePage())

        result = await executor.execute(AgentAction(action="click", reasoning="test"))

        self.assertFalse(result.success)
        self.assertEqual(result.recovery_hint, "choose_visible_target_or_scroll")


class GoalVerifierTests(unittest.IsolatedAsyncioTestCase):
    async def test_verifier_accepts_complete_response(self):
        verifier = GoalVerifier(FakeLLMClient({"complete": True, "reason": "found"}))

        complete = await verifier.verify("find price", "page has price", "memory")

        self.assertTrue(complete)

    async def test_verifier_rejects_empty_response(self):
        verifier = GoalVerifier(FakeLLMClient({}))

        complete = await verifier.verify("find price", "page", "memory")

        self.assertFalse(complete)


class SignatureTests(unittest.TestCase):
    TRACKED = "https://www.amazon.com/s?k=iPhone+16&crid=2MUZMVD8M5O0F&ref=nb_sb_noss_1"
    CLEAN = "https://amazon.com/s?k=iPhone+16"

    def test_normalize_url_strips_tracking_but_keeps_query(self):
        self.assertEqual(normalize_url(self.TRACKED), self.CLEAN)

    def test_normalize_url_drops_www_fragment_and_trailing_slash(self):
        self.assertEqual(
            normalize_url("https://www.example.com/a/#section"),
            normalize_url("https://example.com/a"),
        )

    def test_page_key_ignores_tracking_parameters(self):
        self.assertEqual(page_key(self.TRACKED), page_key(self.CLEAN))

    def test_page_key_distinguishes_meaningful_query(self):
        self.assertNotEqual(
            page_key("https://amazon.com/s?k=iphone"),
            page_key("https://amazon.com/s?k=ipad"),
        )

    def test_view_signature_ignores_playwright_index_renumbering(self):
        first = [{"playwright_index": "pw-id-0", "tag": "a", "text": "Books"}]
        # The extractor recounts indices on every extraction.
        renumbered = [{"playwright_index": "pw-id-9", "tag": "a", "text": "Books"}]

        self.assertEqual(
            view_signature(self.CLEAN, first),
            view_signature(self.CLEAN, renumbered),
        )

    def test_view_signature_ignores_dom_reordering(self):
        elements = [
            {"tag": "a", "text": "Books"},
            {"tag": "h1", "text": "Results"},
        ]

        self.assertEqual(
            view_signature(self.CLEAN, elements),
            view_signature(self.CLEAN, list(reversed(elements))),
        )

    def test_view_signature_changes_when_viewport_changes(self):
        top = [{"tag": "a", "text": "Books"}]
        scrolled = [{"tag": "h2", "text": "Next page"}]

        self.assertNotEqual(
            view_signature(self.CLEAN, top),
            view_signature(self.CLEAN, scrolled),
        )

    def test_element_descriptor_excludes_unstable_attributes(self):
        stable = {"tag": "button", "text": "Go"}
        with_generated = {
            "tag": "button",
            "text": "Go",
            "playwright_index": "pw-id-4",
            "id": "react-select-3-input",
            "className": "css-1x2y3z4",
        }

        self.assertEqual(element_descriptor(stable), element_descriptor(with_generated))

    def test_element_descriptor_strips_href_query(self):
        self.assertEqual(
            element_descriptor({"tag": "a", "href": "/dp/B0C?ref=sr_1_1&psc=1"}),
            element_descriptor({"tag": "a", "href": "/dp/B0C"}),
        )


class StepVerifierTests(unittest.TestCase):
    A = "https://a.test/"
    B = "https://b.test/"

    def setUp(self):
        self.verifier = StepVerifier()

    def _result(self, success=True, url_after=None, value=None):
        return ActionResult(
            success=success,
            action="x",
            url_before=self.A,
            url_after=url_after or self.A,
            value=value,
        )

    def test_url_change_verifies_a_click(self):
        verdict = self.verifier.verify(
            AgentAction(action="click"), self._result(url_after=self.B)
        )

        self.assertEqual(verdict.status, "verified")
        self.assertEqual(verdict.evidence, "url_changed")

    def test_viewport_change_verifies_a_scroll(self):
        verdict = self.verifier.verify(
            AgentAction(action="scroll"), self._result(), "view_aaa", "view_bbb"
        )

        self.assertEqual(verdict.status, "verified")
        self.assertEqual(verdict.evidence, "view_changed")

    def test_clean_execution_with_no_change_is_not_verified(self):
        # The gap ActionResult.success hides: Playwright did not raise, but
        # nothing on the page moved.
        verdict = self.verifier.verify(
            AgentAction(action="click"), self._result(), "view_aaa", "view_aaa"
        )

        self.assertTrue(self._result().success)
        self.assertEqual(verdict.status, "unverified")
        self.assertEqual(verdict.evidence, "no_observable_change")

    def test_navigation_to_the_same_url_is_not_verified(self):
        verdict = self.verifier.verify(AgentAction(action="navigate"), self._result())

        self.assertEqual(verdict.status, "unverified")

    def test_url_change_with_only_tracking_params_is_not_verified(self):
        # A URL that only gained tracking noise is the same page, not a step
        # that did something.
        result = ActionResult(
            success=True,
            action="x",
            url_before="https://a.test/?ref=abc",
            url_after="https://a.test/?utm_source=x",
        )
        verdict = self.verifier.verify(AgentAction(action="click"), result, "view_aaa", "view_aaa")

        self.assertEqual(verdict.status, "unverified")
        self.assertEqual(verdict.evidence, "no_observable_change")

    def test_extract_is_judged_on_its_value(self):
        got = self.verifier.verify(AgentAction(action="extract"), self._result(value=" 19.99 "))
        empty = self.verifier.verify(AgentAction(action="extract"), self._result(value="   "))

        self.assertEqual(got.status, "verified")
        self.assertEqual(empty.status, "unverified")

    def test_failed_execution_is_never_verified(self):
        verdict = self.verifier.verify(
            AgentAction(action="click"), self._result(success=False, url_after=self.B)
        )

        self.assertEqual(verdict.status, "unverified")
        self.assertEqual(verdict.evidence, "execution_failed")

    def test_wait_and_done_are_not_applicable(self):
        for name in ("wait", "done"):
            verdict = self.verifier.verify(AgentAction(action=name), self._result())
            self.assertEqual(verdict.status, "not_applicable", name)

    def test_dom_is_not_re_extracted_when_the_answer_is_already_known(self):
        cases = {
            "url already changed": (AgentAction(action="click"), self._result(url_after=self.B)),
            "navigation": (AgentAction(action="navigate"), self._result()),
            "extraction": (AgentAction(action="extract"), self._result(value="x")),
            "wait": (AgentAction(action="wait"), self._result()),
            "failed": (AgentAction(action="click"), self._result(success=False)),
        }
        for label, (action, result) in cases.items():
            self.assertFalse(self.verifier.needs_view_signature(action, result), label)

    def test_dom_is_re_extracted_when_it_is_the_only_evidence(self):
        self.assertTrue(
            self.verifier.needs_view_signature(AgentAction(action="click"), self._result())
        )


class TargetDescriptorTests(unittest.TestCase):
    ELEMENTS = [
        {"playwright_index": "pw-id-0", "tag": "a", "text": "Home"},
        {"playwright_index": "pw-id-3", "tag": "input", "type": "submit", "aria_label": "Search"},
    ]

    def test_target_resolves_to_a_descriptor_without_the_index(self):
        descriptor = descriptor_for_target("pw-id-3", self.ELEMENTS)

        self.assertIn("aria_label=Search", descriptor)
        self.assertNotIn("pw-id", descriptor)

    def test_same_element_at_a_new_index_yields_the_same_descriptor(self):
        # What a later run sees after the extractor recounts.
        renumbered = [
            {"playwright_index": "pw-id-9", "tag": "input", "type": "submit", "aria_label": "Search"},
        ]

        self.assertEqual(
            descriptor_for_target("pw-id-3", self.ELEMENTS),
            descriptor_for_target("pw-id-9", renumbered),
        )

    def test_missing_or_absent_target_is_none(self):
        self.assertIsNone(descriptor_for_target(None, self.ELEMENTS))
        self.assertIsNone(descriptor_for_target("pw-id-99", self.ELEMENTS))
        self.assertIsNone(descriptor_for_target("pw-id-0", []))

    def test_ambiguous_target_descriptor_is_not_remembered(self):
        elements = [
            {"playwright_index": "pw-id-1", "tag": "a", "text": "Vote"},
            {"playwright_index": "pw-id-2", "tag": "a", "text": "Vote"},
        ]

        self.assertIsNone(descriptor_for_target("pw-id-2", elements))


class VerifiedActionIdTests(unittest.TestCase):
    def test_same_action_for_same_goal_reinforces_one_entry(self):
        first = MemoryState.verified_action_id("buy milk", "page_a", "click", "desc", "")
        again = MemoryState.verified_action_id("buy milk", "page_a", "click", "desc", "")

        self.assertEqual(first, again)

    def test_goal_and_page_both_separate_memories(self):
        base = MemoryState.verified_action_id("buy milk", "page_a", "click", "desc", "")
        other_goal = MemoryState.verified_action_id("buy bread", "page_a", "click", "desc", "")
        other_page = MemoryState.verified_action_id("buy milk", "page_b", "click", "desc", "")

        self.assertNotEqual(base, other_goal)
        self.assertNotEqual(base, other_page)


class NavigationGraphTests(unittest.TestCase):
    A, B, C = "page_a", "page_b", "page_c"

    def setUp(self):
        self._dir = tempfile.mkdtemp()
        self.graph = NavigationGraph(self._dir)

    def tearDown(self):
        self.graph.close()
        shutil.rmtree(self._dir, ignore_errors=True)

    def test_edge_is_recorded_and_readable(self):
        self.graph.record_edge(self.A, self.B, "click", "tag=a|text=Next")

        edges = self.graph.neighbours(self.A)

        self.assertEqual(len(edges), 1)
        self.assertEqual(edges[0]["to_page"], self.B)
        self.assertEqual(edges[0]["target_descriptor"], "tag=a|text=Next")

    def test_repeating_a_route_reinforces_rather_than_duplicates(self):
        for _ in range(3):
            self.graph.record_edge(self.A, self.B, "click", "tag=a|text=Next")

        edges = self.graph.neighbours(self.A)

        self.assertEqual(len(edges), 1)
        self.assertEqual(edges[0]["times_seen"], 3)

    def test_self_transition_is_not_an_edge(self):
        # The action changed the page in place; it went nowhere.
        self.assertFalse(self.graph.record_edge(self.A, self.A, "click"))
        self.assertEqual(self.graph.edge_count(), 0)

    def test_shortest_path_is_found_across_hops(self):
        self.graph.record_edge(self.A, self.B, "click", "to-b")
        self.graph.record_edge(self.B, self.C, "click", "to-c")

        route = self.graph.path_between(self.A, self.C)

        self.assertEqual([e["to_page"] for e in route], [self.B, self.C])
        self.assertEqual(route[0]["target_descriptor"], "to-b")

    def test_unknown_route_returns_empty(self):
        self.graph.record_edge(self.A, self.B, "click")

        self.assertEqual(self.graph.path_between(self.A, "page_unreachable"), [])

    def test_survives_reopening_the_store(self):
        self.graph.record_edge(self.A, self.B, "click")
        self.graph.close()

        reopened = NavigationGraph(self._dir)
        try:
            self.assertEqual(len(reopened.neighbours(self.A)), 1)
        finally:
            reopened.close()

    def test_missing_store_degrades_instead_of_raising(self):
        broken = NavigationGraph(self._dir)
        broken.close()

        self.assertFalse(broken.is_available)
        self.assertFalse(broken.record_edge(self.A, self.B, "click"))
        self.assertEqual(broken.neighbours(self.A), [])
        self.assertEqual(broken.path_between(self.A, self.B), [])


class RecallResolutionTests(unittest.TestCase):
    """A remembered action is only useful if it can be pointed at a live element."""

    SEARCH = {"playwright_index": "pw-id-3", "tag": "input", "type": "submit", "aria_label": "Search"}

    def test_remembered_descriptor_resolves_to_todays_index(self):
        descriptor = descriptor_for_target("pw-id-3", [self.SEARCH])
        # Next run: the extractor gave the same element a different index.
        today = [{**self.SEARCH, "playwright_index": "pw-id-11"}]

        self.assertEqual(target_for_descriptor(descriptor, today), "pw-id-11")

    def test_element_not_on_screen_resolves_to_none(self):
        descriptor = descriptor_for_target("pw-id-3", [self.SEARCH])
        scrolled_away = [{"playwright_index": "pw-id-0", "tag": "h2", "text": "Footer"}]

        self.assertIsNone(target_for_descriptor(descriptor, scrolled_away))

    def test_round_trip_is_stable(self):
        descriptor = descriptor_for_target("pw-id-3", [self.SEARCH])

        self.assertEqual(target_for_descriptor(descriptor, [self.SEARCH]), "pw-id-3")

    def test_ambiguous_descriptor_does_not_guess_a_live_target(self):
        descriptor = element_descriptor({"tag": "a", "text": "Vote"})
        elements = [
            {"playwright_index": "pw-id-1", "tag": "a", "text": "Vote"},
            {"playwright_index": "pw-id-2", "tag": "a", "text": "Vote"},
        ]

        self.assertIsNone(target_for_descriptor(descriptor, elements))


class MemoryRecallContextTests(unittest.TestCase):
    def setUp(self):
        self.builder = ContextBuilder()

    def test_resolved_action_is_rendered_with_its_live_target(self):
        text = self.builder.build_memory_recall(
            [{"action": "click", "target_descriptor": "tag=input", "live_target": "pw-id-7",
              "evidence": "url_changed"}],
            [],
        )

        self.assertIn("click on pw-id-7", text)
        self.assertIn("url_changed", text)

    def test_unresolved_action_is_still_offered_as_a_hint(self):
        text = self.builder.build_memory_recall(
            [{"action": "click", "target_descriptor": "tag=a|text=Next", "live_target": None}],
            [],
        )

        self.assertIn("not currently visible", text)
        self.assertIn("tag=a|text=Next", text)

    def test_routes_are_rendered_with_how_often_they_were_taken(self):
        text = self.builder.build_memory_recall(
            [], [{"action": "navigate", "to_page": "page_b", "times_seen": 4, "target_descriptor": ""}]
        )

        self.assertIn("page_b", text)
        self.assertIn("4x", text)

    def test_nothing_recalled_renders_nothing(self):
        self.assertEqual(self.builder.build_memory_recall([], []), "")


if __name__ == "__main__":
    unittest.main()
