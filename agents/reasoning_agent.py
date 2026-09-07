import asyncio
import os
import time
import uuid
from typing import Awaitable, Callable, Optional

from loguru import logger

from agents.goal_verifier import GoalVerifier
from agents.recovery_policy import RecoveryPolicy
from agents.step_verifier import StepVerifier
from browser.actions import BrowserExecutor
from browser.controller import BrowserController
from browser.dom_extractor import DOMExtractor
from context.context_builder import ContextBuilder
from llm.intent_parser import IntentParser
from llm.llm_client import LLMClient
from memory.history import MemoryState, StepRecord
from memory.signature import descriptor_for_target, page_key, target_for_descriptor, view_signature
from models.action_models import AgentAction
from models.orchestration_models import ActionResult, AgentRunResult, AgentState, StepDecision


async def default_cli_printer(event_type: str, data: dict):
    if event_type == "log":
        message = data.get("message", "")
        level = data.get("level", "info").lower()
        if level == "success":
            logger.success(message)
        elif level == "error":
            logger.error(message)
        elif level == "warning":
            logger.warning(message)
        else:
            logger.info(message)
    elif event_type == "action":
        action = data.get("action")
        target = data.get("target")
        value = data.get("value")
        reasoning = data.get("reasoning")
        logger.info(f"Action: {str(action).upper()} | Target: {target} | Value: {value}")
        logger.info(f"Reasoning: {reasoning}")
    elif event_type == "extraction":
        key = data.get("key")
        value = data.get("value")
        logger.success(f"Extracted Data [{key}]: {value}")


class ReasoningAgent:
    def __init__(
        self,
        headless: bool = False,
        max_iterations: int = 15,
        model_name: Optional[str] = None,
        on_event: Optional[Callable[[str, dict], Awaitable[None]]] = None,
    ):
        self.browser_controller = BrowserController(headless=headless)
        self.llm_client = LLMClient(model_name=model_name)
        self.intent_parser = IntentParser(self.llm_client)
        self.context_builder = ContextBuilder()
        self.memory = MemoryState()
        self.recovery_policy = RecoveryPolicy()
        self.goal_verifier = GoalVerifier(self.llm_client)
        self.step_verifier = StepVerifier()
        self.max_iterations = max_iterations
        self.on_event = on_event if on_event is not None else default_cli_printer

    async def _emit(self, event_type: str, data: dict):
        if self.on_event:
            await self.on_event(event_type, data)

    async def _emit_log(self, message: str, level: str = "info"):
        await self._emit("log", {"message": message, "level": level})

    async def _emit_screenshot(self, name_prefix: str = "screenshot") -> Optional[str]:
        try:
            if self.browser_controller.page:
                os.makedirs("screenshots", exist_ok=True)
                filename = f"screenshots/{name_prefix}_{int(time.time())}.png"
                await self.browser_controller.page.screenshot(path=filename, full_page=False)
                return filename
        except Exception as e:
            logger.error(f"Failed to capture screenshot: {e}")
        return None

    def _agent_system_prompt(self, user_query: str) -> str:
        return f"""
        You are an autonomous web browser agent.
        Your goal is: {user_query}

        You only see elements currently visible in the viewport. If the target is not visible, use scroll.
        Prefer precise data-playwright-id targets from the context, such as pw-id-3.

        Supported actions:
        - click: click a target element.
        - type: fill a target input with value.
        - scroll: scroll the viewport or target element. Use value "down" or "up" when no target is needed.
        - wait: wait for value seconds.
        - navigate: go to the URL in value.
        - extract: extract text from target.
        - back: go to the previous page.
        - hover, drag_to, press_key, select: use only when required by the page control.
        - done: use only after the user goal is satisfied by visible page evidence or extracted data.

        If an element is hidden in an iframe, modal, captcha, or visual-only widget, set fallback_to_vision true.

        Respond ONLY with a valid JSON object matching this schema:
        {{
            "action": "<action_type>",
            "target": "<playwright_index_or_null>",
            "value": "<value_or_null>",
            "reasoning": "<brief_reasoning>",
            "fallback_to_vision": <boolean>
        }}
        """

    def _agent_user_prompt(self, state: AgentState) -> str:
        # Recall goes first: it is the most actionable thing the model is given.
        recalled = f"\n\n{state.recalled_context}\n\n" if state.recalled_context else ""
        semantic = "\n\nRelevant Semantic Memory:\n" + "\n---\n".join(state.semantic_memory) if state.semantic_memory else ""
        result = ""
        if state.last_result:
            result = f"""

        Last Action Result:
        success={state.last_result.success}
        action={state.last_result.action}
        target={state.last_result.target}
        error={state.last_result.error}
        recovery_hint={state.last_result.recovery_hint}
        url_after={state.last_result.url_after}
        """
        return f"{recalled}{state.memory_context}{semantic}{result}\n\nCurrent Page Context:\n{state.page_context}"

    async def _build_state(
        self,
        user_query: str,
        iteration: int,
        last_action: Optional[AgentAction],
        last_result: Optional[ActionResult],
    ) -> AgentState:
        dom_extractor = DOMExtractor(self.browser_controller.page)
        elements = await dom_extractor.extract_interactive_elements()
        await self._emit_log(f"Extracted {len(elements)} interactive DOM elements.")

        current_url = self.browser_controller.page.url
        self.memory.add_url(current_url)

        # Computed once per iteration and carried on the state: page identity
        # for storage keys, viewport signature for detecting that a step
        # changed something.
        current_page_key = page_key(current_url)
        current_view_signature = view_signature(current_url, elements)

        page_context = self.context_builder.build_context(current_url, elements)
        self.memory.index_page_content(current_url, page_context, current_page_key)
        memory_context = self.memory.get_context_string()
        semantic_memory = self.memory.semantic_search(user_query, n_results=2)

        # Recall what worked here before, and resolve each remembered target
        # back to the index this extraction gave it.
        recalled = []
        for remembered in self.memory.recall_actions(user_query, current_page_key):
            recalled.append({
                **remembered,
                "live_target": target_for_descriptor(remembered.get("target_descriptor"), elements),
            })
        routes = self.memory.recall_routes(current_page_key)
        recalled_context = self.context_builder.build_memory_recall(recalled, routes)
        if recalled_context:
            await self._emit_log(f"Recalled {len(recalled)} action(s) and {len(routes)} route(s) for this page.")

        return AgentState(
            user_query=user_query,
            iteration=iteration,
            current_url=current_url,
            page_key=current_page_key,
            view_signature=current_view_signature,
            elements=elements,
            page_context=page_context,
            memory_context=memory_context,
            recalled_context=recalled_context,
            semantic_memory=semantic_memory,
            last_action=last_action,
            last_result=last_result,
        )

    async def _decide_next_step(self, state: AgentState) -> Optional[StepDecision]:
        await self._emit_log("Querying LLM for next action...")
        response_json = await self.llm_client.generate_json(
            self._agent_system_prompt(state.user_query),
            self._agent_user_prompt(state),
        )
        if not response_json:
            await self._emit_log("Empty response from LLM. Retrying after wait.", "warning")
            return StepDecision(
                action=AgentAction(action="wait", value="2", reasoning="LLM returned an empty response."),
                source="orchestrator",
            )

        try:
            action = AgentAction(**response_json)
        except Exception as e:
            await self._emit_log(f"Invalid LLM action schema: {e}", "error")
            return None

        return StepDecision(action=action, source="llm", needs_verification=action.action == "done")

    async def _execute_with_recovery(
        self,
        executor: BrowserExecutor,
        action: AgentAction,
        state: AgentState,
    ) -> ActionResult:
        self.memory.add_action(action)
        self.memory.save_state()

        origin = "llm"
        if self.memory.detect_loop():
            await self._emit_log("Detected repeated action loop. Applying recovery policy.", "warning")
            action = self.recovery_policy.from_loop()
            origin = "loop_recovery"
            self.memory.add_action(action)
            await self._emit("action", action.model_dump())

        result = await self._execute_step(executor, action, state, origin, state.view_signature)

        if not result.success:
            recovery_action = self.recovery_policy.from_result(result)
            if recovery_action:
                await self._emit_log(f"Applying recovery action: {recovery_action.action}", "warning")
                self.memory.add_action(recovery_action)
                await self._emit("action", recovery_action.model_dump())
                # The recovery acts on the page as the failed action left it.
                result = await self._execute_step(
                    executor, recovery_action, state, "result_recovery", self._signature_after(result, state)
                )

        if (not result.success or action.fallback_to_vision) and action.action == "click":
            result = await self._run_vision_fallback(executor, action, state, result)

        return result

    async def _execute_step(
        self,
        executor: BrowserExecutor,
        action: AgentAction,
        state: AgentState,
        origin: str,
        signature_before: Optional[str] = None,
        screenshot_path: Optional[str] = None,
    ) -> ActionResult:
        """
        Execute one action, verify it did something, and record the three
        together.

        The id is minted per execution, not per iteration: a single iteration can
        execute the decided action, a loop correction, a recovery action and a
        vision retry, and each is a distinct step that needs its own outcome.
        """
        step_id = uuid.uuid4().hex[:12]
        before = signature_before or state.view_signature

        result = await executor.execute(action)
        result.step_id = step_id
        if screenshot_path:
            result.screenshot_path = screenshot_path

        after = None
        if self.step_verifier.needs_view_signature(action, result):
            after = await self._capture_view_signature()
        if after:
            # Carried so the next step in this iteration knows the state it
            # is acting on, rather than re-using the iteration's opening one.
            result.metadata["view_signature_after"] = after

        verdict = self.step_verifier.verify(action, result, before, after)
        await self._emit_log(
            f"Step {step_id} {action.action}: {verdict.status} ({verdict.evidence})",
            "success" if verdict.status == "verified" else "info",
        )

        # Only demonstrably-effective actions become memory. This gate is what
        # the verifier exists for.
        if verdict.status == "verified":
            descriptor = descriptor_for_target(action.target, state.elements)
            self.memory.record_verified_action(
                goal=state.user_query,
                page=state.page_key,
                action=action,
                descriptor=descriptor,
                evidence=verdict.evidence,
                url_before=result.url_before,
                url_after=result.url_after,
            )
            # A verified step that also landed on a different page is a route
            # worth remembering, not just an action.
            if result.url_after:
                self.memory.record_navigation(
                    from_page=state.page_key,
                    to_page=page_key(result.url_after),
                    action=action,
                    descriptor=descriptor,
                    evidence=verdict.evidence,
                )

        self.memory.add_step(
            StepRecord(
                step_id=step_id,
                iteration=state.iteration,
                origin=origin,
                action=action.model_copy(deep=True),
                result=result.model_copy(deep=True),
                verdict=verdict,
            )
        )
        return result

    async def _capture_view_signature(self) -> Optional[str]:
        """Re-read the viewport after an action, to compare against the before."""
        try:
            page = self.browser_controller.page
            if not page:
                return None
            elements = await DOMExtractor(page).extract_interactive_elements()
            return view_signature(page.url, elements)
        except Exception as e:
            logger.warning(f"Could not capture post-action view signature: {e}")
            return None

    @staticmethod
    def _signature_after(result: ActionResult, state: AgentState) -> Optional[str]:
        """The viewport as the previous step left it, falling back to the iteration's."""
        return result.metadata.get("view_signature_after") or state.view_signature

    async def _run_vision_fallback(
        self,
        executor: BrowserExecutor,
        action: AgentAction,
        state: AgentState,
        previous_result: ActionResult,
    ) -> ActionResult:
        await self._emit_log("Triggering vision fallback for click recovery.", "warning")
        screenshot_path = await self._emit_screenshot(name_prefix="vision_fallback")
        if not screenshot_path:
            previous_result.error = previous_result.error or "Failed to capture screenshot for vision fallback."
            return previous_result

        vision_prompt = self._agent_user_prompt(state)
        vision_json = await self.llm_client.generate_vision_json(screenshot_path, vision_prompt)
        if not vision_json or "x" not in vision_json or "y" not in vision_json:
            previous_result.error = previous_result.error or "Vision model did not return x/y coordinates."
            previous_result.screenshot_path = screenshot_path
            return previous_result

        vision_action = action.model_copy(update={"x": vision_json["x"], "y": vision_json["y"]})
        return await self._execute_step(
            executor,
            vision_action,
            state,
            "vision_fallback",
            self._signature_after(previous_result, state),
            screenshot_path,
        )

    async def _record_result(self, result: ActionResult, iteration: int):
        if result.action == "extract" and result.success:
            key = f"Extraction_Iter_{iteration}"
            self.memory.store_extracted_data(key, result.value)
            await self._emit_log(f"Extracted: {result.value}", "success")
            await self._emit("extraction", {"key": key, "value": result.value})

    async def execute_task(self, user_query: str) -> AgentRunResult:
        logger.info(f"Starting agent task: {user_query}")
        await self._emit_log(f"Starting task: {user_query}")

        last_action = None
        last_result = None
        iterations = 0

        try:
            if not self.llm_client.is_configured:
                await self._emit_log("GROQ_API_KEY is not configured. Add it to .env before running the agent.", "error")
                return AgentRunResult(completed=False, iterations=0, reason="missing_api_key")

            intent = await self.intent_parser.parse(user_query)
            await self._emit_log(f"Parsed Intent: {intent.model_dump_json()}")

            if not intent.website_url:
                import urllib.parse

                query_encoded = urllib.parse.quote(user_query)
                intent.website_url = f"https://duckduckgo.com/?q={query_encoded}"
                await self._emit_log(f"Falling back to web search for: {user_query}", "warning")

            await self._emit_log(f"Launching browser and navigating to {intent.website_url}...")
            success = await self.browser_controller.open_website(intent.website_url)
            if not success:
                await self._emit_log("Failed to load website. Aborting.", "error")
                return AgentRunResult(completed=False, iterations=0, reason="initial_navigation_failed")

            await self.browser_controller.wait_for_load()
            executor = BrowserExecutor(self.browser_controller.page)

            for iterations in range(1, self.max_iterations + 1):
                logger.info(f"--- Iteration {iterations} ---")
                await self._emit_log(f"--- Iteration {iterations} ---")

                state = await self._build_state(user_query, iterations, last_action, last_result)
                decision = await self._decide_next_step(state)
                if not decision:
                    last_result = ActionResult(success=False, action="invalid", error="Invalid LLM action.")
                    await asyncio.sleep(2)
                    continue

                action = decision.action
                await self._emit_log(
                    f"Decided Action: {action.action} on {action.target} (Reasoning: {action.reasoning})",
                    "success",
                )
                await self._emit("action", action.model_dump())

                if decision.needs_verification:
                    verified = await self.goal_verifier.verify(user_query, state.page_context, state.memory_context)
                    if verified:
                        await self._emit_log("Goal verified as complete.", "success")
                        return AgentRunResult(
                            completed=True,
                            iterations=iterations,
                            reason="goal_verified",
                            extracted_data=self.memory.extracted_data,
                            last_url=state.current_url,
                        )
                    await self._emit_log("Done was not verified; continuing with recovery scroll.", "warning")
                    action = self.recovery_policy.from_loop()

                last_result = await self._execute_with_recovery(executor, action, state)
                self.memory.add_result(last_result)
                self.memory.save_state()
                last_action = action
                await self._record_result(last_result, iterations)

                screenshot_path = await self._emit_screenshot(name_prefix="iteration")
                if screenshot_path:
                    last_result.screenshot_path = screenshot_path

                await self.browser_controller.wait_for_load()
                await asyncio.sleep(1)

            await self._emit_log("Reached maximum iterations before achieving the goal.", "warning")
            return AgentRunResult(
                completed=False,
                iterations=iterations,
                reason="max_iterations_reached",
                extracted_data=self.memory.extracted_data,
                last_url=self.browser_controller.page.url if self.browser_controller.page else None,
            )
        finally:
            if self.browser_controller.page:
                await self._emit_log("Task execution finished. Keeping browser open for 5 seconds for visual inspection.")
                await asyncio.sleep(5)
            else:
                await self._emit_log("Task execution finished.")
            await self.browser_controller.close_browser()
            self.memory.close()
