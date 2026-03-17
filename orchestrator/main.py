import asyncio
import os
import json
import google.generativeai as genai
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from datetime import datetime
from contextlib import AsyncExitStack
import logging

# ====== Configuration & Paths ======
base_path = os.path.dirname(__file__)
MEMORY_FILE = os.path.join(base_path, "long_term_memory.json")
LOG_BASE = os.path.join(base_path, "logs")

# Ensure directories exist for logs and memory
os.makedirs(LOG_BASE, exist_ok=True)
os.makedirs(os.path.dirname(MEMORY_FILE), exist_ok=True)

# Logging setup
logfile = os.path.join(LOG_BASE, "orchestrator.log")
logging.basicConfig(
    filename=logfile,
    level=logging.DEBUG,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# Timeouts (seconds) — conservative defaults, can be tuned
GENERATION_TIMEOUT = float(os.getenv("GENERATION_TIMEOUT", "30"))
TOOL_CALL_TIMEOUT = float(os.getenv("TOOL_CALL_TIMEOUT", "20"))

# Initialize Gemini
GOOGLE_API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=GOOGLE_API_KEY)
# We use gemini-2.5-flash for better reflection/reasoning capabilities
model = genai.GenerativeModel("gemini-2.5-flash")


# Async wrapper for blocking model calls with timeout and logging
async def a_generate_content(prompt: str, **kwargs) -> str:
    """Run model.generate_content in a thread and return .text, with timeout.

    Raises asyncio.TimeoutError on timeout and re-raises other exceptions.
    """
    logger.debug("a_generate_content: prompt length=%d", len(prompt) if prompt is not None else 0)

    try:
        # run in thread to avoid blocking the event loop
        res = await asyncio.wait_for(asyncio.to_thread(lambda: model.generate_content(prompt, **kwargs)), timeout=GENERATION_TIMEOUT)
    except asyncio.TimeoutError:
        logger.warning("Model generation timed out after %s seconds", GENERATION_TIMEOUT)
        raise
    except Exception:
        logger.exception("Model generation raised an unexpected exception")
        raise

    # defensive: ensure .text exists
    text = getattr(res, "text", None)
    if text is None:
        logger.warning("Model response has no .text attribute; falling back to str(response)")
        text = str(res)
    logger.debug("a_generate_content: response length=%d", len(text))
    return text

# Server configs — map logical server name to script filename
# xai_server  exposes: predict_depression, shap_text_explain, generate_text_counterfactual, get_top_contributing_words
# knowledge_server exposes: query_knowledge
# logger_server exposes: log_event
server_configs = {
    "xai": "servers/xai_server.py",
    "knowledge": "servers/knowledge_server.py",
    "logger": "servers/logger_server.py",
}


# Canonical tool aliases (orchestrator preferred names -> server registered names)
TOOL_ALIASES = {
    "predict_depression": "predict_depression",
    "explain_prediction": "shap_text_explain",
    "get_counterfactual": "generate_text_counterfactual",
    "top_contributing_words": "get_top_contributing_words",
    "query_knowledge": "query_knowledge",
    "log_event": "log_event",
}


# Helper: normalize and extract tool results robustly
def _extract_tool_result(result) -> dict:
    """Normalize various tool call result formats into a dict:
    returns {'ok': True, 'raw': str, 'json': dict|None}
    """
    raw_text = None
    parsed = None
    try:
        # Common mcp client result has .content which is a list of objects with .text
        content = getattr(result, "content", None)
        if isinstance(content, list) and len(content) and hasattr(content[0], "text"):
            raw_text = content[0].text
        elif isinstance(content, str):
            raw_text = content
        else:
            raw_text = str(result)

        # try parse JSON
        try:
            parsed = json.loads(raw_text)
        except Exception:
            parsed = None
    except Exception:
        logger.exception("Failed to extract tool result")
        raw_text = str(result)
    return {"ok": True, "raw": raw_text, "json": parsed}


# ====== 1. Conversational Memory Management ======

class SessionMemory:
    """
    Manages three memory layers:
      - short_term       : rolling window of recent chat turns (role + content)
      - reasoning_steps  : per-turn log of tool calls, raw results, draft & final
                           explanations — feeds session-aware orchestration
      - long_term        : persistent JSON across sessions; stores per-session
                           summaries rather than a single overwritten string
    """

    MAX_SHORT_TERM_TURNS = 10   # chat turns kept in context window
    MAX_REASONING_STEPS  = 5    # recent tool-call records injected into planner

    def __init__(self):
        self.short_term:      list[dict] = []   # {role, content}
        self.reasoning_steps: list[dict] = []   # {turn, tool, args, raw, draft, final, refined}
        self.intermediate_outputs: dict  = {}   # tool_name -> latest raw result (kept for reflection)

        # ---- long-term: load all past session summaries ----
        self._lt_data = self._load_long_term_file()
        # Provide a compact summary string for prompt injection
        self.long_term_summary: str = self._build_lt_summary_string()

    # ------------------------------------------------------------------
    # Long-term helpers
    # ------------------------------------------------------------------

    def _load_long_term_file(self) -> dict:
        """Load the full long-term JSON file; return empty structure on miss."""
        if os.path.exists(MEMORY_FILE):
            try:
                with open(MEMORY_FILE, "r") as f:
                    return json.load(f)
            except json.JSONDecodeError:
                logger.warning("Long-term memory file exists but contains invalid JSON: %s", MEMORY_FILE)
            except Exception:
                logger.exception("Error reading long-term memory file: %s", MEMORY_FILE)
        return {"sessions": []}

    def _build_lt_summary_string(self) -> str:
        """
        Condense the last 3 session summaries into a single string for prompt use.
        This gives the planner cross-session continuity without token overload.
        """
        sessions = self._lt_data.get("sessions", [])
        if not sessions:
            return "No previous session history."
        recent = sessions[-3:]  # last 3 sessions
        parts = [
            f"[{s.get('date', 'unknown')}] {s.get('summary', '')}"
            for s in recent
        ]
        return "\n".join(parts)

    def save_long_term(self, summary: str):
        """Append a new session summary entry and persist to disk."""
        sessions = self._lt_data.setdefault("sessions", [])
        sessions.append({
            "date": str(datetime.now().strftime("%Y-%m-%d %H:%M")),
            "summary": summary,
        })
        # Keep only the last 20 sessions to avoid unbounded growth
        self._lt_data["sessions"] = sessions[-20:]
        try:
            with open(MEMORY_FILE, "w") as f:
                json.dump(self._lt_data, f, indent=2)
            logger.info("Saved long-term memory to %s", MEMORY_FILE)
        except Exception:
            logger.exception("Failed to write long-term memory to %s", MEMORY_FILE)

    # ------------------------------------------------------------------
    # Short-term chat memory
    # ------------------------------------------------------------------

    def add_turn(self, role: str, content: str):
        """Add a chat turn; evict oldest when window is full."""
        self.short_term.append({"role": role, "content": content})
        if len(self.short_term) > self.MAX_SHORT_TERM_TURNS:
            self.short_term.pop(0)

    def get_context_string(self) -> str:
        """Return recent chat turns as a formatted string for prompt injection."""
        return "\n".join(
            f"{m['role'].capitalize()}: {m['content']}"
            for m in self.short_term
        )

    # ------------------------------------------------------------------
    # Reasoning-step memory (session-aware orchestration)
    # ------------------------------------------------------------------

    def log_reasoning_step(
        self,
        tool: str,
        args: dict,
        raw_result: str,
        draft: str,
        final_output: str,
        was_refined: bool,
    ):
        """
        Record a complete reasoning step so the planner can reference prior
        tool invocations and explanations within the same session.
        """
        step = {
            "turn":      len(self.short_term),   # approximate turn index
            "tool":      tool,
            "args":      args,
            "raw":       raw_result[:300],        # truncate for token budget
            "draft":     draft[:300],
            "final":     final_output[:300],
            "refined":   was_refined,
        }
        self.reasoning_steps.append(step)
        if len(self.reasoning_steps) > self.MAX_REASONING_STEPS:
            self.reasoning_steps.pop(0)

    def get_reasoning_context(self) -> str:
        """
        Compact summary of recent reasoning steps injected into the planner.
        Lets the orchestrator know what tools were already called and what was
        concluded, enabling session-aware tool selection.
        """
        if not self.reasoning_steps:
            return "No tool calls made yet this session."
        lines = []
        for s in self.reasoning_steps:
            refined_tag = " [refined]" if s["refined"] else ""
            lines.append(
                f"- Tool: {s['tool']} | Args: {s['args']} | "
                f"Conclusion{refined_tag}: {s['final'][:120]}..."
            )
        return "\n".join(lines)

    def get_last_explanation(self, tool: str) -> str | None:
        """
        Retrieve the most recent final explanation for a given tool.
        Used by the iterative refinement step to seed follow-up reasoning.
        """
        for step in reversed(self.reasoning_steps):
            if step["tool"] == tool:
                return step["final"]
        return None


# ====== 2. Refined System Prompts ======

SYSTEM_PROMPT = """
You are an Agentic XAI Clinical Assistant specialising in depression assessment.
Your goal is to provide explainable depression assessments by orchestrating four specialised tools.

TOOL ROLES:
- predict_depression   → classify a user statement into a depression severity class
- shap_text_eclear
xplain   → run SHAP token-level attribution to explain WHY a prediction was made
- generate_text_counterfactual   → use DiCE/TextAttack to show WHAT WOULD CHANGE the prediction
- query_knowledge      → retrieve DSM-5 / ICD-11 clinical grounding via RAG

REASONING GUIDELINES:
1. Always predict first before explaining or generating counterfactuals.
2. Reference memory: use previous session summaries and recent turns for continuity.
3. Explainability: interpret tool outputs using clinical context, not raw scores alone.
"""


# ====== 3. The Agentic Loop with Reflection ======

async def run_agentic_dialogue():
    memory = SessionMemory()
    print(f"--- Session Started ---")
    logger.info("Session started. Long-term context: %s", memory.long_term_summary[:200])
    print(f"Long-term context:\n{memory.long_term_summary[:200]}...\n")

    async with AsyncExitStack() as stack:
        # Start Servers
        sessions = {}
        all_tools = []
        for name, path in server_configs.items():
            server_params = StdioServerParameters(command="python", args=["-u", path])
            transport = await stack.enter_async_context(stdio_client(server_params))
            session = await stack.enter_async_context(ClientSession(transport[0], transport[1]))
            await session.initialize()

            tools = await session.list_tools()
            for t in tools.tools:
                all_tools.append({"name": t.name, "desc": t.description, "schema": t.inputSchema, "session": session})
            sessions[name] = session

        # Build alias resolution map for orchestrator-preferred names -> actual registered tool names
        available_tools = {t["name"]: t for t in all_tools}
        alias_to_actual = {}
        for canon, registered in TOOL_ALIASES.items():
            if registered in available_tools:
                alias_to_actual[canon] = registered
            elif canon in available_tools:
                alias_to_actual[canon] = canon
            else:
                logger.warning("Tool alias '%s' (registered='%s') not found among available tools", canon, registered)

        while True:
            user_input = input("\nUser > ").strip()
            if user_input.lower() in ["exit", "quit"]:
                # Generate a summary for long-term memory before quitting
                summary_prompt = (
                    f"Summarize this clinical interaction for continuity in future sessions. "
                    f"Include key topics discussed, tools used, and any conclusions reached.\n"
                    f"Conversation:\n{memory.get_context_string()}\n"
                    f"Reasoning steps:\n{memory.get_reasoning_context()}"
                )
                try:
                    summary = await a_generate_content(summary_prompt)
                except asyncio.TimeoutError:
                    logger.warning("Summary generation timed out; saving minimal summary")
                    summary = "Session ended (summary timed out)."
                except Exception:
                    logger.exception("Failed to generate session summary; using placeholder")
                    summary = "Session ended (summary generation failed)."

                memory.save_long_term(summary)
                print("\n[Session summary saved. Goodbye.]")
                break

            # ----------------------------------------------------------------
            # STEP 1: Session-Aware Intent Recognition & Tool Selection
            # ----------------------------------------------------------------
            context_string     = memory.get_context_string()
            reasoning_context  = memory.get_reasoning_context()

            plan_prompt = f"""
                {SYSTEM_PROMPT}
                
                === Cross-Session Memory ===
                {memory.long_term_summary}
                
                === Recent Conversation (Short-Term) ===
                {context_string}
                
                === Prior Tool Calls This Session ===
                {reasoning_context}
                
                === Current User Input ===
                {user_input}
                
                === Available Tools ===
                {chr(10).join(
                    f'- {t["name"]}: {t["desc"]}  |  args schema: {json.dumps(t["schema"])}'
                    for t in all_tools
                )}
                
                Tool selection rules:
                - Use "predict_depression"   : when the user shares a statement / feeling and wants a diagnosis or class label.
                - Use "explain_prediction"   : when the user asks WHY, WHAT FACTORS, or WHICH WORDS drove the prediction (SHAP).
                - Use "get_counterfactual"   : when the user asks WHAT IF, HOW TO IMPROVE, or wants alternative phrasing (DiCE).
                - Use "query_knowledge"      : when the user asks about clinical criteria, DSM-5/ICD-11 definitions, or background.
                - If this is a follow-up question about a topic already explained (see Prior Tool Calls),
                  prefer the same tool so the explanation can be iteratively refined.
                - Do NOT repeat a tool call with identical args if the result is already in Prior Tool Calls.
                - Respond ONLY with valid JSON: {{"tool": "tool_name", "args": {{...}}}}
                """

            try:
                plan_response = await a_generate_content(plan_prompt)
                logger.debug("Plan response: %s", plan_response[:1000])
                # Clean and parse tool call
                tool_call = json.loads(plan_response.replace("```json", "").replace("```", "").strip())
                tool_name = tool_call["tool"]
                tool_args = tool_call.get("args", {})

                # ----------------------------------------------------------------
                # Auto-inject last prediction text into explain/counterfactual args
                # ----------------------------------------------------------------
                if tool_name in ("explain_prediction", "get_counterfactual"):
                    last_predict = next(
                        (s for s in reversed(memory.reasoning_steps) if s["tool"] == "predict_depression"),
                        None,
                    )
                    if last_predict and "text" not in tool_args:
                        tool_args["text"] = last_predict["args"].get("text", "")

                # Resolve alias -> actual registered tool name
                actual_tool = alias_to_actual.get(tool_name, tool_name)
                if actual_tool not in available_tools:
                    logger.error("Requested tool '%s' resolved to '%s' which is not available", tool_name, actual_tool)
                    raise RuntimeError(f"Tool not available: {tool_name} -> {actual_tool}")

                tool_meta = available_tools[actual_tool]

                # ----------------------------------------------------------------
                # STEP 2: Execution & Intermediate Storage
                # ----------------------------------------------------------------
                try:
                    logger.info("Calling tool '%s' (resolved=%s) with args=%s", tool_name, actual_tool, tool_args)
                    raw_result_obj = await asyncio.wait_for(tool_meta["session"].call_tool(actual_tool, tool_args), timeout=TOOL_CALL_TIMEOUT)
                except asyncio.TimeoutError:
                    logger.warning("Tool call '%s' timed out after %s seconds", actual_tool, TOOL_CALL_TIMEOUT)
                    raise
                except Exception:
                    logger.exception("Tool call '%s' failed", actual_tool)
                    raise

                # robust extraction of tool output
                extracted = _extract_tool_result(raw_result_obj)
                raw_data = extracted.get("raw")
                parsed_json = extracted.get("json")

                memory.intermediate_outputs[tool_name] = raw_data
                logger.debug("Tool '%s' (resolved=%s) returned data length=%d", tool_name, actual_tool, len(raw_data) if raw_data is not None else 0)

                # ----------------------------------------------------------------
                # STEP 3: Iterative Explanation Generation
                # ----------------------------------------------------------------
                prior_explanation = memory.get_last_explanation(tool_name)
                if prior_explanation:
                    iterative_note = (
                        f"\nPrior Explanation (from earlier in this session):\n{prior_explanation}\n"
                        f"The user has asked a follow-up. Build on or refine the prior explanation."
                    )
                else:
                    iterative_note = ""

                explanation_prompt = (
                    f"User: {user_input}\n"
                    f"Tool Result: {raw_data}\n"
                    f"{iterative_note}\n"
                    f"Provide a natural, clinically grounded explanation:"
                )
                try:
                    draft_explanation = await a_generate_content(explanation_prompt)
                except asyncio.TimeoutError:
                    logger.warning("Explanation generation timed out for tool '%s'", tool_name)
                    draft_explanation = "(explanation generation timed out)"

                # ----------------------------------------------------------------
                # STEP 4: Reflection Mechanism
                # ----------------------------------------------------------------
                reflection_prompt = f"""
                    You are reviewing a draft clinical explanation for quality and consistency.

                    === Draft Explanation ===
                    {draft_explanation}

                    === Conversation Context ===
                    {context_string}

                    === Clinical Knowledge Grounding ===
                    {memory.intermediate_outputs.get('query_knowledge', 'Not yet retrieved.')}

                    === Prior Explanation for This Topic ===
                    {prior_explanation or 'None — this is the first explanation for this tool.'}

                    Review checklist:
                    1. Does the draft contradict any clinical knowledge or previous explanations?
                    2. Is it clear and jargon-free for a clinician?
                    3. Does it meaningfully build on prior context rather than repeating it?

                    If the draft passes all checks, output exactly:
                    NO_CHANGE: <repeat the draft>

                    If refinement is needed, output exactly:
                    REFINED: <your improved explanation>
                    """

                try:
                    reflection_raw = (await a_generate_content(reflection_prompt)).strip()
                except asyncio.TimeoutError:
                    logger.warning("Reflection generation timed out for tool '%s'", tool_name)
                    reflection_raw = "NO_CHANGE: " + draft_explanation

                # Parse reflection output to detect whether a change was made
                if reflection_raw.startswith("REFINED:"):
                    final_output = reflection_raw[len("REFINED:"):].strip()
                    was_refined  = True
                else:
                    # "NO_CHANGE:" prefix or unexpected format → use draft
                    final_output = draft_explanation.strip()
                    was_refined  = False

                # ----------------------------------------------------------------
                # Store the complete reasoning step for future session-awareness
                # ----------------------------------------------------------------
                memory.log_reasoning_step(
                    tool=tool_name,
                    args=tool_args,
                    raw_result=raw_data,
                    draft=draft_explanation,
                    final_output=final_output,
                    was_refined=was_refined,
                )

                refined_tag = " [refined ✓]" if was_refined else ""
                print(f"\nAssistant{refined_tag} > {final_output}")
                memory.add_turn("user", user_input)
                memory.add_turn("assistant", final_output)

            except Exception as e:
                # Log the parse error for debugging, then fall back to direct chat
                logger.exception("Orchestrator tool dispatch failed: %s", e)
                print(f"[Orchestrator] Tool dispatch failed ({type(e).__name__}: {e}). Falling back to direct chat.")
                fallback_prompt = (
                    f"{SYSTEM_PROMPT}\n"
                    f"Session Memory: {memory.long_term_summary}\n"
                    f"{context_string}\n"
                    f"User: {user_input}"
                )
                try:
                    response = await a_generate_content(fallback_prompt)
                except asyncio.TimeoutError:
                    logger.warning("Fallback generation timed out; sending minimal reply")
                    response = "I'm sorry, I'm having trouble responding right now. Please try again."
                except Exception:
                    logger.exception("Fallback generation failed")
                    response = "I'm sorry, I'm having trouble responding right now."

                print(f"\nAssistant > {response.strip()}")
                memory.add_turn("user", user_input)
                memory.add_turn("assistant", response)


if __name__ == "__main__":
    asyncio.run(run_agentic_dialogue())
