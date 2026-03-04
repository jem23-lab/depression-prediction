import asyncio
import os
import json
import google.generativeai as genai
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from datetime import datetime
from contextlib import AsyncExitStack

# ====== Configuration & Paths ======
base_path = os.path.dirname(__file__)
MEMORY_FILE = os.path.join(base_path, "long_term_memory.json")
LOG_BASE = os.path.join(base_path, "logs")

# Initialize Gemini
GOOGLE_API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=GOOGLE_API_KEY)
# We use gemini-2.5-flash for better reflection/reasoning capabilities
model = genai.GenerativeModel("gemini-2.5-flash")

# Server configs — map logical server name to script filename
# xai_server  exposes: predict_depression, explain_prediction, get_counterfactual
# knowledge_server exposes: query_knowledge
server_configs = {
    "xai": "servers/xai_server.py",
    "knowledge": "servers/knowledge_server.py",
    "logger": "servers/logger_server.py",
}


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
                pass
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
        with open(MEMORY_FILE, "w") as f:
            json.dump(self._lt_data, f, indent=2)

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
- shap_text_explain   → run SHAP token-level attribution to explain WHY a prediction was made
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
                summary = model.generate_content(summary_prompt).text
                memory.save_long_term(summary)
                print("\n[Session summary saved. Goodbye.]")
                break

            # ----------------------------------------------------------------
            # STEP 1: Session-Aware Intent Recognition & Tool Selection
            #
            # The plan prompt now includes:
            #   - cross-session long-term summary
            #   - recent chat turns (short-term)
            #   - prior tool calls + conclusions from THIS session (reasoning_steps)
            # This lets the planner avoid redundant calls and build on past work.
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

            plan_response = model.generate_content(plan_prompt).text
            try:
                # Clean and parse tool call
                tool_call = json.loads(plan_response.replace("```json", "").replace("```", "").strip())
                tool_name = tool_call["tool"]
                tool_args = tool_call.get("args", {})

                # ----------------------------------------------------------------
                # Auto-inject last prediction text into explain/counterfactual args
                # so the planner never needs to re-state the input text manually.
                # ----------------------------------------------------------------
                if tool_name in ("explain_prediction", "get_counterfactual"):
                    last_predict = next(
                        (s for s in reversed(memory.reasoning_steps) if s["tool"] == "predict_depression"),
                        None,
                    )
                    if last_predict and "text" not in tool_args:
                        tool_args["text"] = last_predict["args"].get("text", "")

                tool_meta = next(t for t in all_tools if t["name"] == tool_name)

                # ----------------------------------------------------------------
                # STEP 2: Execution & Intermediate Storage
                # ----------------------------------------------------------------
                result   = await tool_meta["session"].call_tool(tool_name, tool_args)
                raw_data = result.content[0].text
                memory.intermediate_outputs[tool_name] = raw_data

                # ----------------------------------------------------------------
                # STEP 3: Iterative Explanation Generation
                #
                # If there is a prior explanation for this same tool, inject it so
                # the model can refine/extend rather than start from scratch.
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
                draft_explanation = model.generate_content(explanation_prompt).text

                # ----------------------------------------------------------------
                # STEP 4: Reflection Mechanism
                #
                # The agent critiques its own draft against:
                #   - the short-term conversation context
                #   - any knowledge-base grounding already retrieved
                #   - the prior explanation (for consistency across turns)
                # It returns the refined explanation AND a flag indicating whether
                # a change was actually made — stored in reasoning_steps for audit.
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

                reflection_raw = model.generate_content(reflection_prompt).text.strip()

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
                print(f"[Orchestrator] Tool dispatch failed ({type(e).__name__}: {e}). Falling back to direct chat.")
                fallback_prompt = (
                    f"{SYSTEM_PROMPT}\n"
                    f"Session Memory: {memory.long_term_summary}\n"
                    f"{context_string}\n"
                    f"User: {user_input}"
                )
                response = model.generate_content(fallback_prompt).text
                print(f"\nAssistant > {response.strip()}")
                memory.add_turn("user", user_input)
                memory.add_turn("assistant", response)


if __name__ == "__main__":
    asyncio.run(run_agentic_dialogue())
