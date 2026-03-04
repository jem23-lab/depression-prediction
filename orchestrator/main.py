import asyncio
import os
import json
import google.generativeai as genai
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from llama_cpp import Llama
from datetime import datetime
from contextlib import AsyncExitStack

base_path = os.path.dirname(__file__)
log_base = os.path.join(base_path, "logs")
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
log_dir = os.path.join(log_base, f"session_{timestamp}")
os.makedirs(log_dir, exist_ok=True)

log_file_path = os.path.join(log_dir, "dialogue_log.txt")


def log_to_file(text):
    with open(log_file_path, "a", encoding="utf-8") as log_file:
        log_file.write(text + "\n")


# === System Prompt ===
SYSTEM_PROMPT = """
    You are an XAI assistant. You have access to specialized tools for depression prediction (xai), 
    clinical knowledge retrieval (knowledge), and session logging (logger).

    Decide which tools to call, in what order. Keep answers concise unless the user asks for depth. 
    When you use tools, explain their results in plain language and cite which tool you used (e.g., "(from SHAP)").

    - If the user asks "why", prefer SHAP tools. 
    - If they ask "how to change outcome", prefer counterfactuals. 
    - If they ask about meanings/definitions, use knowledge tools. 
    - If they provide text for assessment, call predict_depression.
"""

# === Server configurations ===
# Ensure paths are absolute or relative to this main.py file
server_configs = {
    "xai": "servers/xai_server.py",
    "knowledge": "servers/knowledge_server.py",
    "logger": "servers/logger_server.py",
}

GOOGLE_API_KEY = os.getenv("GEMINI_API_KEY", "your-api-key-here")
genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel("gemini-2.5-flash")  # Updated to existing model name


def llm_client(message: str):
    response = model.generate_content(f"{SYSTEM_PROMPT}\n\nUser query: {message}")
    return response.text.strip() if response.text else ""


def get_prompt_to_identify_tool_and_arguments(query, tools_list):
    tools_description = "\n".join(
        [f"- {t['name']}, {t['description']}, schema: {json.dumps(t['schema'])}"
         for t in tools_list]
    )

    return (
        f"You are a helpful assistant with access to these tools:\n\n"
        f"{tools_description}\n\n"
        f"User's Question: {query}\n\n"
        "Respond ONLY with this JSON format if a tool is needed:\n"
        "{\n"
        '    "tool": "tool-name",\n'
        '    "arguments": { "arg": "val" }\n'
        "}\n"
        "If no tool is needed, reply with a direct text answer."
    )


async def run_multi_server_dialogue():
    log_to_file("=== NEW MULTI-SERVER SESSION STARTED ===")

    # Use AsyncExitStack to manage multiple server connections simultaneously
    async with AsyncExitStack() as stack:
        sessions = {}
        all_tools = []

        # Start all servers and store their sessions
        for name, path in server_configs.items():
            print(f"Connecting to {name} server...")
            server_params = StdioServerParameters(command="python", args=[path])

            # Connect to the stdio transport
            transport = await stack.enter_async_context(stdio_client(server_params))
            read, write = transport

            # Initialize the session
            session = await stack.enter_async_context(ClientSession(read, write))
            await session.initialize()

            # Fetch tools and store with a reference back to the session
            server_tools = await session.list_tools()
            for tool in server_tools.tools:
                all_tools.append({
                    "name": tool.name,
                    "description": tool.description,
                    "schema": tool.inputSchema,
                    "session": session  # Map tool to its session
                })

            sessions[name] = session
            print(f"Connected to {name}. Loaded {len(server_tools.tools)} tools.")

        print("\nAll servers ready. Type 'exit' to quit.")

        while True:
                # genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
                #
                # print("Available models that support generateContent:")
                # for m in genai.list_models():
                #     if 'generateContent' in m.supported_generation_methods:
                #         print(f"- {m.name}")
            query = input("\nUser > ").strip()
            if query.lower() in ["exit", "quit"]:
                break

            # Identify which tool to use
            prompt = get_prompt_to_identify_tool_and_arguments(query, all_tools)
            llm_response = llm_client(prompt)

            # Check for tool calls
            try:
                # Basic cleaning for common JSON output issues
                cleaned = llm_response.replace("```json", "").replace("```", "").strip()
                tool_call = json.loads(cleaned)

                tool_name = tool_call.get("tool")
                # Find which session owns this tool
                tool_meta = next((t for t in all_tools if t["name"] == tool_name), None)

                if tool_meta:
                    session = tool_meta["session"]
                    result = await session.call_tool(tool_name, arguments=tool_call["arguments"])

                    # Generate natural language response
                    raw_output = result.content[0].text
                    explanation_prompt = f"User: {query}\nTool {tool_name} output: {raw_output}\n\nExplain this to the user naturally:"
                    friendly_response = model.generate_content(explanation_prompt).text

                    print(f"Assistant > {friendly_response.strip()}")
                    log_to_file(f"Assistant > {friendly_response.strip()}")
                else:
                    print(f"Assistant > {llm_response}")

            except (json.JSONDecodeError, KeyError):
                # If no tool was identified or JSON failed, treat as direct chat
                print(f"Assistant > {llm_response}")
                log_to_file(f"Assistant > {llm_response}")


if __name__ == "__main__":
    asyncio.run(run_multi_server_dialogue())