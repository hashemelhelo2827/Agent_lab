from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage, AIMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from dotenv import load_dotenv, find_dotenv
import os
import asyncio
import json
import time
import logging
from typing import TypedDict, Annotated, Optional
from openai import RateLimitError
from google.genai.errors import ClientError

load_dotenv(find_dotenv(usecwd=True)) or load_dotenv(r'../../openai-venv/.env')
GEMINI_KEY = os.getenv('GEMINI_API_KEY2')
GROQ_KEY = os.getenv('GROQ_API_KEY')
if not (GEMINI_KEY or GROQ_KEY):
    raise RuntimeError(
        "No API key found. Add GEMINI_API_KEY2 or GROQ_API_KEY to the .env file."
    )

if GEMINI_KEY:
    os.environ["GOOGLE_API_KEY"] = GEMINI_KEY
logging.getLogger("google_genai._api_client").setLevel(logging.ERROR)
logging.getLogger("langchain_google_genai._function_utils").setLevel(logging.ERROR)

providers = []
if GEMINI_KEY:
    providers.append(("gemini", ChatGoogleGenerativeAI(
        model='gemini-3.5-flash',
        api_key=GEMINI_KEY,
        retries=0,
        request_timeout=30,
    )))
if GROQ_KEY:
    providers.append(("groq", ChatOpenAI(
        model='llama-3.3-70b-versatile',
        api_key=GROQ_KEY,
        base_url='https://api.groq.com/openai/v1'
    )))


def is_quota_error(e) -> bool:
    if isinstance(e, RateLimitError):
        return True
    if isinstance(e, ClientError) and getattr(e, "code", None) == 429:
        return True
    return False
client = MultiServerMCPClient(
    {
       "system_tools": {
                       "command": "python",
                        "args": ["system_mcp_server.py"],
                        "transport": "stdio",
                        },
       "db_tools": {
                       "command": "python",
                        "args": ["db_mcp_server.py"],
                        "transport": "stdio",
                        },
       "memory_tools": {
                    'command': 'npx',
                    'args': ["-y", "@modelcontextprotocol/server-memory"],
                    "transport": "stdio",
                    }
    })

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    active_incident_id: Optional[str]

SYSTEM_PROMPT = """You are an IT incident-response and systems operations agent.
Available tools:
- System monitoring: read_system_metrics, list_directory_logs
- Incident database: query_incidents, log_incident_resolution
- Long-term memory: create_entities, create_relations, add_observations, search_nodes

Call a tool whenever the answer requires current or stored data. Always consult
memory before repeating prior context. Call each tool at most once per turn unless
truly necessary. Be concise and report numbers exactly."""

MAX_TOOL_CALLS = 8


def invoke_llm(messages, max_retries=3):
    """Call the LLM across providers, retrying transient errors and failing over."""
    for name, model in providers:
        for attempt in range(max_retries):
            try:
                return model.invoke([SystemMessage(content=SYSTEM_PROMPT)] + messages)
            except Exception as e:
                if is_quota_error(e):
                    break
                if attempt == max_retries - 1:
                    break
                time.sleep(2 * (attempt + 1))
        print(f"[warn] provider '{name}' failed, trying next...")
    raise RuntimeError("All LLM providers failed (quota exhausted or errors). Retry later.")


def build_agent(tools):
    for i, (name, model) in enumerate(providers):
        providers[i] = (name, model.bind_tools(tools))

    def call_model(state: AgentState) -> dict:
        response = invoke_llm(state["messages"])
        return {"messages": [response]}

    def extract_incident_id(result) -> Optional[str]:
        def find_incident(obj):
            if isinstance(obj, dict):
                if "id" in obj and "status" in obj and obj.get("status") != "RESOLVED":
                    return str(obj["id"])
                for v in obj.values():
                    found = find_incident(v)
                    if found:
                        return found
            elif isinstance(obj, list):
                for item in obj:
                    found = find_incident(item)
                    if found:
                        return found
            return None

        if isinstance(result, str):
            try:
                result = json.loads(result)
            except json.JSONDecodeError:
                return None
        for block in result if isinstance(result, list) else [result]:
            if isinstance(block, dict) and block.get("type") == "text":
                text = block.get("text", "")
                if isinstance(text, str):
                    try:
                        parsed = json.loads(text)
                    except json.JSONDecodeError:
                        continue
                    found = find_incident(parsed)
                    if found:
                        return found
        return None

    async def call_tools(state: AgentState) -> dict:
        tool_map = {t.name: t for t in tools}
        last = state["messages"][-1]
        tool_messages = []
        active_id = state.get("active_incident_id")
        for tc in last.tool_calls[:MAX_TOOL_CALLS]:
            tool = tool_map.get(tc["name"])
            if tool is None:
                tool_messages.append(ToolMessage(content="Tool not found", tool_call_id=tc["id"], name=tc["name"]))
                continue
            result = None
            for attempt in range(3):
                try:
                    result = await tool.ainvoke(tc["args"])
                    break
                except Exception as e:
                    if is_quota_error(e):
                        raise RuntimeError(
                            "LLM provider quota reached. Retry later or switch model/key."
                        ) from e
                    if attempt == 2:
                        result = {"error": f"{type(e).__name__}: {str(e)[:200]}"}
                    else:
                        await asyncio.sleep(2 * (attempt + 1))
            if tc["name"] == "query_incidents":
                found = extract_incident_id(result)
                if found:
                    active_id = found
            elif tc["name"] == "log_incident_resolution":
                active_id = None
            tool_messages.append(ToolMessage(content=str(result), tool_call_id=tc["id"], name=tc["name"]))
        update = {"messages": tool_messages}
        if active_id != state.get("active_incident_id"):
            update["active_incident_id"] = active_id
        return update

    def should_continue(state: AgentState) -> str:
        if not state["messages"][-1].tool_calls:
            return END
        tool_msgs = sum(1 for m in state["messages"] if isinstance(m, ToolMessage))
        if tool_msgs > MAX_TOOL_CALLS * 3:
            return END
        return "tools"

    graph = StateGraph(AgentState)
    graph.add_node("agent", call_model)
    graph.add_node("tools", call_tools)
    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
    graph.add_edge("tools", "agent")
    return graph.compile()


def display_content(content) -> str:
    """Render AIMessage/ToolMessage content (str or list of blocks) as plain text."""
    if isinstance(content, str):
        return content
    parts = []
    for block in content if isinstance(content, list) else [content]:
        if isinstance(block, str):
            parts.append(block)
        elif isinstance(block, dict):
            if block.get("type") == "text":
                parts.append(block.get("text", ""))
            else:
                parts.append(str(block))
        else:
            parts.append(str(block))
    return "\n".join(p for p in parts if p)


async def main():
    tools = await client.get_tools()
    app = build_agent(tools)
    print("Agent ready. Tools:", ", ".join(sorted(t.name for t in tools)))
    print("Type 'exit' to quit.")

    state: AgentState = {"messages": [], "active_incident_id": None}
    while True:
        user = input("\nYou: ").strip()
        if user.lower() in {"exit", "quit"}:
            break
        state["messages"] = [*state["messages"], HumanMessage(content=user)]
        try:
            async with asyncio.timeout(120):
                async for event in app.astream(state, stream_mode="values"):
                    last = event["messages"][-1]
                    if isinstance(last, (AIMessage, ToolMessage)):
                        content = display_content(last.content)
                        if isinstance(last, ToolMessage):
                            content = f"  [tool {last.name}] -> {content}"
                        if content.strip():
                            print(content)
                state = {"messages": event["messages"], "active_incident_id": event.get("active_incident_id")}
        except TimeoutError:
            print("\n[error] turn timed out after 120s")
        except RuntimeError as e:
            print(f"\n[error] {e}")


if __name__ == "__main__":
    asyncio.run(main())
