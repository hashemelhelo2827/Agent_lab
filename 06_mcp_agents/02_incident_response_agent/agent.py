import os
import asyncio
from typing import TypedDict, Annotated, Optional
from dotenv import load_dotenv
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, ToolMessage, AIMessage
from langgraph.graph.message import add_messages
from langgraph.prebuilt import create_react_agent

# Load Environment Variables
load_dotenv(dotenv_path=r'../../openai-venv/.env')
API_KEY = os.getenv('Mistral_API_key')


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
            "command": "npx",
            "args": ["-y", "@modelcontextprotocol/server-memory"],
            "transport": "stdio",
        }
    }
)

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


model = ChatOpenAI(
    model="mistral-medium-2604",
    api_key=API_KEY,
    base_url="https://api.mistral.ai/v1"
)


def display_content(content) -> str:
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict) and "text" in block:
                parts.append(block["text"])
            else:
                parts.append(str(block))
        return "\n".join(parts)
    return str(content)


async def main():
   
    tools = await client.get_tools()
    
    app = create_react_agent(
        model=model,
        tools=tools,
        prompt=SYSTEM_PROMPT
    )
    
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
                            
                state = {
                    "messages": event["messages"],
                    "active_incident_id": event.get("active_incident_id")
                }
                
        except TimeoutError:
            print("\n[error] Turn timed out after 120s")
        except Exception as e:
            print(f"\n[error] {e}")


if __name__ == "__main__":
    asyncio.run(main())