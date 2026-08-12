import asyncio
import sys
sys.path.insert(0, r"C:\Users\hashe\Desktop\_\Agent_lab\06_\2")
import agent
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

SCENARIOS = [
    ("system", "Check the system CPU and memory usage"),
    ("incidents", "List all open incidents"),
    ("resolve", "Resolve incident #2 with summary 'diagnostics complete, fixed'"),
    ("memory-write", "Remember that the database host is DB-PROD-01"),
    ("memory-read", "What do you remember about DB-PROD-01?"),
]

async def run_scenario(app, scenario):
    state = {"messages": [HumanMessage(content=scenario)], "active_incident_id": None}
    steps = 0
    tools_called = []
    final_answer = ""
    last_event = None
    async for event in app.astream(state, stream_mode="values"):
        last_event = event
        last = event["messages"][-1]
        if isinstance(last, AIMessage):
            if last.tool_calls:
                tools_called.extend(tc["name"] for tc in last.tool_calls)
            elif last.content:
                final_answer = str(last.content)
        elif isinstance(last, ToolMessage) and last.name:
            tools_called.append(last.name)
        steps += 1
        if steps > 15:
            raise RuntimeError("runaway loop")
    return tools_called, final_answer, last_event.get("active_incident_id") if last_event else None

async def main():
    tools = await agent.client.get_tools()
    app = agent.build_agent(tools)
    print(f"Loaded {len(tools)} tools", flush=True)

    for it in range(1, 11):
        failures = 0
        print(f"=== Iteration {it} ===", flush=True)
        for name, scenario in SCENARIOS:
            try:
                called, answer, act_id = await run_scenario(app, scenario)
                if not called and not answer:
                    raise RuntimeError("no tools called, no answer")
                print(f"  [OK] {name}: tools={called} act_id={act_id}", flush=True)
            except Exception as e:
                failures += 1
                print(f"  [FAIL] {name}: {type(e).__name__}: {str(e)[:300]}", flush=True)
        print(f"  -> {len(SCENARIOS) - failures}/{len(SCENARIOS)} passed\n", flush=True)
        if failures == 0:
            print("All clean — stopping early after full pass.", flush=True)
            break

asyncio.run(main())
