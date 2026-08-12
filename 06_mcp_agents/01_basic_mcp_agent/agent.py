from langchain_mcp_adapters.client import MultiServerMCPClient
import os
import asyncio
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from dotenv import load_dotenv

load_dotenv(dotenv_path=r"..\..\openai-venv\.env")
API_KEY=os.getenv('GROQ_API_KEY') 
async def main():

    client = MultiServerMCPClient(
        {
            "my_tools": {
                "command": "python",
                "args": ["server.py"],
                "transport": "stdio",
            }
        }
    )

    llm = ChatOpenAI(
        model="llama-3.3-70b-versatile",
        api_key=API_KEY,
        base_url="https://api.groq.com/openai/v1"
    )

    tools= await client.get_tools()

    agent = create_agent(llm, tools)

    response=await agent.ainvoke({
        "messages": [("user", "Calculate the trace for matrix [[1, 2], [3, 4]]")]
    })

    print("\nAgent Output:")
    print(response["messages"][-1].content)

if __name__ == "__main__":
    asyncio.run(main())