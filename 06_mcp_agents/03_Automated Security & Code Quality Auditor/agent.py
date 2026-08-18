import os
import asyncio
from dotenv import load_dotenv
from pydantic import BaseModel, Field

from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import JsonOutputParser
from langchain.agents import create_agent


class OutputSchema(BaseModel):
    syntax: dict = Field(description="Output of the analyze_syntax function")
    secu_debug: dict = Field(description="Output of the run_linter function")
    code_edited: dict = Field(description="Output of the apply_patch function")
    markdown: dict = Field(description="Output of the create_audit_log function")

load_dotenv(dotenv_path=r'../../openai-venv/.env')
API_KEY = os.getenv("Mistral_API_key")

model = ChatOpenAI(
    model="mistral-medium-2604",
    base_url="https://api.mistral.ai/v1",
    api_key=API_KEY
)

SYSTEM_PROMPT = """You are an expert AI Code Security Auditor & Quality Engineer.
Your primary role is to inspect Python source code for security vulnerabilities, syntax bugs, and anti-patterns, then apply refactored patches and generate audit documentation.

### WORKFLOW INSTRUCTIONS:
1. **Initial Inspection**:
   - Run `analyze_syntax` with the target file path to verify the code parses correctly and extract structural stats.
   - Run `run_linter` to identify security flaws (e.g., hardcoded secrets, `eval()`/`exec()` calls, bare `except:` blocks).

2. **Patch & Refactor**:
   - If security vulnerabilities or anti-patterns are found, write a clean, secure version of the file that removes all risks while preserving business logic.
   - Call `apply_patch` to apply the updated, refactored code directly to the file.

3. **Documentation**:
   - Call `create_audit_log` to save a structured Markdown audit report documenting the findings and fixes.

4. **Output Format**:
   - Output must follow these instructions:
   {format_instructions}
"""


async def main():

    with open("app.py", "w", encoding="utf-8") as f:
        f.write('''import os

API_KEY = "secret_12345"

def calculate(user_input):
    try:
        return eval(user_input)
    except:
        return None
''')

    client = MultiServerMCPClient({
        "code_analysis": {
            "command": "python",
            "args": ["code_analysis_mcp_server.py"],
            "transport": "stdio"
        },
        "patch_manager": {
            "command": "python",
            "args": ["patch_manager_mcp_server.py"],
            "transport": "stdio"
        }
    })

    tools = await client.get_tools()

    parser = JsonOutputParser(pydantic_object=OutputSchema)
    format_instructions = parser.get_format_instructions()

    agent = create_agent(
        model=model,
        tools=tools,
        system_prompt=SYSTEM_PROMPT.format(format_instructions=format_instructions)
    )

    user_input = input("Enter path to file or code to audit (e.g., app.py): ")

    response = await agent.ainvoke({
        "messages": [("user", f"Audit the file: {user_input}")]
    })

    print("\n🏁 Agent Output:")
    print(response["messages"][-1].content)


if __name__ == "__main__":
    asyncio.run(main())