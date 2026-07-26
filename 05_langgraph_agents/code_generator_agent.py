import os
from dotenv import load_dotenv
from pydantic import BaseModel,Field
from typing import Literal,TypedDict
from langchain_core.output_parsers import JsonOutputParser
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph,START,END

load_dotenv(dotenv_path=r"C:\Users\hashe\Desktop\Agent_lab\openai-venv\.env")
API_KEY=os.getenv("GEMINI_API_KEY")

class code(BaseModel):
    status: Literal["pass", "fail"]
    error_message: str = Field(description="If syntax is broken or requirements missing, explain why. Otherwise leave blank.")

class AgentState(TypedDict):
    user_requirement: str
    generated_code: str
    feedback: str
    iterations: int

llm=ChatOpenAI(
    model='gemini-2.0-flash-lite',
    base_url='https://generativelanguage.googleapis.com/v1beta/openai/',
    api_key=API_KEY
)

parser=JsonOutputParser(pydantic_object=code)

def codegeneration(state:AgentState):
    print(f"\n--- [Node: Generate Code] Iteration {state['iterations'] + 1} ---")

    prompt=ChatPromptTemplate.from_messages(
        [
             ("system", "You are an expert Python developer. Write clean Python code that fulfills the user requirement. "
              "Provide ONLY raw code. Do not wrap in markdown blocks like ```python. "
              "If you received negative feedback from previous attempts, fix the bug mentioned: {feedback}"),
             ("user", "{user_requirement}")
        ]
    )

    chain=prompt|llm

    response=chain.invoke(
        {
         'feedback':state['feedback'],
         'user_requirement':state['user_requirement']
        }
    )

    return {
        "generated_code": response.content.strip(),
        "iterations": state["iterations"] + 1
    }

def codeevaluation(state:AgentState):
    print("--- [Node: Evaluate Code] Inspecting syntax and quality ---")

    if "def" not in state["generated_code"]:
        return {"feedback": "Critique: Code fails safety protocols because no functional execution block was found."}
    
    prompt=ChatPromptTemplate.from_messages(
       [ ("system", "You are a senior code auditor. Review the generated code against the initial user requirements.\n"
                   "Determine if it passes syntax rules and fulfills the requirement.\n\n"
                   "{format_instructions}"),
        ("user", "Requirement: {user_requirement}\nCode to Evaluate:\n{generated_code}")]
    )

    chain=prompt|llm|parser

    response=chain.invoke(
        {'format_instructions':parser.get_format_instructions(),
         'user_requirement':state['user_requirement'],
         'generated_code':state['generated_code']
        }
    )

    print(f"Auditor verdict: {response['status'].upper()} | Reason: {response['error_message']}")
    
    if response["status"] == "fail":
        return {"feedback": response["error_message"]}
    else:
        return {"feedback": "pass"}
    
def router(state: AgentState):
    if state["iterations"] >= 3:
        print("--- [Max Iterations Reached] Hard stop. ---")
        return END
        
    if state["feedback"] == "pass":
        return END
    else:
        return "generate_code" 

workflow=StateGraph(AgentState)

workflow.add_node("generate_code", codegeneration)
workflow.add_node("evaluate_code", codeevaluation)

workflow.add_edge(START, "generate_code")
workflow.add_edge("generate_code", "evaluate_code")

workflow.add_conditional_edges(
    'evaluate_code',
    router,
    {
        END:END,
        "generate_code" :"generate_code" 
    }
)

app=workflow.compile()

user_prompt = input("Enter what Python function you want the agent to build:")

initial_state = {
    "user_requirement": user_prompt,
    "generated_code": "",
    "feedback": "None",
    "iterations": 0
}

final_output = app.invoke(initial_state)

print("\n=== FINAL AGENT OUTPUT ===")
print(final_output["generated_code"])

