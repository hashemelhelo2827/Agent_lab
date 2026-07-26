import os
import json
from pydantic import BaseModel,Field
from langgraph.graph import START,END,StateGraph
from langchain_core.output_parsers import JsonOutputParser
from langchain_openai import ChatOpenAI
from typing import TypedDict,Literal,Optional
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

load_dotenv(dotenv_path=r"C:\Users\hashe\Desktop\Agent_lab\openai-venv\.env")
API_KEY=os.getenv("GEMINI_API_KEY")

llm=ChatOpenAI(
    model='gemini-2.5-flash',
    api_key=API_KEY,
    base_url='https://generativelanguage.googleapis.com/v1beta/openai/'
)

class Shared_State(TypedDict):
    email_content: str
    category:  str
    sentiment: str
    summary: str
    processed_output: dict

class Triage_Pydantic(BaseModel):
    category:  Literal["billing", "technical", "general", "urgent"]
    customer_mood: Literal ["angry", "neutral"]
    summary:str=Field(description=" A quick 1-sentence summary of the user's issue.")

class Escalation_Pydantic(BaseModel):
    priority: str = Field(description="Must be 'PRIORITY ESCALATION' if escalated")
    category: str
    customer_mood: str
    summary: str
    internal_notes: list[str] = Field(description="List of bullet-point reasons why this was escalated")

class Standard_Pydantic(BaseModel):
    status: str = Field(description="Must be 'resolved' or 'pending'")
    category: str
    customer_mood: str
    summary: str
    notes: str = Field(description="Brief resolution or follow-up notes")

parser=JsonOutputParser(pydantic_object=Triage_Pydantic)

def triage_email(state:Shared_State):
    prompt=ChatPromptTemplate.from_messages([
        ('system',
        'Act as triaging agent who works in customer service department that receives chaotic, mixed emails'
        'you need to categorize the email, know the mood of customer  '
        'give an output about him {outputstructure}' 
        ),
        ('user',
         '{email_content}' 
        )
    ])
    chain=prompt|llm| parser
    response=chain.invoke({
        "outputstructure":parser.get_format_instructions(),
        'email_content':state['email_content']
    })

    return{
        'category':response['category'],
        'sentiment':response['customer_mood'],
        'summary':response['summary']
    }

escalation_parser = JsonOutputParser(pydantic_object=Escalation_Pydantic)
standard_parser = JsonOutputParser(pydantic_object=Standard_Pydantic)

def escalate_ticket(state:Shared_State):
    prompt=ChatPromptTemplate.from_messages([
        ('system',
         "Act as escalation agent. Return a strict JSON object.\n{format_instructions}"
         "\n\nCategory: {category}\nCustomer mood: {sentiment}\nSummary: {summary}"
        ),
        ('user',
         '{email_content}'
        )
    ])
    chain=prompt|llm|escalation_parser

    response=chain.invoke({
        'email_content':state['email_content'],
        'category':state['category'],
        'sentiment':state['sentiment'],
        'summary':state['summary'],
        'format_instructions': escalation_parser.get_format_instructions()
    })
    return {"processed_output": response}


def standard_process(state:Shared_State):
    prompt=ChatPromptTemplate.from_messages([
        ('system',
         "Act as a customer service agent processing a standard ticket. "
         "Return a strict JSON object.\n{format_instructions}"
         "\n\nCategory: {category}\nCustomer mood: {sentiment}\nSummary: {summary}"
        ),
        ('user',
         '{email_content}'
        )
    ])
    chain=prompt|llm|standard_parser

    response=chain.invoke({
        'email_content':state['email_content'],
        'category':state['category'],
        'sentiment':state['sentiment'],
        'summary':state['summary'],
        'format_instructions': standard_parser.get_format_instructions()
    })
    return {"processed_output": response}

def router(state:Shared_State):
    if state['category'] in {'urgent', "billing"} or state['sentiment']=='angry':
        return 'escalate_ticket'
    else:
        return 'standard_process'
    
workflow=StateGraph(Shared_State)
workflow.add_node('triage_email',triage_email)
workflow.add_node('escalate_ticket',escalate_ticket)
workflow.add_node('standard_process',standard_process)

workflow.add_edge(START,'triage_email')
workflow.add_conditional_edges(
    'triage_email',
    router,{
        'escalate_ticket':'escalate_ticket',
        'standard_process'  :'standard_process'
    }
)
workflow.add_edge( 'escalate_ticket',END)
workflow.add_edge( 'standard_process',END)

app=workflow.compile()

initial_state = {
    "email_content": input("Paste the email content: "),
    "category": "",
    "sentiment": "",
    "summary": "",
    "processed_output": {}
}

try:
    final_output = app.invoke(initial_state)
    print(json.dumps(final_output, indent=2, ensure_ascii=False))
except Exception as e:
    print(f"An error occurred: {e}")

