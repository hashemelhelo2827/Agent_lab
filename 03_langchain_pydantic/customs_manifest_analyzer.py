from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import JsonOutputParser
from typing import Optional
import os
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel,Field

load_dotenv(r"..\openai-venv\.env")
llm=ChatOpenAI(
    model='gemini-2.5-flash',
    api_key=os.getenv("GEMINI_API_KEY"),
    base_url='https://generativelanguage.googleapis.com/v1beta/openai/'
)

class SupportTicketsManifestItem(BaseModel):
    name: str=Field(description='(e.g., "Laptop Computers")') 
    quantity: int=Field(description='')
    total_value_usd: float=Field(description='')
    requires_hazard_isolation: bool=Field(description="(True if it's a chemical, weapon, or flammable substance, otherwise False)")

class SupportTicketsCustomsManifest(BaseModel):
    container_id: str=Field(default='Unknown',description='(look for identifiers like "alpha-9").')
    items: list[SupportTicketsManifestItem]=Field(description='output an array of objects')
    target_risk_level: str=Field(description='Must strictly be one of: "low", "medium", or "high". (If hazardous items are present, this must be high) ')  

parser=JsonOutputParser(pydantic_object=SupportTicketsCustomsManifest)
customer_input=input('Describe your product: ')
prompt=ChatPromptTemplate.from_messages(
    [
        (
         'system',
         "You are a support ticket customs clearance agent for an international shipping port.\n{format_instructions}"    
        ),
        (
            'user',
            "Real container: {customer_input}"
        )
    ]
)

chain=prompt|llm|parser

try:
    final_response = chain.invoke({
        'customer_input': customer_input,
        'format_instructions': parser.get_format_instructions()
    })
    total_container_value=0

    print("\n--- LangChain Analysis Result ---")
    print(final_response)
    print("---------------------------------\n")
    print("--- Cargo Evaluation ---")
    for item in final_response['items']:
        total_container_value += item['total_value_usd']
        if item['requires_hazard_isolation']:
            print(f"HAZMAT WARNING: {item['name']} requires isolated containment storage")
    print(f'Total value of the container is {total_container_value}')
    if final_response['target_risk_level']=='high' or total_container_value >100000:
        print('❌ HOLD: Container requires manual inspection.')
    else:
        print('✅ CLEAR: Container cleared for port entry.')
except Exception as e:
    print(f'An error has occurred: {e}')