from langchain_core.output_parsers import JsonOutputParser
from langchain_openai import ChatOpenAI
from typing import Optional
import os
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel,Field

load_dotenv(r"..\openai-venv\.env")
llm=ChatOpenAI(
    model="gemini-2.5-flash",
    api_key=os.getenv("GEMINI_API_KEY"),
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

class SupportTicket(BaseModel):
    category:str=Field(description='Must strictly be one of: "Electronics", "Clothing", "Home & Kitchen", or "Books"') 
    condition:str=Field(description=' Must strictly be one of: "New", "Like New", "Used", or "Damaged"')
    estimated_value_usd:Optional[float]=Field(default=None,description= 'Look for price hints in the text. Extract it as a float if mentioned')
    contains_restricted_items:  bool=Field(description='It should be True if the item description contains weapons, alcohol, or prescription drugs.')  

chatprompts=ChatPromptTemplate.from_messages([
    (
        'system',
        "You are a support ticket Product Tagging System. Analyze the user's text.\n{format_instructions}"
    ),
    
    (
        'user',
        "Real customer product: {customer_input}"
    )
])
customer_input=input('Describe you product: ')
praser=JsonOutputParser(pydantic_object=SupportTicket)
chain=chatprompts|llm|praser
try:
    final_response=chain.invoke(
       { 'customer_input':customer_input,
        'format_instructions':praser.get_format_instructions()}
    )

    if final_response['contains_restricted_items']:
        print('POLICY VIOLATION: Listing blocked for restricted items.')
    else:
        print(f"✅ Clean Inventory Confirmation:")
        print(f"   Category: {final_response['category']}")
        print(f"   Condition: {final_response['condition']}")
        print(f"   Price: ${final_response['estimated_value_usd'] if final_response['estimated_value_usd'] else 'Not Detected'}")
except Exception as e:
    print(f'error occurred: {e}')