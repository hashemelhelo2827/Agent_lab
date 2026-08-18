import os
from typing import Optional
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field


load_dotenv(r"..\openai-venv\.env")
llm = ChatOpenAI(
    model="gemini-2.5-flash",
    api_key=os.getenv("GEMINI_API_KEY") , 
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

class ticketsupport(BaseModel):
    sentiment: str = Field(description="Must be exactly: 'positive', 'neutral', or 'negative'")
    department: str = Field(description="Choose exactly: 'billing', 'tech_support', 'shipping', or 'general'")
    urgency: str = Field(description="Must be exactly: 'low', 'medium', or 'high'")
    order_id: Optional[str] = Field(default=None, description="The extracted order number string, or null if missing")

parser = JsonOutputParser(pydantic_object=ticketsupport)

prompt_template = ChatPromptTemplate.from_messages([
    (
        "system", 
        "You are a support ticket router. Analyze the user's text.\n{format_instructions}"
    ),
    (
        "user", 
        "Real customer complaint: {customer_input}"
    )
])

chain = prompt_template | llm | parser

customer_input = input("Enter customer complaint: ")

try:
    final_response = chain.invoke({  
        'customer_input': customer_input,
        'format_instructions': parser.get_format_instructions()
    })
    
    print("\n--- LangChain Analysis Result ---")
    print(final_response)
    print("---------------------------------\n")


    if final_response['urgency'] == 'high' or final_response['sentiment'] == 'negative':
        print("🚨 CRITICAL ATTENTION REQUIRED FOR THIS TICKET! 🚨")
        
    print(f"Routing ticket to the {final_response['department'].replace('_', ' ').title()} Department...")

except Exception as e:
    print(f'error: {e}')  