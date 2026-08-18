import json
import os
from openai import OpenAI
from dotenv import load_dotenv

customer_input = input("Enter customer complaint: ")

load_dotenv(r"..\openai-venv\.env")
client = OpenAI(
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    api_key=os.getenv("GEMINI_API_KEY")
)

prompt_1 = (
    "evaluate the customer's text and return a strict JSON object with these exact keys:\n"
    "- sentiment: Must be exactly one of these strings: positive, neutral, or negative\n"
    "- department: Choose exactly one: billing, tech_support, shipping, or general\n"
    "- urgency: Must be exactly one of these strings: low, medium or high\n"
    "- order_id: Look for an order number. Extract it as a string. If none, return null."
)

try:
    response = client.chat.completions.create(
        model='gemini-2.5-flash',
        response_format={'type': 'json_object'},
        messages=[

            {'role': 'system', 'content': prompt_1},
            
            {'role': 'user', 'content': "I want a refund for order #111."},
            
            {'role': 'assistant', 'content': '{"sentiment": "neutral", "department": "billing", "urgency": "medium", "order_id": "111"}'},
            
            {'role': 'user', 'content': f"Real customer complaint: {customer_input}"}
        ]
    )


    raw_response = response.choices[0].message.content
    final_response = json.loads(raw_response)
    
    print("\n--- AI Analysis Result ---")
    print(final_response)
    print("--------------------------\n")
    
    
    if final_response['urgency'] == 'high' or final_response['sentiment'] == 'negative':
        print("🚨 CRITICAL ATTENTION REQUIRED FOR THIS TICKET! 🚨")
        

    print(f"Routing ticket to the {final_response['department'].replace('_', ' ').title()} Department Queue...")
    

    if final_response['order_id']:
        print(f"Searching database for Order ID: [{final_response['order_id']}]...")
    else:
        print("No order ID detected in message.")

except Exception as e:
    print(f"There is an error: {e}")