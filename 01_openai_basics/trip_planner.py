import json
import os
from openai import OpenAI
from dotenv import load_dotenv

location=input("enter your location: ")
num_of_days=input("enter the number of days: ")

prompt_instructions=(
    f"Create a trip planner for location {location} for {num_of_days} days. "
    "Return a strict JSON object with the following keys:\n"
    "- 'destination': String (the city name)\n"
    "- 'days': An array (list) of objects. Each object represents a day and must have:\n"
    "    - 'day_number': Integer (e.g., 1, 2)\n"
    "    - 'theme': String (e.g., Historic Landmarks or Food & Shopping)\n"
    "    - 'activities': A list of strings (at least 3 things to do that day)"
)

load_dotenv(r"..\openai-venv\.env")
client=OpenAI(
    api_key=os.getenv("GEMINI_API_KEY"),
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

try:
    response=client.chat.completions.create(
        model="gemini-2.5-flash",
        response_format={'type':'json_object'},
        messages =[{"role":'user','content':prompt_instructions}]
    )
    
    raw_jason=response.choices[0].message.content

    planner_dict=json.loads(raw_jason)

    print(f"your plan in {location} for {num_of_days} days")

    for i,day in enumerate(planner_dict['days'],1):
        print("\n")
        print(f"Day {day['day_number']}: {day['theme']}")
        print("\n")
        for j,activity in enumerate(day['activities'],1):
            print(f"{j} . {activity}")

except Exception as e:
    print(f" There is an error: {e}")