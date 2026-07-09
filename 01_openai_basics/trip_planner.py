import json
from openai import OpenAI

location=input("enter your location: ")
num_of_days=input("enter the number of days: ")

prompt_instructions=(f"use location {location} and number of days"
 f"{num_of_days} to creat a planner trip in jason format with this keys :"
 "destination: String (The city name)"
 "days: An array (list) of objects. Each object represents a day and should have:"
 "day_number: Integer (e.g., 1, 2)"
 "theme: String (e.g., Historic Landmarks or Food & Shopping)"
'activities: A list of strings (at least 3 things to do that day)')

client=OpenAI(
    api_key="AQ.Ab8RN6L1O3TBhD43Nfji76qdJIa0KunIE69j_HMbJQuLlVZjbg",  
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