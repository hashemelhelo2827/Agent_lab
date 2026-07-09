import json  
from openai import OpenAI

x = input("Choose three random things in your fridge to cook: ")

client = OpenAI(
    api_key="AQ.Ab8RN6L1O3TBhD43Nfji76qdJIa0KunIE69j_HMbJQuLlVZjbg",  
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

prompt_instruction = (
    f"Use these 3 ingredients to make a creative recipe: {x}. "
    "You must return your response strictly as a JSON object with the following keys:\n"
    "- 'recipe_name': The name of the meal\n"
    "- 'prep_time': Estimated cooking time\n"
    "- 'instructions': A list of step-by-step strings"
)

try:
    response = client.chat.completions.create(
        model="gemini-2.5-flash",
        response_format={"type": "json_object"},
        messages=[
            {"role": "user", "content": prompt_instruction}
        ]
    )
    
  
    raw_json_string = response.choices[0].message.content
    
    
    recipe_dict = json.loads(raw_json_string)
    
    print("\n==============================")
    print(f"🍳 RECIPE: {recipe_dict['recipe_name'].upper()}")
    print(f"⏱️ PREP TIME: {recipe_dict['prep_time']}")
    print("==============================\n")
    
    print("STEPS:")
    for i, step in enumerate(recipe_dict['instructions'], 1):
        print(f"{i}. {step}")

except Exception as e:
    print(f"There is an error: {e}")