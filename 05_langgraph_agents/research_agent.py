# pyrefly: ignore [missing-import]
from langgraph.graph import StateGraph,START,END
import os
from  langchain_core.output_parsers import JsonOutputParser
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel,Field
from typing import Annotated,TypedDict,Literal
import operator
from dotenv import load_dotenv
from langgraph.types import Send

load_dotenv(dotenv_path=r"C:\Users\hashe\Desktop\Agent_lab\openai-venv\.env")
API_KEY=os.getenv("GROQ_API_KEY")
llm=ChatOpenAI(
    model='llama-3.3-70b-versatile',
    api_key=API_KEY,
    base_url='https://api.groq.com/openai/v1'
)


class ResearchState(TypedDict):
    topic: str
    questions: list[str]
    research_notes: Annotated[list[str], operator.add] 
    final_report: str
    feedback: str
    verification_status:Literal["verified","contradictory",'either']


class questionparser(BaseModel):
    questions: list[str] =Field(description="3 question about topic in the list")
    
class research_parser(BaseModel):
    notes: list[str] =Field(description="some notes to use while writing the paragraphe there is another 2 response on diffrent questions related to this topic")

class report_parser(BaseModel):
     final_report: str =Field(description="use rules of panctution and be sure about your info")

class rev_parser(BaseModel):
    feedback: str =Field(description="if there is some facts wrong illustrate why and say what it is")
    verification_status: Literal["verified","contradictory"] =Field(description="use rules of panctution and be sure about your info")


def route_to_parallel_workers(state: ResearchState):
    return [
        Send("research_question", {"question": q}) 
        for q in state["questions"]
    ]


def router(state: ResearchState):
    if state['verification_status']=="contradictory":
        return 'compile_report'
    else :
        return END

def generate_question(state:ResearchState):
    
    print("generation question has started")

    prompt=ChatPromptTemplate([
        (
            'system',
            'Act as an Enginner and philosopher to think'
            'about complex topic and give 3 question to make an essay about it has  '
            '3 paragraphs every paragraphe talks about question you asked and put the questions in this format {format_instructions} '
        ),
        (
            'user',
            '{topic}'
        )
    ])
    gen_parser=JsonOutputParser(pydantic_object=questionparser)
    chain=prompt|llm|gen_parser
    response=chain.invoke({
        "topic":state['topic'],
        'format_instructions':gen_parser.get_format_instructions()
    })

    return{
        'questions':response['questions'],
    }



def research_question(state:ResearchState):
    print("start geathering info")
    prompt=ChatPromptTemplate([
        (
            'system',
            'Act as a tecnical writer to answer this  question in some notes and make sure'
            'everyone reads this paragraph understand it even if he did not have any knowladge about it in this format {format_instructions} '
        ),
        (
            'user',
            '{question}'
        )
    ])
    res_parser=JsonOutputParser(pydantic_object=research_parser)
    chain=prompt|llm|res_parser
    response=chain.invoke({
        "question":state['question'],
        'format_instructions':res_parser.get_format_instructions()
    })
    return{
        'research_notes':response['notes'],
    }






def compile_report(state:ResearchState):
    print("start writing")
    prompt=ChatPromptTemplate([
        (
            'system',
            'Act as a tecnical writer to use this notes to write a report'
            'everyone reads this report understand it even if he did not have any knowladge about it in this format {format_instructions} '
            "If you received negative feedback from previous attempts, fix the bug mentioned: {feedback}"
        ),
        (
            'user',
            '{research_notes}'
        )
    ])
    fi_parser=JsonOutputParser(pydantic_object=report_parser)
    chain=prompt|llm|fi_parser
    response=chain.invoke({
        'research_notes':state['research_notes'],
        'feedback':state['feedback'],
        'format_instructions':fi_parser.get_format_instructions()
    })
    
    return{
        'final_report':response['final_report']
    }




def fact_checker(state:ResearchState):
    print("start writing")
    prompt=ChatPromptTemplate([
        (
            'system',
            'Act as a tecnical writer and an expert in the field of this topic to use this notes to review  '
            'the facts that this paraphe provides it in this format {format_instructions} '
        ),
        (
            'user',
            '{research_notes}'
        )
    ])
    fi_parser=JsonOutputParser(pydantic_object=rev_parser)
    chain=prompt|llm|fi_parser
    response=chain.invoke({
        'research_notes':state['research_notes'],
        'format_instructions':fi_parser.get_format_instructions()
    })
    
    return{
        'feedback':response['feedback'],
        'verification_status':response['verification_status']
    }

workflow=StateGraph(ResearchState)

workflow.add_node("research_question",research_question)
workflow.add_node("generate_question",generate_question)
workflow.add_node("compile_report",compile_report)
workflow.add_node("fact_checker",fact_checker)

workflow.add_edge(START,'generate_question')

workflow.add_conditional_edges(
    'generate_question',
    route_to_parallel_workers,
    ["research_question"]
)
workflow.add_edge("research_question", "compile_report")
workflow.add_edge("compile_report", "fact_checker")
workflow.add_conditional_edges(
    "fact_checker",
    router,
    {
        END:END,
        'compile_report':'compile_report'
    }
)

initial_state={
    'topic': input('Enter the name of topic: '),
    'questions': [],
    'research_notes': [] ,
    'final_report': '',
    'feedback': '',
    'verification_status': 'either'

}

app=workflow.compile()

try:
    response=app.invoke(initial_state)
    print(response['final_report'])
except Exception as e:
    print(f"An error occurred: {e}")