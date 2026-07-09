import os
from langchain_core.runnables import RunnableParallel , RunnablePassthrough
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_openai import ChatOpenAI
from typing import List,Optional
from pydantic import Field , BaseModel
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader

load_dotenv()
API_KEY=os.getenv('GEMINI_API_KEY')

llm=ChatOpenAI(
    model='gemini-2.5-flash',
    base_url='https://generativelanguage.googleapis.com/v1beta/openai/',
    api_key=API_KEY
)

embeddings_model=GoogleGenerativeAIEmbeddings(
    model='models/gemini-embedding-001',
    google_api_key=API_KEY
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

prompt=ChatPromptTemplate.from_messages(
    [
        (
            'system',
            "You are an expert support customs clearance agent.\n"
            "Analyze the provided context data rules and user manifest to build your assessment structured response.\n\n"
            "CRITICAL CUSTOMS LAWS CONTEXT:\n{context}\n\n"
            "{format_instructions}"
        )
        ,
        (
            'user',
            "{customer_input}"
        )
    ]
)

script_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(script_dir, "rulebook.txt")
loader=TextLoader(file_path)
raw_data=loader.load()

textsplitter=RecursiveCharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=30
)
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

chunks=textsplitter.split_documents(raw_data)

vector_db=Chroma.from_documents(documents=chunks,embedding=embeddings_model)

retriever=vector_db.as_retriever(search_kwargs={"k": 1})

inputs=RunnableParallel(
    context=retriever|format_docs,
    customer_input=RunnablePassthrough(),
    format_instructions=lambda _:parser.get_format_instructions()
)

chain=inputs|prompt|llm|parser

customer_input = input('Describe your container cargo: ')

try:
    final_response = chain.invoke(customer_input)
    print("\n--- LangChain Automated Analysis Result ---")
    print(final_response) 
except Exception as e:
    print(f"An error occurred: {e}") 
