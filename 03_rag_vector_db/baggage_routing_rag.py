import os   
from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel,RunnablePassthrough
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import Field,BaseModel
from dotenv import load_dotenv

load_dotenv()
API_KEY=os.getenv("GEMINI_API_KEY")

llm=ChatOpenAI(
    model='gemini-2.5-flash',
    base_url='https://generativelanguage.googleapis.com/v1beta/openai/',
    api_key=API_KEY
)
embeddings_model=GoogleGenerativeAIEmbeddings(
    model='models/gemini-embedding-001',
    google_api_key=API_KEY
)
class SupportTickets(BaseModel):
    passenger_name:str=Field(description='The name of the passenger.') 
    total_bags:int=Field(description='The total count of bags they are checking in.') 
    requires_manual_screening:bool=Field(description=' A boolean (True or False)') 
    assigned_gate:str=Field(description='The destination gate string (e.g., "Gate B12")') 
    
parser=JsonOutputParser(pydantic_object=SupportTickets)

prompt=ChatPromptTemplate.from_messages(
    [
        (
        "system",
        "You are an automated international airport baggage routing system.\n"
        "Analyze the provided airport security rules and the passenger's luggage description to determine the correct routing and screening requirements.\n\n"
        "AIRPORT SECURITY & ROUTING RULES:\n{context}\n\n"
        "{format_instructions}"
    ),
    (
        "user",
        "{customer_input}"
    )
    ]
)

script_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(script_dir, "baggage_rules.txt")
loader=TextLoader(file_path)
raw_text=loader.load()

text_splitter=RecursiveCharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=30
)
chunk=text_splitter.split_documents(raw_text)

vector_db=Chroma.from_documents(documents=chunk,embedding=embeddings_model)

retriever=vector_db.as_retriever(search_kwargs={'k':1})

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

inputs=RunnableParallel(
    context=retriever|format_docs,
    customer_input=RunnablePassthrough(),
    format_instructions=lambda _:parser.get_format_instructions()
)

chain=inputs|prompt|llm|parser

customer_input=input("luggage tag: ")

try:
    final_response=chain.invoke(customer_input)
    print(final_response)
except Exception as e:
    print(f"An error occurred: {e}")