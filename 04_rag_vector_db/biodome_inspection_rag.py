import os
from typing import Literal,List,Union, Annotated
from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel,RunnablePassthrough
from pydantic import Field,BaseModel
from dotenv import load_dotenv

load_dotenv(dotenv_path=os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))
API_KEY=os.getenv("GEMINI_API_KEY")

llm=ChatOpenAI(
    model='gemini-2.5-flash',
    base_url='https://generativelanguage.googleapis.com/v1beta/openai/',
    api_key=API_KEY
)

embedding_model=GoogleGenerativeAIEmbeddings(
    model='models/gemini-embedding-001',
    google_api_key=API_KEY
)

def doc_formats(docs):
    return '\n\n'.join(doc.page_content for doc in docs )

class BiomassItem(BaseModel):
    cargo_type: Literal [ "biomass"]
    species_name: str=Field(description='e.g., "Hydroponic Soy", "Martian Fern"')
    is_perishable: bool
    temperature_celsius: float

class MachineryItem(BaseModel):
    cargo_type: Literal [ "machinery"]
    equipment_id: str=Field(description='e.g., "Hydroponic Soy", "Martian Fern"')
    power_source: str
    has_lithium_batteries: bool

class BioDomeInspectionManifest(BaseModel):
    arrival_dock:str= Field(description="The dock where the drone landed.")
    inspection_action: Literal [ "quarantine_hold","refrigerated_storage",  "standard_clearance"]
    cargo_items: List[Annotated[Union[BiomassItem, MachineryItem], Field(discriminator='cargo_type')]]

parser=JsonOutputParser(pydantic_object=BioDomeInspectionManifest)

prompt=ChatPromptTemplate.from_messages(
    [
        (
        "system",
        "You are an automated planetary bio-dome cargo and hazard inspector.\n"
        "Analyze the incoming mixed delivery logs and verify them against the safety protocols to determine the final routing and clearance action.\n\n"
        "CRITICAL: Evaluate every item in the log individually. If even one item triggers a quarantine protocol, you must apply the Protocol Override rule and mark the entire manifest action as quarantine_hold.\n\n"
        "SAFETY PROTOCOLS & LOGIC:\n{context}\n\n"
        "{format_instructions}"
    ),
    (
        "user",
        "{customer_input}"
    )
    ]
)

script_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(script_dir, "biodome_rules.txt")

# Pass the absolute path to your loader or file read block
loader = TextLoader(file_path)
raw_data=loader.load()

text_splitter=RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=50
)

chunk=text_splitter.split_documents(raw_data)

vector_db=Chroma.from_documents(documents=chunk,embedding=embedding_model)

retriever=vector_db.as_retriever(search_kwargs={'k':4})

inputs=RunnableParallel(
    context=retriever|doc_formats,
    customer_input=RunnablePassthrough(),
    format_instructions= lambda _:parser.get_format_instructions()
)
chain=inputs|prompt|llm|parser

customer_input=input("Enter cargo description: ")

try:
    response=chain.invoke(customer_input)
    print(response)
except Exception as e:
    print(f"An error occurred: {e}")