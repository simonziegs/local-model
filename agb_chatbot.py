from operator import itemgetter
from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import PyPDFLoader
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import DocArrayInMemorySearch


# For OpenAI usage
# import os
# from dotenv import load_dotenv

# load_dotenv()

# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
# MODEL = "gpt-3.5-turbo"

# Model Selection
MODEL = "mistral"
# MODEL = "deepseek-r1:7b"
# MODEL = "llama3.1:8b"
# MODEL = "llama3.2"

# Model and embeddings
model = Ollama(model=MODEL)
embeddings = OllamaEmbeddings(model=MODEL)

# Output Parser
parser = StrOutputParser()

# PDF Loader
loader = PyPDFLoader("Allgemeine Geschäftsbedingungen SUS gmbh.pdf")
pages = loader.load_and_split()

# Prompt Template
template = """" \
"Answer the question based on the context below. If you can't answer the question, reply "I don't know"." \
"" \
"Context: {context}" \
"" \
"Question: {question}" \
"""
prompt = PromptTemplate.from_template(template)

# Vector Store
vectorstore = DocArrayInMemorySearch.from_documents(pages, embedding=embeddings)
retriever = vectorstore.as_retriever()

# Chain
chain = (
    {
        "context": itemgetter("question") | retriever,
        "question": itemgetter("question")
    }
    | prompt
    | model
    | parser
)

# Invoke
print(chain.invoke({"question": "Tell me about the Payment terms in this AGB."}), end="", flush=True)