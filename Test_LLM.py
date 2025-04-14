from langchain_community.document_loaders.csv_loader import CSVLoader
from pathlib import Path
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
import os
from dotenv import load_dotenv
import pandas as pd
import faiss
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.documents import Document

# Load environment variables from a .env file
load_dotenv()

# Set the OpenAI API key environment variable
os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')

llm = ChatOpenAI(model="gpt-3.5-turbo-0125")

file_path = ('./data/sportsref.csv')  # Insert the path of the CSV file
data = pd.read_csv(file_path)

# Check if required columns exist
required_columns = ['Player', 'Team', 'Fantasy Points', 'Fantasy Position']
for col in required_columns:
    if col not in data.columns:
        raise KeyError(f"Missing required column in CSV: {col}")

# Convert the CSV data into a list of Document objects
docs = []
for _, row in data.iterrows():
    text = (
        f"Player: {row['Player']}\n"
        f"Team: {row['Team']}\n"
        f"Fantasy Points: {row['Fantasy Points']}\n"
        f"Position: {row['Fantasy Position']}"
    )
    docs.append(Document(page_content=text))

embeddings = OpenAIEmbeddings()
vector_store = FAISS.from_documents(docs, embeddings)

retriever = vector_store.as_retriever()

# Set up system prompt
system_prompt = (
    "You are a sports statistics assistant. Use the following data context to answer fantasy sports-related questions. "
    "Answer concisely in 2-3 sentences, and only refer to data from the table if available."
    "\n\n"
    "{context}"
)

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}"),
])

# Create the question-answer chain
question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

answer = rag_chain.invoke({"input": "Who scored the most fantasy points this season?"})
print(answer['answer'])