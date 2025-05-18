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
required_columns = [
    'Fantasy Rank', 'Player', 'Team', 'Fantasy Position', 'Age', 'Games Played', 
    'Games Started', 'Passes Completed', 'Passes Attempted', 'Passing Yards', 
    'Passing Touchdowns', 'Interceptions', 'Rushing Attempts', 'Rushing Yards', 
    'Rushing Yards Per Attempt', 'Pass Targets', 'Receptions', 'Receiving Yards', 
    'Receiving Yards Per Reception', 'Fumbles', 'Fumbles Lost', 'Touchdowns', 
    'Fantasy Points', 'Position Rank'
]
for col in required_columns:
    if col not in data.columns:
        raise KeyError(f"Missing required column in CSV: {col}")

docs = []

for _, row in data.iterrows():
    name = row['Player']
    position = row['Fantasy Position']
    team = row['Team']
    age = row['Age']
    games_played = row['Games Played']
    games_started = row['Games Started']
    fantasy_rank = row['Fantasy Rank']
    position_rank = row['Position Rank']
    fantasy_points = row['Fantasy Points']
    touchdowns = row['Touchdowns']

    card = (
        f"{name}, a {position} for the {team}, was ranked #{fantasy_rank} overall in fantasy last season "
        f"and #{position_rank} among all {position}s. At age {age}, they played {games_played} games and started {games_started} of them, "
        f"scoring {fantasy_points} fantasy points with {touchdowns} total touchdowns.\n\n"
    )

    if row['Passes Completed'] > 0 or row['Passing Yards'] > 0:
        card += (
            f"Passing: {row['Passes Completed']} completions on {row['Passes Attempted']} attempts, "
            f"{row['Passing Yards']} passing yards, {row['Passing Touchdowns']} touchdowns, and "
            f"{row['Interceptions']} interceptions.\n"
        )

    if row['Rushing Attempts'] > 0:
        card += (
            f"Rushing: {row['Rushing Attempts']} attempts for {row['Rushing Yards']} yards "
            f"({row['Rushing Yards Per Attempt']} YPA).\n"
        )

    if row['Receptions'] > 0:
        card += (
            f"Receiving: {row['Receptions']} receptions on {row['Pass Targets']} targets for "
            f"{row['Receiving Yards']} yards "
            f"({row['Receiving Yards Per Reception']} YPR).\n"
        )

    if row['Fumbles'] > 0:
        card += f"Fumbles: {row['Fumbles']} total, {row['Fumbles Lost']} lost.\n"

    docs.append(Document(page_content=card.strip()))


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

answer = rag_chain.invoke({"input": "Name the top 3 Quarterbacks this season and their passing yards?"})
print(answer['answer'])