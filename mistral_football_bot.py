import pandas as pd
import ollama
import json
import os
import re
from data_loader import load_and_process_data

# Load and process all relevant CSV data
combined_df = load_and_process_data('./data')
print(combined_df['Dataset'].unique())
print(combined_df['Dataset'].unique())
print(combined_df['Fantasy Position'].unique())
# Normalize column names
combined_df.columns = combined_df.columns.str.strip()

# Function to extract relevant data based on keywords
def get_relevant_data(query: str):
    query = query.lower()
    data = combined_df.copy()

    # Filter by dataset type
    if "defense" in query or "points allowed" in query:
        data = data[data['Dataset'] == 'Defenses']
    elif "passer" in query or "quarterback" in query or "qb" in query:
        data = data[data['Dataset'] == 'Passers']
    elif "fantasy" in query or "player" in query:
        data = data[data['Fantasy Position'].notna()]

    # Filter by fantasy position if specified
    if "rb" in query or "running back" in query:
        data = data[data['Fantasy Position'] == 'RB']
    elif "wr" in query or "wide receiver" in query:
        data = data[data['Fantasy Position'] == 'WR']
    elif "qb" in query or "quarterback" in query or "passer" in query:
        data = data[data['Fantasy Position'] == 'QB']
    elif "te" in query or "tight end" in query:
        data = data[data['Fantasy Position'] == 'TE']
    elif "fb" in query or "fullback" in query:
        data = data[data['Fantasy Position'] == 'FB']

    # Filter by year if specified
    years = re.findall(r'\b(20\d{2})\b', query)
    if 'Year' in data.columns and years:
        data = data[data['Year'].isin(map(int, years))]

    # Sort by relevant stats
    if "most passing yards" in query and 'Passing Yards' in data.columns:
        data = data.sort_values(by='Passing Yards', ascending=False)
    elif "fewest points allowed" in query and 'Points Allowed' in data.columns:
        data = data.sort_values(by='Points Allowed', ascending=True)
    elif 'Fantasy Points' in data.columns:
        data = data.sort_values(by='Fantasy Points', ascending=False)
    elif 'Passing Yards' in data.columns:  # Default sorting for passers
        data = data.sort_values(by='Passing Yards', ascending=False)

    # Return top results
    return data.head(10).to_dict(orient='records')

# Ask Mistral a question with selected stats
def ask_model(question: str):
    data = get_relevant_data(question)
    if not data:
        return "No relevant stats found to answer this question."

    system_prompt = (
        "You are a fantasy football analyst. "
        "You MUST answer ONLY using the statistics provided in the JSON below. "
        "Do not guess or use outside knowledge. "
        "If the answer cannot be found in the stats, say 'I don't have that information.'\n\n"
        "If the question is about the 2024 season or 'next season', use the 2022 and 2023 stats to make a reasoned prediction for 2024. "
        "Base your prediction on trends, improvements, or consistent high performance in the stats. "
        "If asked for good pickups, suggest players who have shown recent improvement, high fantasy points, or upward trends in the most recent years. "
        "If the question asks for the 'top', 'best', or 'most valuable' player(s) at a position, select the player(s) with the highest Fantasy Points."
    )

    user_prompt = f"""
Question: {question}
Stats (ONLY use these to answer): {json.dumps(data, indent=2)}
Answer:"""

    try:
        response = ollama.chat(model='mistral', messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ])
        return response['message']['content']
    except Exception as e:
        return f"Error: {e}"

# Continuous loop
if __name__ == "__main__":
    print("Welcome to the Fantasy Football LLM Bot! Type 'exit' to quit.")
    while True:
        user_input = input("\nAsk a question: ")
        if user_input.lower() in ['exit', 'quit']:
            break
        answer = ask_model(user_input)
        print(f"\nAnswer: {answer}")
