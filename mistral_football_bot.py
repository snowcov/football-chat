import pandas as pd
import ollama
import json
import os
import re
from data_loader import load_and_process_data

# Load and process all relevant CSV data
combined_df = load_and_process_data('./data')

# Normalize column names
combined_df.columns = combined_df.columns.str.strip()

# Define relevant stat columns
relevant_columns = [
    'Fantasy Rank', 'Player', 'Team', 'Fantasy Position', 'Fantasy Points',
    'Passing Yards', 'Passing TD', 'Rushing Yards', 'Rushing TD',
    'Receiving Yards', 'Receiving TD', 'Year'
]

# Keep only relevant columns that exist in the dataset
relevant_columns = [col for col in relevant_columns if col in combined_df.columns]
combined_df = combined_df[relevant_columns]

# Function to extract relevant data based on keywords
def get_relevant_data(query: str):
    query = query.lower()
    data = combined_df.copy()

    # 1. Position filtering (robust)
    position = None
    if "wide receiver" in query or "wr" in query:
        position = "wr"
    elif "running back" in query or "rb" in query:
        position = "rb"
    elif "quarterback" in query or "qb" in query:
        position = "qb"
    if position and 'Fantasy Position' in data.columns:
        data = data[data['Fantasy Position'].str.lower().fillna('') == position]
        # Fallback if empty
        if data.empty:
            data = combined_df.copy()

    # 2. Predict next season (2024)
    if "2024" in query or "next season" in query:
        group_cols = ['Player', 'Team', 'Fantasy Position']
        stat_cols = [col for col in data.columns if col not in group_cols + ['Year', 'Source File']]
        pred = data.groupby(group_cols, dropna=False)[stat_cols].mean(numeric_only=True).reset_index()
        pred['Year'] = 2024
        data = pred

    # 3. Year filtering
    years = re.findall(r'\b(20\d{2})\b', query)
    if 'Year' in data.columns:
        if years:
            data = data[data['Year'].isin(map(int, years))]
        elif not ("2024" in query or "next season" in query):
            most_recent = data['Year'].dropna().max()
            data = data[data['Year'] == most_recent]
        if data.empty:
            # Fallback to most recent year
            most_recent = combined_df['Year'].dropna().max()
            data = combined_df[combined_df['Year'] == most_recent]

    # 4. Player name matching (partial)
    player_matches = []
    if 'Player' in data.columns:
        for player in data['Player'].dropna().unique():
            if player and player.lower() in query:
                player_matches.append(player)
        if player_matches:
            data = data[data['Player'].isin(player_matches)]
            if not data.empty:
                return data.dropna().to_dict(orient='records')

    # 5. Sorting logic for "top", "best", "pick up", or position queries
    if (any(word in query for word in ["top", "best", "pick up", "rb", "running back", "wr", "wide receiver", "qb", "quarterback"])
        and 'Fantasy Points' in data.columns):
        data = data.sort_values(by='Fantasy Points', ascending=False)
        data = data.drop_duplicates(subset=['Player'], keep='first').head(5)
    elif "most passing yards" in query and 'Passing Yards' in data.columns:
        data = data.sort_values(by='Passing Yards', ascending=False)
        data = data.drop_duplicates(subset=['Player'], keep='first').head(5)
    elif 'Fantasy Points' in data.columns:
        data = data.sort_values(by='Fantasy Points', ascending=False)
        data = data.drop_duplicates(subset=['Player'], keep='first').head(10)
    else:
        data = data.head(10)

    # 6. Final fallback: always return something
    if data.empty and 'Fantasy Points' in combined_df.columns:
        most_recent = combined_df['Year'].dropna().max()
        fallback = combined_df[combined_df['Year'] == most_recent]
        data = fallback.sort_values(by='Fantasy Points', ascending=False).head(10)

    if not data.empty:
        return data.dropna().to_dict(orient='records')
    else:
        return []

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
