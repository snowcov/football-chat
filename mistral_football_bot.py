import pandas as pd
import ollama
import json

# Load CSV data
df = pd.read_csv('./data/sportsref.csv')

# Normalize column names
df.columns = df.columns.str.strip()

# Define relevant stat columns (customize as needed)
relevant_columns = [
    'Fantasy Rank', 'Player', 'Team', 'Fantasy Position', 'Fantasy Points',
    'Passing Yards', 'Passing TD', 'Rushing Yards', 'Rushing TD',
    'Receiving Yards', 'Receiving TD'
]

# Filter only relevant columns that exist in the dataset
relevant_columns = [col for col in relevant_columns if col in df.columns]
df = df[relevant_columns]

# Function to extract relevant data based on keywords
def get_relevant_data(query: str):
    query = query.lower()

    # Determine position
    if "wide receiver" in query or "wr" in query:
        pos = "WR"
    elif "running back" in query or "rb" in query:
        pos = "RB"
    elif "quarterback" in query or "qb" in query:
        pos = "QB"
    else:
        pos = None

    # Filter by position
    data = df.copy()
    if pos:
        data = data[data['Fantasy Position'] == pos]

    # Determine sorting
    if "most passing yards" in query and 'Passing Yards' in data.columns:
        data = data.sort_values(by='Passing Yards', ascending=False).head(5)
    elif "top" in query or "best" in query:
        data = data.sort_values(by='Fantasy Points', ascending=False).head(5)
    else:
        data = data.head(10)

    return data.dropna().to_dict(orient='records')

# Ask Mistral a question with selected stats
def ask_model(question: str):
    data = get_relevant_data(question)
    if not data:
        return "No relevant stats found to answer this question."

    system_prompt = (
        "You are a fantasy football analyst. You MUST answer ONLY using the statistics provided in the JSON below. "
        "Do not guess or use outside knowledge. If the answer cannot be found in the stats, say 'I don't have that information.'"
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
