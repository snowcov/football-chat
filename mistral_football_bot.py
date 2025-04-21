import pandas as pd
import ollama
import json

# Load CSV data
df = pd.read_csv('./data/sportsref.csv')

# Normalize column names
df.columns = df.columns.str.strip()

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

    # Determine sorting
    if "most passing yards" in query:
        data = df[df['Fantasy Position'] == 'QB']
        return data.sort_values(by='Passing Yards', ascending=False).head(5).to_dict(orient='records')

    if pos:
        data = df[df['Fantasy Position'] == pos]
        if "top" in query or "best" in query:
            return data.sort_values(by='Fantasy Points', ascending=False).head(5).to_dict(orient='records')
        return data.head(10).to_dict(orient='records')

    return df.sort_values(by='Fantasy Points', ascending=False).head(10).to_dict(orient='records')

# Ask Mistral a question with selected stats
def ask_model(question: str):
    data = get_relevant_data(question)
    system_prompt = (
        "You are a fantasy football expert. You are provided with player stats in JSON format. "
        "Use these to answer user questions about player performance, rankings, and comparisons."
    )

    user_prompt = f"""
Question: {question}
Stats: {json.dumps(data, indent=2)}
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
        print(f"\n🧠 {answer}")
