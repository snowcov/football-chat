import pandas as pd
import json

# Load and clean the CSV
csv_path = "./data/sportsref.csv"  # Update this if your file is in a different path
df = pd.read_csv(csv_path)
df.fillna(0, inplace=True)
df.columns = [col.strip() for col in df.columns]  # Remove whitespace in column names

# Supported fantasy positions
positions = ['QB', 'RB', 'WR', 'TE']

# Output list
prompt_data = []

# Top 5 players by position
def get_top_players_by_position(df, position, n=5):
    filtered = df[df['Fantasy Position'] == position]
    sorted_df = filtered.sort_values(by='Fantasy Points', ascending=False).head(n)
    return sorted_df[['Player', 'Fantasy Points']]

for pos in positions:
    top_players = get_top_players_by_position(df, pos)
    prompt = f"List the top 5 {pos}s based on fantasy points."
    response_lines = [
        f"{i+1}. {row['Player']} – {round(row['Fantasy Points'], 1)} pts"
        for i, row in top_players.iterrows()
    ]
    response = "\n".join(response_lines)
    prompt_data.append({"prompt": prompt, "response": response})

# Filtered queries: Under 25 and >200 fantasy points
for pos in positions:
    filtered = df[(df['Fantasy Position'] == pos) & (df['Age'] < 25) & (df['Fantasy Points'] > 200)]
    if not filtered.empty:
        sorted_df = filtered.sort_values(by='Fantasy Points', ascending=False)
        prompt = f"Who are the best {pos}s under age 25 with more than 200 fantasy points?"
        response_lines = [
            f"{i+1}. {row['Player']} – {round(row['Fantasy Points'], 1)} pts"
            for i, row in sorted_df.iterrows()
        ]
        response = "\n".join(response_lines)
        prompt_data.append({"prompt": prompt, "response": response})

# Team-specific QB query example (you can expand to other positions later)
teams = df['Team'].unique()
for team in teams[:5]:  # Limit to 5 teams for brevity
    qbs = df[(df['Team'] == team) & (df['Fantasy Position'] == 'QB')]
    if not qbs.empty:
        sorted_qbs = qbs.sort_values(by='Fantasy Points', ascending=False)
        prompt = f"Top fantasy quarterbacks (QB) on the {team}?"
        response_lines = [
            f"{i+1}. {row['Player']} – {round(row['Fantasy Points'], 1)} pts"
            for i, row in sorted_qbs.iterrows()
        ]
        response = "\n".join(response_lines)
        prompt_data.append({"prompt": prompt, "response": response})

# Save all prompts to a JSONL file
output_path = "./data/fantasy_rank_prompts.jsonl"
with open(output_path, "w") as f:
    for entry in prompt_data:
        f.write(json.dumps(entry) + "\n")

print(f"✅ Prompt dataset saved as: {output_path}")
