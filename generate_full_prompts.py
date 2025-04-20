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

# Function to create a detailed prompt/response for each player
def create_player_prompt(player_row):
    stats = {
        "Fantasy Rank": player_row['Fantasy Rank'],
        "Player": player_row['Player'],
        "Team": player_row['Team'],
        "Fantasy Position": player_row['Fantasy Position'],
        "Age": player_row['Age'],
        "Games Played": player_row['Games Played'],
        "Games Started": player_row['Games Started'],
        "Passes Completed": player_row['Passes Completed'],
        "Passes Attempted": player_row['Passes Attempted'],
        "Passing Yards": player_row['Passing Yards'],
        "Passing Touchdowns": player_row['Passing Touchdowns'],
        "Interceptions": player_row['Interceptions'],
        "Rushing Attempts": player_row['Rushing Attempts'],
        "Rushing Yards": player_row['Rushing Yards'],
        "Rushing Yards Per Attempt": player_row['Rushing Yards Per Attempt'],
        "Pass Targets": player_row['Pass Targets'],
        "Receptions": player_row['Receptions'],
        "Receiving Yards": player_row['Receiving Yards'],
        "Receiving Yards Per Reception": player_row['Receiving Yards Per Reception'],
        "Fumbles": player_row['Fumbles'],
        "Fumbles Lost": player_row['Fumbles Lost'],
        "Touchdowns": player_row['Touchdowns'],
        "Fantasy Points": player_row['Fantasy Points'],
        "Position Rank": player_row['Position Rank']
    }
    
    # Construct a prompt and completion for the player
    prompt = f"Details of player {player_row['Player']}:"
    response = "\n".join([f"{key}: {value}" for key, value in stats.items()])
    
    return {"prompt": prompt, "response": response}

# Iterate over each player and generate prompt/response data
for index, row in df.iterrows():
    prompt_data.append(create_player_prompt(row))

# Save all prompts to a JSONL file
output_path = "full_fantasy_stats_prompts.jsonl"
with open(output_path, "w") as f:
    for entry in prompt_data:
        f.write(json.dumps(entry) + "\n")

print(f"✅ Prompt dataset saved as: {output_path}")
