import pandas as pd

# Load the source files
passers_file = "22-23passersw1-18.csv"
defenses_file = "defenses 22-23.csv"
sportsref_file = "sportsref2022.csv"

# Load the data
passers_df = pd.read_csv(passers_file)
defenses_df = pd.read_csv(defenses_file)
sportsref_df = pd.read_csv(sportsref_file)

# Define the column mappings for quarterbacks (passers)
passers_columns_mapping = {
    "Player": "Player",
    "Team": "Team",
    "Week": "Week",  # Include the Week column
    "Cmp%": "Pass Completion Percentage",
    "Passes Completed": "Passes Completed",
    "Passes Attempted": "Passes Attempted",
    "Yds": "Passing Yards",
    "TD": "Passing Touchdowns",
    "Int": "Interceptions",
    "Sk": "Sacks Taken",
    "Y/A": "Yards Per Attempt",
    "Fantasy Position": "Fantasy Position",
    "year": "Year"
}

# Filter and rename columns for passers
passers_reformatted = passers_df[list(passers_columns_mapping.keys())].rename(columns=passers_columns_mapping)

# Remove rows that are not quarterbacks
passers_reformatted = passers_reformatted[passers_reformatted["Fantasy Position"] == "QB"]

# Define the column mappings for defenses
defenses_columns_mapping = {
    "Tm": "Team",
    "PA": "Points Allowed",
    "Yds": "Total Yards Allowed",
    "TO": "Turnovers",
    "Int": "Interceptions",
    "year": "Year"
}

print(defenses_df.columns)
# Filter and rename columns for defenses
defenses_reformatted = defenses_df[list(defenses_columns_mapping.keys())].rename(columns=defenses_columns_mapping)

# Save the reformatted data to new CSV files
passers_reformatted.to_csv("reformatted_passers.csv", index=False)
defenses_reformatted.to_csv("reformatted_defenses.csv", index=False)

print("Reformatted files saved as 'reformatted_passers.csv' and 'reformatted_defenses.csv'.")