# data_loader.py
import pandas as pd
import os
import glob

def load_and_process_data(folder_path='./data/'):
    # Define the files to load
    files_to_load = [
        'sportsref2022.csv',
        'sportsref2023.csv',
        'reformatted_defenses.csv',
        'reformatted_passers.csv'
    ]
    processed_dfs = []

    for file_name in files_to_load:
        file_path = os.path.join(folder_path, file_name)
        if os.path.exists(file_path):
            df = process_file(file_path)
            if df is not None:
                print(f"Loaded {file_name} with Dataset: {df['Dataset'].unique()}")
                processed_dfs.append(df)
        else:
            print(f"File not found: {file_name}")

    if processed_dfs:
        return pd.concat(processed_dfs, ignore_index=True)
    else:
        return pd.DataFrame()  # Return empty DataFrame if nothing was loaded

def process_file(file_path):
    try:
        df = pd.read_csv(file_path)
        df.columns = df.columns.str.strip()

        # Clean the 'Player' column to remove suffixes like * and *+
        if 'Player' in df.columns:
            df['Player'] = df['Player'].str.replace(r'\*|\*\+|\+', '', regex=True)

        # Identify and process format
        if 'Fantasy Rank' in df.columns:
            return process_fantasy_stats(df, file_path)
        elif 'Team' in df.columns and 'Points Allowed' in df.columns:
            return process_defense_stats(df, file_path)
        elif 'Player' in df.columns and 'Week' in df.columns:
            return process_passer_stats(df, file_path)
        else:
            print(f"Skipping unknown format: {file_path}")
            return None
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

def process_fantasy_stats(df, file_path):
    relevant_columns = [
        'Fantasy Rank', 'Player', 'Team', 'Fantasy Position', 'Fantasy Points',
        'Passing Yards', 'Passing TD', 'Rushing Yards', 'Rushing TD',
        'Receiving Yards', 'Receiving TD', 'Year'
    ]
    for col in relevant_columns:
        if col not in df.columns:
            df[col] = None
    df = df[relevant_columns]
    df['Dataset'] = 'Fantasy Stats'
    df['Source File'] = os.path.basename(file_path)
    return df

def process_defense_stats(df, file_path):
    relevant_columns = [
        'Team', 'Points Allowed', 'Total Yards Allowed', 'Turnovers',
        'Interceptions', 'Year'
    ]
    for col in relevant_columns:
        if col not in df.columns:
            df[col] = None
    df = df[relevant_columns]
    df['Dataset'] = 'Defenses'
    df['Source File'] = os.path.basename(file_path)
    return df

def process_passer_stats(df, file_path):
    relevant_columns = [
        'Player', 'Team', 'Week', 'Pass Completion Percentage', 'Passes Completed',
        'Passes Attempted', 'Passing Yards', 'Passing Touchdowns', 'Interceptions',
        'Sacks Taken', 'Yards Per Attempt', 'Fantasy Position', 'Year'
    ]
    for col in relevant_columns:
        if col not in df.columns:
            df[col] = None
    df = df[relevant_columns]
    df['Dataset'] = 'Passers'
    df['Source File'] = os.path.basename(file_path)
    return df
