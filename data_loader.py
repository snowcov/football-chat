# data_loader.py
import pandas as pd
import os
import glob

def load_and_process_data(folder_path='./data/'):
    all_files = glob.glob(os.path.join(folder_path, '*.csv'))
    processed_dfs = []

    for file_path in all_files:
        df = process_file(file_path)
        if df is not None:
            processed_dfs.append(df)

    if processed_dfs:
        return pd.concat(processed_dfs, ignore_index=True)
    else:
        return pd.DataFrame()  # Return empty DataFrame if nothing was loaded

def process_file(file_path):
    try:
        df = pd.read_csv(file_path)
        df.columns = df.columns.str.strip()

        # Identify and process format
        if 'Fantasy Position' in df.columns:
            return process_fantasy_stats(df, file_path)
        elif 'Player Name' in df.columns and 'Total Yards' in df.columns:
            return process_team_stats(df, file_path)
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
        'Receiving Yards', 'Receiving TD'
    ]
    for col in relevant_columns:
        if col not in df.columns:
            df[col] = None
    df = df[relevant_columns]
    # Extract year from filename, e.g., sportsref2022.csv -> 2022
    basename = os.path.basename(file_path)
    year_match = None
    import re
    m = re.search(r'(\d{4})', basename)
    if m:
        year_match = int(m.group(1))
    df['Year'] = year_match
    df['Source File'] = basename
    return df

def process_team_stats(df, file_path):
    df = df[['Player Name', 'Total Yards']]
    df.rename(columns={'Player Name': 'Player', 'Total Yards': 'Yards'}, inplace=True)
    df['Fantasy Points'] = df['Yards'] / 10
    df['Fantasy Position'] = 'Unknown'
    df['Source File'] = os.path.basename(file_path)
    return df
