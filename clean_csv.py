import pandas as pd
import glob
import os

def clean_csv_files(folder='./data'):
    csv_files = glob.glob(os.path.join(folder, '*.csv'))
    for file in csv_files:
        df = pd.read_csv(file)
        # Remove * and + from all string columns
        for col in df.select_dtypes(include='object').columns:
            df[col] = df[col].str.replace(r'[\*\+]', '', regex=True)
        df.to_csv(file, index=False)
        print(f"Cleaned: {file}")

if __name__ == "__main__":
    clean_csv_files()