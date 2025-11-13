# 02_preprocessing.py

import pandas as pd
from sklearn.model_selection import train_test_split

# Parametri
INPUT_PATH = 'star_classification.csv'
OUTPUT_PATH = 'data_sample_15k.csv'
TARGET_COL = 'class'
N_SAMPLES = 15000
RANDOM_STATE = 42

#mapping delle classi target
label_map = {
    'GALAXY': 0,
    'STAR': 1,
    'QSO': 2
}


def stratified_sample(df, target_col, n, random_state=42):
    """Restituisce un campione stratificato di n istanze dal dataframe."""
    df_sample, _ = train_test_split(
        df,
        train_size=n,
        stratify=df[target_col],
        random_state=random_state
    )
    return df_sample.reset_index(drop=True)

def clean_dataset(df):
    print(f"Valori nulli:\n{df.isnull().sum()}")
    print(f"Duplicati: {df.duplicated().sum()}")

    # Rimuove duplicati
    df_clean = df.drop_duplicates().reset_index(drop=True)
    return df_clean

def encode_target(df, target_col):
    label_map = {'GALAXY': 0, 'STAR': 1, 'QSO': 2}
    df[target_col] = df[target_col].map(label_map)
    return df

def main():
    df = pd.read_csv('star_classification.csv')
    df_sampled = stratified_sample(df, TARGET_COL, N_SAMPLES)

    # Pulizia
    df_cleaned = clean_dataset(df_sampled)

    # Encoding della colonna target
    df_encoded = encode_target(df_cleaned, TARGET_COL)

    # Salvataggio
    df_encoded.to_csv('data_sample_15k.csv', index=False)
    print(f"Campione pulito ed encodato salvato in: data_sample_15k.csv")


if __name__ == '__main__':
    main()

