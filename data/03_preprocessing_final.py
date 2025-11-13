import pandas as pd
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

INPUT_PATH = 'data_sample_15k.csv'
OUTPUT_PATH = 'data_ready.csv'
TARGET_COL = 'class'
RANDOM_STATE = 42

def scale_features(X):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled

def apply_smote(X, y):
    smote = SMOTE(random_state=RANDOM_STATE)
    X_res, y_res = smote.fit_resample(X, y)
    return X_res, y_res

def main():
    df = pd.read_csv(INPUT_PATH)

    # Separazione X e y
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]

    # Scaling (evitiamo di scalare ID o colonne categoriche inutili)
    cols_to_scale = X.select_dtypes(include=['float64', 'int64']).columns
    X_scaled = pd.DataFrame(scale_features(X[cols_to_scale]), columns=cols_to_scale)

    # Applichiamo SMOTE
    X_resampled, y_resampled = apply_smote(X_scaled, y)

    # Unione in un unico DataFrame
    df_ready = pd.DataFrame(X_resampled, columns=X_scaled.columns)
    df_ready[TARGET_COL] = y_resampled

    # Salvataggio
    df_ready.to_csv(OUTPUT_PATH, index=False)
    print(f"Dataset bilanciato e scalato salvato in: {OUTPUT_PATH}")

if __name__ == '__main__':
    main()
