import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, accuracy_score, f1_score

# Percorso ai modelli salvati
MODEL_PATH = "models"
MODEL_NAMES = [
    ("Decision Tree", "tree_model.pkl"),
    ("K-Nearest Neighbors", "knn_model.pkl"),
    ("Support Vector Machine", "svm_model.pkl"),
    ("Naive Bayes", "naive_bayes_model.pkl"),
    ("Neural Network (MLP)", "mlp_model.pkl"),
    ("Random Forest", "random_forest_model.pkl"),
    ("Bagging", "bagging_model.pkl"),
    ("AdaBoost", "boosting_model.pkl")
]

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd

DATA_PATH = "data_ready.csv"
df = pd.read_csv(DATA_PATH)
X = df.drop("class", axis=1)
y = df["class"]

#divisione test set coerente con quella usata durante il training
_, X_test, _, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)

# Scaling (necessario per MLP, SVM, KNN)
scaler = StandardScaler()
X_test_scaled = scaler.fit(X).transform(X_test)

# Raccoglitore risultati
results = []

for name, filename in MODEL_NAMES:
    model = joblib.load(os.path.join(MODEL_PATH, filename))

    # Usa lo scaling solo per i modelli che lo richiedono
    if name in ["K-Nearest Neighbors", "Support Vector Machine", "Neural Network (MLP)"]:
        X_input = X_test_scaled
    else:
        X_input = X_test

    y_pred = model.predict(X_input)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="macro")

    results.append({
        "Model": name,
        "Accuracy": acc,
        "F1-macro": f1
    })

# Creazione DataFrame risultati
results_df = pd.DataFrame(results)
results_df = results_df.sort_values(by="Accuracy", ascending=False)
results_df = results_df.reset_index(drop=True)
print("\n📊 Confronto dei Modelli:\n")
print(results_df.to_string(index=False))

bar_width = 0.4
indices = np.arange(len(results_df))

# Bar plot
plt.figure(figsize=(10, 6))
plt.barh(indices + bar_width/2, results_df["Accuracy"], height=bar_width, color="skyblue", label="Accuracy")
plt.barh(indices - bar_width/2, results_df["F1-macro"], height=bar_width, color="orange", alpha=0.7, label="F1-macro")
plt.yticks(indices, results_df["Model"])
plt.xlabel("Score")
plt.title("Confronto tra Modelli")
plt.legend()
plt.tight_layout()
plt.gca().invert_yaxis()
plt.show()
