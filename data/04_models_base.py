# 04_models_base.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

# Caricamento del dataset pronto
df = pd.read_csv('data_ready.csv')

# Separazione feature e target
X = df.drop('class', axis=1)
y = df['class']

# Train-test split stratificato
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.3,
    random_state=42,
    stratify=y
)

# Inizializzazione e training del modello
tree_clf = DecisionTreeClassifier(random_state=42)
tree_clf.fit(X_train, y_train)

# Valutazione
y_pred = tree_clf.predict(X_test)

print("🌳 Decision Tree Classifier 🌳")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:")
print(classification_report(y_test, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Salvataggio modello
#joblib.dump(tree_clf, 'tree_model.pkl')
joblib.dump(tree_clf, 'models/tree_model.pkl') #salviamo gli output nella subdir models

# -----------------------------
# 🌌 K-Nearest Neighbors (KNN)
# -----------------------------
from sklearn.neighbors import KNeighborsClassifier

print("\n🔵 K-Nearest Neighbors Classifier 🔵")

# Istanzia e allena il KNN (k=5 come valore standard iniziale)
knn_clf = KNeighborsClassifier(n_neighbors=5)
knn_clf.fit(X_train, y_train)

# Predizioni
y_pred_knn = knn_clf.predict(X_test)

# Accuracy
print("Accuracy:", accuracy_score(y_test, y_pred_knn))

# Classification report
print("Classification Report:")
print(classification_report(y_test, y_pred_knn))

# Confusion Matrix
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_knn))

# Salva il modello KNN
joblib.dump(knn_clf, 'models/knn_model.pkl')


# -----------------------------
# ⚙️ Support Vector Machine (SVM)
# -----------------------------
from sklearn.svm import SVC

print("\n⚙️ Support Vector Machine (SVM) ⚙️")

# Istanzia e allena il support vector machine
svm_clf = SVC(random_state=42)
svm_clf.fit(X_train, y_train)

# Predizioni
y_pred_svm = svm_clf.predict(X_test)

# Accuracy
print("Accuracy:", accuracy_score(y_test, y_pred_svm))

# Classification report
print("Classification Report:")
print(classification_report(y_test, y_pred_svm))

# Confusion Matrix
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_svm))

# Salva il modello support vector machine
joblib.dump(svm_clf, 'models/svm_model.pkl')

# -----------------------------
# 🧠 Naive Bayes
# -----------------------------
from sklearn.naive_bayes import GaussianNB

print("\n🧠 Naive Bayes Classifier 🧠")

# Istanzia e allena il Naive Bayes
nb_clf = GaussianNB()
nb_clf.fit(X_train, y_train)

# Predizioni
y_pred_nb = nb_clf.predict(X_test)

# Accuracy
print("Accuracy:", accuracy_score(y_test, y_pred_nb))

# Classification report
print("Classification Report:")
print(classification_report(y_test, y_pred_nb))

# Confusion Matrix
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_nb))

# Salva il modello Naive Bayes
joblib.dump(nb_clf, 'models/naive_bayes_model.pkl')


# 🧠 Rete Neurale (MLPClassifier)
print("\n🧠 Rete Neurale (MLPClassifier) 🧠")
from sklearn.neural_network import MLPClassifier

#una sola hidden layer con 50 neuroni
#500: numero massimo di iterazioni per la convergenza
#42: per riproducibilità
mlp_clf = MLPClassifier(hidden_layer_sizes=(50,), max_iter=500, random_state=42)
mlp_clf.fit(X_train, y_train)
y_pred_mlp = mlp_clf.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred_mlp))
print("Classification Report:\n", classification_report(y_test, y_pred_mlp))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_mlp))

# Salvataggio modello
joblib.dump(mlp_clf, 'models/mlp_model.pkl')





