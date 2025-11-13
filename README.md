# 🌌 Esplorazione Stellare - Classificazione di Oggetti Celesti

> Un approccio di Machine Learning per la classificazione di stelle, galassie e quasar utilizzando i dati della Sloan Digital Sky Survey (SDSS) per un progetto universitario.

## Descrizione del Progetto

Il progetto **Esplorazione Stellare** nasce con l’obiettivo di automatizzare la classificazione degli oggetti celesti a partire dai dati spettrali. L'analisi copre l'intero ciclo di vita del dato: dall’analisi esplorativa (EDA) e preprocessing avanzato, fino all’addestramento e alla valutazione comparativa di molteplici modelli di Machine Learning.

L'obiettivo finale è identificare il modello più performante per distinguere tra le tre classi principali: **GALAXY**, **STAR** e **QSO** (Quasar).

## Guida alla Lettura
Tutta la narrazione del progetto, dall'analisi visiva dei grafici alle motivazioni tecniche dietro la scelta dei modelli, è documentata passo dopo passo nel notebook principale:
**`notebooks/Esplorazione_stellare.ipynb`**

## Tecnologie Utilizzate

Il progetto è sviluppato in **Python** sfruttando le principali librerie per Data Science:

* **Manipolazione Dati:** `pandas`, `numpy`
* **Visualizzazione:** `matplotlib`, `seaborn`
* **Machine Learning:** `scikit-learn`
* **Bilanciamento Dati:** `imblearn` (SMOTE)
* **Gestione Modelli:** `joblib`

---

## Pipeline del Progetto

Il flusso di lavoro segue una numerazione progressiva per garantire ordine e riproducibilità:

### 1.  Esplorazione Dati (`01_esplorazione_stellare.ipynb`)
* Analisi delle feature spettrali (redshift, bande u, g, r, i, z).
* Visualizzazione della distribuzione delle classi (forte sbilanciamento verso *GALAXY*).
* Studio delle correlazioni tramite Heatmap.

### 2.  Preprocessing Iniziale (`02_preprocessing.py`)
* Caricamento del dataset grezzo (`star_classification.csv`).
* **Campionamento Stratificato:** Riduzione del dataset a **15.000 istanze** (`data_sample_15k.csv`) mantenendo le proporzioni originali delle classi per velocizzare il training senza perdere rappresentatività.
* Pulizia base dei dati (rimozione ID non necessari).

### 3.  Preprocessing Finale & Bilanciamento (`03_preprocessing_final.py`)
* **Encoding:** Trasformazione della target variable (`GALAXY=0`, `STAR=1`, `QSO=2`).
* **Scaling:** Standardizzazione delle feature numeriche (StandardScaler) per ottimizzare la convergenza.
* **SMOTE (Synthetic Minority Over-sampling):** Generazione di dati sintetici per bilanciare le classi minoritarie (*STAR* e *QSO*) ed evitare bias verso la classe dominante.
* Output: Generazione del file pronto per il training (`data_ready.csv`).

### 4.  Training dei Modelli (`04_...` e `05_...`)
I modelli vengono addestrati su un split 70% Train / 30% Test e salvati in formato `.pkl`.
* **Classificatori Base (`04_models_base.py`):**
    * Decision Tree, K-Nearest Neighbors (KNN), Support Vector Machine (SVM), Naive Bayes, Multi-Layer Perceptron (MLP).
* **Modelli Ensemble (`05_models_ensemble.py`):**
    * Random Forest, Bagging Classifier, AdaBoost.

### 5.  Valutazione Comparativa (`06_evaluation.py`)
* Caricamento automatico di tutti i modelli addestrati dalla cartella `data/models`.
* Calcolo delle metriche chiave: **Accuracy** e **F1-Macro** (essenziale per classi sbilanciate).
* Generazione di report comparativi per decretare il vincitore.

---

## Risultati Finali

Di seguito il confronto delle performance sui dati di test. I modelli Ensemble hanno dimostrato una superiorità netta.

| Modello | Accuracy | F1-macro |
| :--- | :---: | :---: |
|  **Random Forest** | **0.9871** | **0.9871** |
|  **Bagging** | 0.9863 | 0.9863 |
|  Decision Tree | 0.9717 | 0.9717 |
|  Neural Network (MLP) | 0.9578 | 0.9578 |
|  SVM | 0.9485 | 0.9486 |
|  AdaBoost | 0.9276 | 0.9270 |
|  K-Nearest Neighbors | 0.9161 | 0.9165 |
|  Naive Bayes | 0.8943 | 0.8946 |

### Analisi dei Risultati

*  **Top Performers:** **Random Forest** e **Bagging** si confermano i modelli migliori. La combinazione di più alberi decisionali riduce drasticamente la varianza e previene l’overfitting, adattandosi perfettamente alla complessità dei dati spettrali.
*  **Deep Learning:** Il modello **MLP** e l'**SVM** hanno fornito ottime prestazioni catturando relazioni non lineari, ma con tempi di training sensibilmente più elevati rispetto agli alberi.
*  **Punti Deboli:** **AdaBoost** ha risentito del "rumore" nei dati, mentre **KNN** e **Naive Bayes** si sono dimostrati troppo semplici per gestire efficacemente le correlazioni tra le feature spettrali.

---

## Autori

* 👨‍🚀 **Vito Simone Goffredo**
* 🧑‍🚀 **Andrea Attadia**
