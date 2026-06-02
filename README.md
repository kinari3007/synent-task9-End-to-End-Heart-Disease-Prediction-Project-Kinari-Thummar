# ❤️ Heart Disease Prediction — End-to-End ML Project

**InfyChain Summer Internship | Task 9 | Kinari Thummar**

A production-grade, end-to-end machine learning web application that predicts whether a patient has heart disease based on clinical features. Built with a modular pipeline architecture mirroring real-world ML deployment practices.

---

## 🌐 Live Demo

> Run locally using the setup instructions below.  
> Open `http://127.0.0.1:5000` after starting the Flask app.

---

## 📌 Problem Statement

Heart disease is one of the leading causes of death worldwide. Early prediction using patient clinical data can significantly improve treatment outcomes. This project builds a binary classification model that predicts the presence or absence of heart disease from 13 clinical features.

---

## 📂 Dataset

| Detail | Info |
|--------|------|
| Source | [UCI Heart Disease Dataset via Kaggle](https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset) |
| Rows | 1025 |
| Features | 13 clinical inputs |
| Target | `target` — 0 (No Disease), 1 (Heart Disease) |

### Features Used

| Feature | Description |
|---------|-------------|
| age | Age of patient |
| sex | Sex (0 = Female, 1 = Male) |
| cp | Chest pain type (0–3) |
| trestbps | Resting blood pressure |
| chol | Serum cholesterol (mg/dl) |
| fbs | Fasting blood sugar > 120 mg/dl (0/1) |
| restecg | Resting ECG results (0–2) |
| thalach | Maximum heart rate achieved |
| exang | Exercise-induced angina (0/1) |
| oldpeak | ST depression induced by exercise |
| slope | Slope of peak exercise ST segment (0–2) |
| ca | Number of major vessels coloured by fluoroscopy (0–4) |
| thal | Thalassemia type (0–3) |

---

## 🏗️ Project Structure

```
synent-task9-End-to-End-Heart-Disease-Prediction-Project-Kinari-Thummar/
│
├── artifacts/                  # Auto-generated (git-ignored)
│   ├── heart.csv               # Raw dataset
│   ├── train.csv               # Train split
│   ├── test.csv                # Test split
│   ├── model.pkl               # Saved best model
│   └── preprocessor.pkl        # Saved preprocessing pipeline
│
├── notebook/
│   └── EDA_and_Training.ipynb  # Exploratory analysis & experiments
│
├── src/
│   ├── __init__.py
│   ├── exception.py            # Custom exception handler
│   ├── logger.py               # Logging setup
│   ├── utils.py                # Save/load pkl, model evaluation
│   │
│   ├── components/
│   │   ├── __init__.py
│   │   ├── data_ingestion.py       # Loads CSV, creates train/test split
│   │   ├── data_transformation.py  # Imputation + scaling pipeline
│   │   └── model_trainer.py        # Trains & selects best model
│   │
│   ├── pipeline/
│   │   ├── __init__.py
│   │   ├── train_pipeline.py       # Orchestrates full training flow
│   │   └── predict_pipeline.py     # Loads artifacts, returns prediction
│   │
│   └── templates/
│       ├── index.html          # Landing page
│       └── home.html           # Prediction form + result page
│
├── app.py                      # Flask app — routes & prediction logic
├── setup.py                    # Makes src/ pip-installable
├── requirements.txt            # All dependencies
├── .gitignore
└── README.md
```

---

## 🔄 Project Workflow

```
heart.csv
    │
    ▼
Data Ingestion        →  train.csv + test.csv
    │
    ▼
Data Transformation   →  Impute missing values + StandardScaler → preprocessor.pkl
    │
    ▼
Model Training        →  Logistic Regression, Decision Tree, Random Forest
                          Best model saved → model.pkl
    │
    ▼
Flask App             →  User inputs → predict_pipeline → result
```

---

## 🤖 Models Evaluated

| Model | Description |
|-------|-------------|
| Logistic Regression | Linear baseline classifier |
| Decision Tree | Rule-based depth-5 tree |
| Random Forest | 100-estimator ensemble (best performer) |

**Best Model:** Random Forest  
**Accuracy:** ~88–92% on test set

---

## 📊 Key Insights from EDA

- Patients with **chest pain type 0** (asymptomatic) have the highest heart disease rate
- **Max heart rate** is inversely correlated with heart disease — lower max HR = higher risk
- **ST depression (oldpeak)** is one of the strongest predictors
- **Thalassemia type** and **number of vessels** are the top two feature importances in Random Forest
- Males in this dataset show a higher prevalence of heart disease than females

---

## ⚙️ Setup Instructions

### 1. Clone the repository

```bash
git clone https://github.com/kinari3007/synent-task9-End-to-End-Heart-Disease-Prediction-Project-Kinari-Thummar.git
cd synent-task9-End-to-End-Heart-Disease-Prediction-Project-Kinari-Thummar
```

### 2. Install dependencies

```bash
pip install -e .
pip install -r requirements.txt
```

### 3. Add the dataset

Download `heart.csv` from [Kaggle](https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset) and place it in `artifacts/heart.csv`.

### 4. Train the model

```bash
python src/pipeline/train_pipeline.py
```

This generates `artifacts/model.pkl` and `artifacts/preprocessor.pkl`.

### 5. Run the app

```bash
python app.py
```

Open `http://127.0.0.1:5000` in your browser.

---

## 🖥️ App Preview

| Page | Description |
|------|-------------|
| `/` | Landing page with project info |
| `/predict` | Form to enter patient details and get prediction |

---

## 🛠️ Tech Stack

| Tool | Purpose |
|------|---------|
| Python 3.x | Core language |
| Flask | Web framework |
| Scikit-learn | ML models + preprocessing |
| Pandas & NumPy | Data manipulation |
| Joblib | Model serialization |
| HTML/Jinja2 | Frontend templates |

---

## 📋 Internship Details

| Field | Info |
|-------|------|
| Organization | InfyChain Tech Solutions Pvt. Ltd. |
| Program | Summer Data Science Internship |
| Task | Task 9 — End-to-End Data Science Project |
| Intern | Kinari Thummar |


