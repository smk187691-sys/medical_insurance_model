# 🏥 Medical Insurance Cost Prediction using Deep Learning

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg?style=for-the-badge&logo=python)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg?style=for-the-badge&logo=tensorflow)](https://www.tensorflow.org/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Pandas](https://img.shields.io/badge/Pandas-150458.svg?style=for-the-badge&logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg?style=for-the-badge)](LICENSE)

An end-to-end Machine Learning and Deep Learning project that predicts individual medical healthcare costs based on demographic and lifestyle indicators (age, BMI, smoking status, region, etc.) using a Multi-Layer Perceptron (Deep Artificial Neural Network).

---

## 📌 Table of Contents
- [Overview](#-overview)
- [Dataset Overview](#-dataset-overview)
- [Pipeline & Architecture](#-pipeline--architecture)
- [Data Preprocessing & Feature Engineering](#-data-preprocessing--feature-engineering)
- [Model Architecture](#-model-architecture)
- [Evaluation & Performance](#-evaluation--performance)
- [Results & Sample Predictions](#-results--sample-predictions)
- [Project Structure](#-project-structure)
- [Installation & Usage](#-installation--usage)
- [Technologies Used](#-technologies-used)
- [Author](#-author)

---

## 📖 Overview

Healthcare costs can vary widely depending on personal demographics and health habits. This project builds a regression model using **Deep Learning (TensorFlow/Keras)** to accurately estimate medical insurance charges. The solution incorporates **Yeo-Johnson power transformation** to normalize highly-skewed target variables and Scikit-Learn `Pipeline` / `ColumnTransformer` for robust, reproducible preprocessing.

---

## 📊 Dataset Overview

The dataset contains patient records with the following attributes:

| Feature | Type | Description |
| :--- | :--- | :--- |
| **`age`** | Numeric | Age of the primary beneficiary (18 – 64) |
| **`sex`** | Categorical | Gender of the insurance contractor (`male`, `female`) |
| **`bmi`** | Numeric | Body Mass Index ($kg/m^2$), ideal range: 18.5 – 24.9 |
| **`children`** | Numeric | Number of children/dependents covered by health insurance (0 – 5) |
| **`smoker`** | Categorical | Smoking habit status (`yes`, `no`) |
| **`region`** | Categorical | Residential area in the US (`northeast`, `northwest`, `southeast`, `southwest`) |
| **`charges`** | Numeric (Target) | Individual medical costs billed by health insurance ($) |

---

## ⚙️ Data Preprocessing & Feature Engineering

1. **Target Variable Transformation**:
   - Medical charges are strongly right-skewed.
   - Applied **`PowerTransformer(method='yeo-johnson')`** to stabilize variance and normalize target distribution for effective gradient descent convergence.
2. **Numerical Pipeline**:
   - Scaled `age`, `bmi`, and `children` using **`StandardScaler`**.
3. **Categorical Pipeline**:
   - Encoded `sex`, `smoker`, and `region` using **`OneHotEncoder(handle_unknown='ignore')`** producing a 11-dimensional feature space.
4. **Train-Test Split**:
   - Partitioned the dataset using an **80/20 train-test ratio** with stratification and random state control.

---

## 🧠 Model Architecture

The regression model is built using a **Multi-Layer Perceptron (Sequential ANN)** in Keras/TensorFlow:

```
Input (11 Features)
       │
       ▼
┌──────────────────────────────┐
│  Dense Layer: 64 Units, ReLU │
└──────────────┬───────────────┘
               │
               ▼
┌──────────────────────────────┐
│  Dense Layer: 32 Units, ReLU │
└──────────────┬───────────────┘
               │
               ▼
┌──────────────────────────────┐
│  Dense Layer: 16 Units, ReLU │
└──────────────┬───────────────┘
               │
               ▼
┌──────────────────────────────┐
│ Dense Output: 1 Unit, Linear │
└──────────────┬───────────────┘
               │
               ▼
Predicted Target (Transformed Charges) ──► Inverse Power Transform ──► Final Charges ($)
```

- **Optimizer:** `Adam(learning_rate=0.002)`
- **Loss Function:** `Mean Squared Error (MSE)`
- **Batch Size:** `32`
- **Epochs:** `150`

---

## 📈 Evaluation & Performance

The model achieves exceptional predictive power on unseen test data:

| Metric | Score (Transformed Scale) | Description |
| :--- | :---: | :--- |
| **$R^2$ Score** | **`0.933` (93.3%)** | High variance explained by features |
| **Mean Squared Error (MSE)** | **`0.064`** | Low squared error variance |
| **Mean Absolute Error (MAE)**| **`0.117`** | Close alignment with ground truth |

---

## 🔬 Results & Sample Predictions

Predictions are converted back to original dollar amounts using inverse power transformation:

| Actual Charges ($) | Predicted Charges ($) | Residual ($) |
| :---: | :---: | :---: |
| **$13,822.80** | **$12,187.52** | - $1,635.28 |
| **$6,358.78** | **$6,890.12** | + $531.34 |
| **$28,287.90** | **$24,109.24** | - $4,178.66 |
| **$7,160.09** | **$7,073.74** | - $86.35 |
| **$17,179.52** | **$17,834.57** | + $655.05 |
| **$13,019.16** | **$12,813.84** | - $205.32 |

---

## 📁 Project Structure

```bash
medical_insurance_model/
│
├── Copy of Untitled3.ipynb      # Main Jupyter Notebook with complete EDA, preprocessing & ANN model
├── README.md                    # Project documentation
└── .git/                        # Git tracking repository
```

---

## 🚀 Installation & Usage

### 1. Clone the repository
```bash
git clone https://github.com/smk187691-sys/medical_insurance_model.git
cd medical_insurance_model
```

### 2. Set up virtual environment (optional but recommended)
```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate
```

### 3. Install required packages
```bash
pip install numpy pandas scikit-learn tensorflow matplotlib seaborn jupyter
```

### 4. Run the Notebook
```bash
jupyter notebook "Copy of Untitled3.ipynb"
```

---

## 🛠️ Technologies Used

- **Language:** Python
- **Data Manipulation & Analysis:** `pandas`, `numpy`
- **Visualization:** `matplotlib`, `seaborn`
- **Machine Learning & Preprocessing:** `scikit-learn` (`StandardScaler`, `OneHotEncoder`, `PowerTransformer`, `ColumnTransformer`, `Pipeline`)
- **Deep Learning Framework:** `TensorFlow` / `Keras`

---

## 👤 Author

- **GitHub:** [@smk187691-sys](https://github.com/smk187691-sys)
- **Repository:** [medical_insurance_model](https://github.com/smk187691-sys/medical_insurance_model)
