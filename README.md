# Cancer Detection ML

A machine learning project to classify tumors as **benign** or **malignant** using **Logistic Regression**.

# Overview

This project uses a dataset containing features of breast cancer tumors to predict whether a tumor is malignant or benign. The model performs:

* Data cleaning and preprocessing
* Handling missing values
* Exploratory Data Analysis (EDA)
* Training and evaluating a **Logistic Regression** model
* Prediction on unseen data

The goal is to assist in early detection of cancer using machine learning techniques.

# Dataset

The dataset `Cancer_Data.xlsx` contains features extracted from breast cancer images. The target variable is:

* `diagnosis`:

  * `0` → Malignant
  * `1` → Benign

**Note:** The dataset is included in the repository for demonstration purposes.

# Installation

1. Clone the repository:

```bash
git clone https://github.com/yourusername/Cancer-Detection-ML.git
cd Cancer-Detection-ML
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

# Usage

Run the Python script to train the model and make predictions:

```bash
python cancer_detection.py
```

You can also modify `input_data` in the script to predict on custom tumor data.

# Features & Skills Applied

* **Python**
* **Pandas** & **NumPy** for data handling
* **Scikit-learn** for machine learning (Logistic Regression)
* **Data preprocessing** (handling missing values, dropping irrelevant columns)
* **Train/Test split & model evaluation**
* **Prediction on unseen data**

# Repository Structure

```
Cancer-Detection-ML/
│
├── cancer_detection.py      - Main ML script
├── Cancer_Data.xlsx         - Dataset
├── README.md                - Project documentation
├── requirements.txt         - Python dependencies
```
