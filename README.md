# 💳 Credit Risk Prediction System

<div align="center">

**A Production-Style Machine Learning Application for Credit Risk Assessment**

[![Python](https://img.shields.io/badge/Python-3.12+-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-red.svg)](https://streamlit.io/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange.svg)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

[Overview](#-overview) • [Features](#-features) • [Architecture](#-solution-architecture) • [Installation](#-installation) • [Usage](#-usage) • [Tech Stack](#-tech-stack)

</div>

---

## 🎯 Overview

The **Credit Risk Prediction System** is an end-to-end machine learning application designed to simulate **real-world credit risk evaluation workflows** used by financial institutions.

The system leverages a trained **Extra Trees Classifier** to assess applicant risk as **GOOD** or **BAD**, supported by a probability score that enhances interpretability and decision-making. A **Streamlit-based web interface** enables real-time inference in a clean, production-oriented environment.

This project focuses on **deployment correctness, feature consistency, and inference reliability**, rather than only model accuracy.

---

## ✨ Key Features

* 📊 **Real-Time Credit Risk Prediction**
* 🧠 **Extra Trees Machine Learning Model**
* 🔄 **Encoder-based categorical preprocessing**
* 🧾 **Strict feature name & order enforcement**
* 📈 **Probability-based risk scoring**
* 🖥️ **Interactive Streamlit web interface**
* 🛠️ **Production-style inference pipeline**

---

## 🧩 Problem Statement

Credit institutions evaluate multiple demographic and financial factors to estimate the likelihood of default before approving credit.

This system replicates that decision-making process by:

* Processing structured applicant data
* Applying consistent preprocessing techniques
* Generating risk classification with confidence scores

---

## 🏗️ Solution Architecture

```
User Input (Streamlit UI)
        ↓
Categorical Encoding (Saved Encoders)
        ↓
Feature Alignment (Training Schema Enforcement)
        ↓
Extra Trees Model Inference
        ↓
Risk Classification + Probability Score
```

The architecture ensures **training-to-inference parity**, preventing common deployment errors such as feature mismatch and encoder inconsistency.

---

## 🧠 Machine Learning Approach

* **Model**: Extra Trees Classifier
* **Learning Type**: Supervised Binary Classification
* **Target Variable**: Credit Risk (GOOD / BAD)

### Input Features

* Age
* Sex
* Job
* Housing
* Saving accounts
* Checking account
* Credit amount
* Duration

All categorical features are transformed using **pre-trained encoders** saved during training.

---

## 🖥️ Application Workflow

1. User enters applicant details via the web interface
2. Categorical values are transformed using stored encoders
3. Input features are reordered to match the training schema
4. Model generates:

   * Risk classification (GOOD / BAD)
   * Prediction probability
5. Result is displayed in real time

---

## 📁 Project Structure

```
Credit_Risk_Modelling_Using_ML/
│
├── app.py                          # Streamlit inference application
├── note.ipynb                      # Model training & experimentation
├── extra_trees_credit_model.pkl    # Trained ML model
├── feature_names.pkl               # Training feature schema
├── Sex_encoder.pkl
├── Housing_encoder.pkl
├── Saving accounts_encoder.pkl
├── Checking account_encoder.pkl
├── german_credit_data.csv          # Dataset
└── README.md
```

---

## 🚀 Installation

### Prerequisites

* Python 3.12+
* Virtual environment support

### Step 1: Clone the Repository

```bash
git clone https://github.com/your-username/credit-risk-prediction.git
cd credit-risk-prediction
```

### Step 2: Create Virtual Environment

```bash
python -m venv venv
venv\Scripts\activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Run the Application

```bash
streamlit run app.py
```

The application will launch at:

```
http://localhost:8501
```

---

## 📖 Usage

* Enter applicant information through the form
* Click **Predict Credit Risk**
* View:

  * Risk classification (GOOD / BAD)
  * Confidence probability

---

## 🛠️ Tech Stack

### Core Technologies

* **Python**
* **Streamlit** – Web interface
* **Pandas** – Data handling
* **scikit-learn** – Machine learning
* **Joblib** – Model & encoder persistence

### ML Components

* Extra Trees Classifier
* Label Encoders
* Probability-based prediction output

---

## 🔍 Engineering Focus Areas

* Training vs inference feature alignment
* Encoder reuse across deployment stages
* Robust handling of categorical variables
* Real-world ML debugging and validation
* Production-safe Streamlit integration

---

## ⚠️ Disclaimer

This project uses **publicly available data** and is intended strictly for **educational and demonstration purposes**.
It does not represent proprietary models, datasets, or decision systems used by financial institutions.

---

## 👤 Author

**Hitesh Moota**
B.E. Artificial Intelligence & Data Science
Aspiring Data Analyst / Machine Learning Engineer

---

<div align="center">

**Built with a focus on real-world ML deployment practices**

</div>

---
