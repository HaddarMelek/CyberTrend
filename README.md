# 📡 Predicting Cyberattacks Using Time-Series Data and Social Trends

## 🧠 Project Overview

The goal of this project is to develop models that predict potential cyberattacks by analyzing time-series data of past attacks and correlating it with trending news from Google News.

### 📈 Impact of News Trends in Prediction

Social media and news events can serve as external indicators of cyberattack risks. Several studies have explored integrating external signals (e.g., Twitter sentiment analysis) to improve predictions. This project investigates the influence of news headlines on forecasting cyber threats.

---

## 🗂️ Dataset

### 📁 `cyber_data.csv`

This dataset provides a comprehensive view of global cyberattacks, including:
- Attack type
- Date of occurrence
- Affected country

The dataset was sourced from an open-access publication by MDPI and is also available on GitHub to support research in cybersecurity and threat intelligence.

### 📰 Google News Collection

To enrich the cyberattack data with contextual information, news headlines were collected using Google News RSS feeds. For each country and date found in the attack data, the top 5 relevant news titles were retrieved.

---

## 📊 Exploratory Data Analysis (EDA)

The EDA phase explored:
- Frequency of attacks over time and across countries
- Correlations between attacks and news volume
- Visualization of attack patterns
- Preliminary signal detection from news headlines

---

## 🧹 Preprocessing and Feature Extraction

### 🔧 Cleaning and Normalization

Data cleaning included:
- Handling missing values
- Normalizing numeric values

### ➕ Derived Features

A new variable `total_attacks` was created by aggregating counts of attack types:
- Spam
- Ransomware
- Local Infection
- Exploit
- Malicious Mail
- Network Attack
- On-Demand Scan
- Web Threat

This simplifies modeling by capturing overall cyber threat intensity.

### 🧠 News Embedding

Each news title was embedded into a 384-dimensional vector using:
- `SentenceTransformer` with `distilbert-base-nli-stsb-mean-tokens`

### ⏳ Temporal Alignment

News data was time-aligned with the cyberattack data to ensure consistency during model training.

---

## 🤖 Predictive Modeling

We developed and compared several predictive models:
- **ARIMA**
- **LightGBM**
- **XGBoost**
- **Facebook Prophet**
- **LSTM (Long Short-Term Memory)**

Each model was evaluated with and without the inclusion of news embeddings to assess the added value of contextual news signals.

---

## 📌 Conclusion

This project highlights the potential of combining traditional time-series modeling with external social signals to enhance cyber threat prediction. News headlines, when encoded and properly aligned, offer valuable insight into emerging threats.
