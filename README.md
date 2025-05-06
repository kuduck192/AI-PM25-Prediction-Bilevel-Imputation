# PM2.5 Prediction using MAE and Bilevel Optimization
This repository presents a comprehensive approach to forecasting fine particulate matter (PM₂.₅) levels in Hanoi using deep learning models combined with a bilevel optimization framework for imputing missing environmental data. Our methodology leverages the predictive power of LSTM networks and a tailored Masked Autoencoder (MAE) to address real-world challenges in spatiotemporal air quality datasets.

---

## 🌫️ Problem Statement

PM₂.₅ pollution poses severe risks to public health, especially in densely populated areas like Hanoi. However, reliable PM₂.₅ forecasting is difficult due to:

- Temporal and spatial dependencies in environmental data.
- Significant missing data in meteorological and satellite features.

---

## 🧠 Methodology

Our solution involves:

- **Feature Engineering & Selection**: 28 environmental, meteorological, and remote sensing features were engineered and filtered down to 15 key predictors using statistical and model-based techniques (e.g., SHAP, F-ANOVA, Mutual Information).
- **Regression Models**: Several models were tested, with LSTM yielding the best performance for short time-window forecasting.
- **Masked Autoencoder (MAE)**: Custom Transformer-based MAE model was developed to learn structured imputation from partially observed features.
- **Bilevel Optimization**: To prevent overfitting via teacher-student feedback loops (artifact exploitation), we design a bilevel training strategy with implicit gradients and dropout-masked supervision.


---

## 📁 Project Structure

```plaintext
.
├── input/
│   └── btlaionkk/
│       ├── data_onkk.csv
│       └── data_onkk_merged.csv
├── models/
│   ├── bilevel_impute.py
│   └── mae.py
├── notebooks/
│   ├── nb-impute-mae-bilevel-ig.ipynb
│   ├── nb-impute-mae.ipynb
│   └── nb-lstm.ipynb
├── pdfs/
│   ├── bilevel_details.pdf
│   └── report.pdf
├── LICENSE
└── README.md
````


---

## 📄 Documentation

* 📘 [Full Report](pdfs/report.pdf) — Full analysis, experiments, and results
* 🔬 [Bilevel Optimization Details](pdfs/bilevel_details.pdf) — Mathematical formulation and analysis

---

## 👥 Members

| Name            | GitHub Handle                                    |
| --------------- | ------------------------------------------------ |
| Lê Minh Đức     | [`@kuduck192`](https://github.com/kuduck192)     |
| Hà Tiến Đông    | [`@phapsucongu`](https://github.com/phapsucongu) |
| Nguyễn Tuấn Anh | [`@shinyEazy`](https://github.com/shinyEazy)     |
| Bùi Đức Anh     | [`@Tenebris2`](https://github.com/Tenebris2)     |
