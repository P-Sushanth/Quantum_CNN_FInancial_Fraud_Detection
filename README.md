# QuantumShield
## Demo Link : https://huggingface.co/spaces/Sushanth-27/quantum-fraud-detection
### Hybrid Quantum-Classical Neural Network for Financial Fraud Detection

QuantumShield is a web-based application that compares a **Classical Artificial Neural Network (ANN)** and a **Hybrid Quantum-Classical Neural Network (HQNN)** for detecting fraudulent credit card transactions.

The project demonstrates how quantum machine learning can be integrated into traditional deep learning workflows using variational quantum circuits while providing an interactive interface for testing real transaction samples.

---

## Overview

Financial fraud detection is a binary classification problem where transactions are classified as either:

- **0 → Legitimate Transaction**
- **1 → Fraudulent Transaction**

Since fraud accounts for only **0.17%** of the dataset, the problem is highly imbalanced, making metrics such as Precision, Recall, F1-score, and ROC-AUC more meaningful than accuracy alone. :contentReference[oaicite:0]{index=0}

QuantumShield compares the performance of:

- Classical Artificial Neural Network
- Hybrid Quantum Neural Network with a Variational Quantum Circuit

---

## Features

- Interactive fraud detection dashboard
- Classical ANN prediction
- Hybrid Quantum-Classical prediction
- Confidence score visualization
- Prediction latency comparison
- Preloaded authentic and fraudulent test cases
- Clean modern dark UI
- Real-time inference

---

## Tech Stack

### Frontend
- React.js
- Tailwind CSS
- Framer Motion

### Backend
- FastAPI / Flask (depending on implementation)
- Python

### Machine Learning
- TensorFlow / Keras
- Scikit-learn

### Quantum Computing
- PennyLane
- Default Qubit Simulator
- Variational Quantum Circuits (VQC)

---

## Dataset

This project uses the **Credit Card Fraud Detection Dataset** from Kaggle.

Dataset Statistics:

| Property | Value |
|----------|-------|
| Transactions | 284,807 |
| Fraud Cases | 492 |
| Fraud Ratio | 0.17% |
| Features | Time, V1–V28, Amount |

The PCA-transformed features (V1–V28) preserve confidentiality while maintaining predictive information. :contentReference[oaicite:1]{index=1}

---

## Model Architecture

### Classical Model

```
Input Features
      │
Dense Layer
      │
ReLU
      │
Dense Layer
      │
Sigmoid Output
```

---

### Hybrid Quantum Model

```
Input Features
       │
Dense Layer
       │
Quantum Data Encoding
       │
Variational Quantum Circuit
       │
Measurement
       │
Dense Layer
       │
Sigmoid Output
```

The hybrid model encodes classical features into quantum states, applies parameterized quantum gates with entanglement, measures the resulting quantum states, and feeds them back into the classical network for final classification. :contentReference[oaicite:2]{index=2}

---

## Application Preview

### Transaction Analysis

- Input transaction values
- Time
- V1–V28
- Amount

Run inference to compare both models.

---

### Quick Test Cases

The application includes:

- Authentic Transaction
- Fraudulent Transaction

These load verified transaction samples for quick testing.

---

### Comparison Dashboard

Each prediction displays:

- Fraud / Not Fraud
- Confidence Score
- Prediction Latency

allowing direct comparison between the Classical ANN and the Hybrid Quantum model.

---

## Performance

### Classical ANN

| Metric | Value |
|--------|------:|
| Accuracy | 99% |
| ROC-AUC | 0.9764 |
| Training Time | 56.15 s |

---

### Hybrid Quantum Model

| Metric | Value |
|--------|------:|
| Accuracy | 96% |
| ROC-AUC | 0.8166 |
| Training Time | 1076.85 s |

According to the experimental results, the classical ANN outperformed the simulated hybrid quantum model on this dataset, achieving higher precision, recall, ROC-AUC, and significantly faster training time. The project demonstrates the practical comparison between conventional deep learning and current quantum machine learning approaches rather than claiming quantum superiority. :contentReference[oaicite:3]{index=3}

---

## Project Structure

```
quantum-fraud-detection/
│
├── frontend/
│   ├── src/
│   ├── components/
│   └── pages/
│
├── backend/
│   ├── models/
│   ├── routes/
│   ├── quantum/
│   └── app.py
│
├── dataset/
│
├── notebooks/
│
├── screenshots/
│
├── requirements.txt
```

## Future Improvements

- Real quantum hardware execution (IBM Quantum)
- Improved feature engineering
- Better handling of class imbalance using SMOTE/Focal Loss
- Quantum kernel methods
- Model explainability using SHAP/LIME
- Deployment with Docker and Kubernetes

---

## Acknowledgements

- Kaggle Credit Card Fraud Detection Dataset
- PennyLane
- TensorFlow
- Scikit-learn
- React

---

## License

This project is licensed under the MIT License.

---

## Author

**Sushanth**

B.Tech Project — Hybrid Quantum-Classical Neural Network for Financial Fraud Detection
