# ⚛️ Quantum vs Classical Fraud Detection Web Application

A hybrid Quantum-Classical Machine Learning web application that compares the performance, inference latency, and prediction metrics of a classical Artificial Neural Network (ANN) against a Variational Quantum Circuit (VQC) / Quantum Neural Network (QNN) built with **PennyLane** and **PyTorch**.

---

## 📌 Features

- **Hybrid Quantum Model**: Utilizes PennyLane's `StronglyEntanglingLayers` and `AngleEmbedding` on a 6-qubit quantum simulator (`default.qubit`), wrapped in a PyTorch layer (`TorchLayer`).
- **Classical Neural Network**: Implements a PyTorch Artificial Neural Network (`ClassicalANN`) trained on 30 input features.
- **Dimensionality Reduction & Preprocessing**: Uses `StandardScaler` for scaling and `PCA` (Principal Component Analysis) to reduce feature dimensions from 30 down to 4 for efficient quantum circuit ingestion.
- **Real-Time Benchmarking**: Interactive web UI allowing real-time fraud probability prediction and direct comparison of execution times (inference latency) between Classical and Quantum models.
- **Containerized Deployment**: Includes a production-ready `Dockerfile` configured for deployment on platforms like Hugging Face Spaces or Docker containers using `gunicorn`.

---

## 🔗 Repository & Live Demo

- **GitHub Repository**: [P-Sushanth/Quantum_CNN_FInancial_Fraud_Detection](https://github.com/P-Sushanth/Quantum_CNN_FInancial_Fraud_Detection)
- **Live Demo Space**: [Hugging Face Space / Live Demo](https://huggingface.co/spaces/P-Sushanth/Quantum_CNN_Financial_Fraud_Detection) *(Update with exact link if hosted on HF Spaces or equivalent)*

---

## 🏗 Project Architecture

```
qml_fraud_detection/
│
├── Quantum_Fraud_Detection.ipynb  # Jupyter Notebook for EDA, data preprocessing, and model training
├── app.py                         # Flask Web Server & Model Inference API
├── classical_model.pth            # Trained PyTorch state dict for Classical ANN
├── quantum_model.pth              # Trained PyTorch state dict for Hybrid Quantum Model
├── scaler.pkl                     # Saved Scikit-Learn StandardScaler object
├── pca.pkl                        # Saved Scikit-Learn PCA object
├── creditcard.csv                 # Dataset (Credit Card Fraud Detection)
├── check_dims.py                  # Script to verify model dimension compatibility
├── requirements.txt               # Python package dependencies
├── Dockerfile                     # Containerization script for Gunicorn deployment
├── templates/
│   └── index.html                 # Web UI Template
└── static/                        # Static assets (CSS/JS)
```

---

## ⚡ Model Architectures

### 1. Classical Model (`ClassicalANN`)
- **Input Dimension**: 30 features
- **Architecture**:
  - `Linear(30, 32)` $\rightarrow$ `ReLU` $\rightarrow$ `Dropout(0.3)`
  - `Linear(32, 16)` $\rightarrow$ `ReLU`
  - `Linear(16, 1)` $\rightarrow$ Output Logits

### 2. Quantum Model (`HybridModel`)
- **Preprocessing**: Dimensionality reduced via PCA from 30 to 4 dimensions.
- **Architecture**:
  - Classical Linear In Layer: `Linear(4, 6)` $\rightarrow$ Tanh activation scaled by $\pi$
  - Quantum Circuit (`PennyLane QNode`):
    - **Embedding**: `AngleEmbedding` across 6 qubits
    - **Variational Circuit**: 3 layers of `StronglyEntanglingLayers`
    - **Measurement**: Expectation value of $\text{PauliZ}$ operator across all 6 qubits
  - Classical Linear Out Layer: `Linear(6, 1)` $\rightarrow$ Output Logits

---

## 🛠️ Getting Started

### Prerequisites

- Python 3.10+
- `pip` package manager

### Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/P-Sushanth/Quantum_CNN_FInancial_Fraud_Detection.git
   cd Quantum_CNN_FInancial_Fraud_Detection
   ```

2. **Create a virtual environment (Optional but recommended)**:
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

---

## 🚀 Running the Application

### 1. Local Development Server

Run the Flask application locally:

```bash
python app.py
```

Open your browser and navigate to `http://127.0.0.1:5000`.

### 2. Docker Deployment

Build and run using Docker (listens on port 7860):

```bash
# Build Docker image
docker build -t qml-fraud-detection .

# Run Docker container
docker run -p 7860:7860 qml-fraud-detection
```

Access the app at `http://localhost:7860`.

---

## 🔌 API Endpoint

### `POST /predict`

Send a JSON payload containing 30 feature values (matching the Credit Card Fraud dataset format):

**Request Payload**:
```json
{
  "features": [0.1, -1.2, 0.4, ..., 0.05]
}
```

**Response Payload**:
```json
{
  "classical_prediction": "Not Fraud",
  "classical_prob": 0.0124,
  "classical_time": 0.0015,
  "quantum_prediction": "Not Fraud",
  "quantum_prob": 0.0241,
  "quantum_time": 0.0452
}
```

---

## 📊 Notebook & Training

The `Quantum_Fraud_Detection.ipynb` notebook contains:
- Exploratory Data Analysis (EDA) on the `creditcard.csv` dataset.
- Data scaling with `StandardScaler` and dimensionality reduction via `PCA`.
- Model definition, loss function (BCEWithLogitsLoss), and optimization.
- Exporting trained model artifacts (`.pth`, `.pkl`).

---

## 📜 License

This project is open-source and available under the [MIT License](LICENSE).
