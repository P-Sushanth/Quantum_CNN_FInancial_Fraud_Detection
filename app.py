from flask import Flask, render_template, request, jsonify
import torch
import numpy as np
import time
import pickle
from torch import nn
import pennylane as qml

app = Flask(__name__)

# -----------------------------
# Load preprocessing objects
# -----------------------------

with open("scaler.pkl","rb") as f:
    scaler = pickle.load(f)

with open("pca.pkl","rb") as f:
    pca = pickle.load(f)

# -----------------------------
# Classical Model
# -----------------------------

class ClassicalANN(nn.Module):

    def __init__(self,input_dim):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(input_dim,32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32,16),
            nn.ReLU(),
            nn.Linear(16,1)
        )

    def forward(self,x):
        return self.model(x)


classic_model = ClassicalANN(30)
classic_model.load_state_dict(torch.load("classical_model.pth"))
classic_model.eval()


# -----------------------------
# Quantum Model
# -----------------------------

n_qubits = 6
dev = qml.device("default.qubit", wires=n_qubits)

@qml.qnode(dev, interface="torch")
def quantum_circuit(inputs, weights):

    qml.AngleEmbedding(inputs, wires=range(n_qubits))
    qml.templates.StronglyEntanglingLayers(weights, wires=range(n_qubits))

    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]


weight_shapes = {"weights": (3,n_qubits,3)}

qlayer = qml.qnn.TorchLayer(quantum_circuit, weight_shapes)


class HybridModel(nn.Module):

    def __init__(self):
        super().__init__()

        self.clayer_in = nn.Linear(4,n_qubits)
        self.qlayer = qlayer
        self.clayer_out = nn.Linear(n_qubits,1)

    def forward(self,x):

        x = self.clayer_in(x)
        x = torch.tanh(x)*np.pi
        x = self.qlayer(x)
        x = self.clayer_out(x)

        return x


quantum_model = HybridModel()
quantum_model.load_state_dict(torch.load("quantum_model.pth"))
quantum_model.eval()


# -----------------------------
# Web Page
# -----------------------------

@app.route("/")
def home():
    return render_template("index.html")


# -----------------------------
# Prediction API
# -----------------------------

@app.route("/predict", methods=["POST"])
def predict():

    values = request.json["features"]

    x = np.array(values).reshape(1,-1)
    x_scaled = scaler.transform(x)

    x_tensor = torch.tensor(x_scaled,dtype=torch.float32)

    # Classical model prediction
    start = time.time()

    logits = classic_model(x_tensor)
    prob_classic = torch.sigmoid(logits).item()

    classical_time = time.time() - start


    # Quantum model prediction
    x_q = pca.transform(x_scaled)
    x_q = torch.tensor(x_q,dtype=torch.float32)

    start = time.time()

    logits_q = quantum_model(x_q)
    prob_quantum = torch.sigmoid(logits_q).item()

    quantum_time = time.time() - start


    return jsonify({

        "classical_prediction": "Fraud" if prob_classic>0.5 else "Not Fraud",
        "classical_prob": prob_classic,
        "classical_time": classical_time,

        "quantum_prediction": "Fraud" if prob_quantum>0.5 else "Not Fraud",
        "quantum_prob": prob_quantum,
        "quantum_time": quantum_time
    })


if __name__ == "__main__":
    app.run(debug=True)