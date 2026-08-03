import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

try:
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    print(f'Scaler features: {scaler.n_features_in_}')
except Exception as e:
    print(f'Error loading scaler: {e}')

try:
    with open('pca.pkl', 'rb') as f:
        pca = pickle.load(f)
    print(f'PCA features: {pca.n_features_in_}')
    print(f'PCA components: {pca.n_components_}')
except Exception as e:
    print(f'Error loading PCA: {e}')
