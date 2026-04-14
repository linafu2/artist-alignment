import pandas as pd
import numpy as np
from numpy.linalg import eig, svd
import matplotlib.pyplot as plt

# load datasets
genre = pd.read_csv("genre_features.csv")
artist = pd.read_csv("artist_features.csv")

G = genre.values
A = artist.values

# center genre matrix
genre_mean = np.mean(G, axis=0)
G_centered = G - genre_mean

# covariance matrix
C = G_centered.T @ G_centered

# eigendecomposition
eigenvalues, eigenvectors = eig(C)

print("Eigenvalues:")
print(eigenvalues)

# project artist songs into genre eigenspace
A_centered = A - genre_mean
projection = A_centered @ eigenvectors

print("Artist projections:")
print(projection)