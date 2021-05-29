import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

path = 'digits.csv'
df = pd.read_csv(path)
k = 10

df_y = df['label']
df_X = df.drop('label', axis=1)
X = df_X.to_numpy()
y = df_y.to_numpy()

X_mean = np.mean(X, axis=0)
X_mean_centered = X - X_mean

X_cov = np.cov(X_mean_centered.T)

eigen_values, eigen_vectors = np.linalg.eig(X_cov)

indices = eigen_values.argsort()[::-1]   
eigen_values = eigen_values[indices]
eigen_vectors = eigen_vectors[:,indices]

pve = eigen_values / np.sum(eigen_values)

pves_100 = pve[:k] * 100

k_vals_to_test = [1, 3, 5, 10, 50, 100, 200, 300]
for k_val in k_vals_to_test:
  pcas = eigen_vectors.T[:k_val, :]
  reduced_form = np.dot(X[0], pcas.T)
  recreated = np.dot(reduced_form, pcas) + X_mean
  plt.imshow(recreated.reshape(28, 28), 'gray')
  plt.title("k = " + str(k_val))
  plt.savefig(str(k_val) + ".png")

for i in range(10):
  plt.title("PCA Component " + str(i + 1))
  plt.imshow((eigen_vectors.T)[i].reshape(28, 28), 'gray')
  plt.savefig(str(i) + ".png")