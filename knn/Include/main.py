import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.decomposition import PCA
from model import KNNCustom

data = load_iris()
X = data.data
y = data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = KNNCustom(k=5, distance_m='euclid')
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
conf = confusion_matrix(y_test, y_pred)

print("Accuracy:", acc)
print("Confusion matrix:\n", conf)

outliers = model.predict(X_test, y_true=y_test, detect_outliers=True)
print("Outlier labels:", outliers)
print("Number of detected outliers:", np.sum(outliers))

pca = PCA(n_components=2)
X_test_2d = pca.fit_transform(X_test)

plt.figure(figsize=(8, 6))

normal_indices = np.where(outliers == 0)[0]
outlier_indices = np.where(outliers == 1)[0]

plt.scatter(X_test_2d[normal_indices, 0], X_test_2d[normal_indices, 1], 
            c='blue', label='Normalne', marker='o', alpha=0.6)
plt.scatter(X_test_2d[outlier_indices, 0], X_test_2d[outlier_indices, 1], 
            c='red', label='Outliery', marker='x', alpha=0.8)

plt.title("Wizualizacja Outlierów z użyciem PCA")
plt.xlabel("PCA1")
plt.ylabel("PCA2")
plt.legend()
plt.show()
