import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
import os

#filename and folder
fname= "walking_1"
train_folder = "/Users/hjy/AMATH/AMATH 582/HW 2/hw2data/train/"


def load_data(folder):
    data = []
    labels = []
    label_map = {'walking': 0, 'jumping': 1, 'running': 2}
    for movement, label in label_map.items():
        for i in range(1, 6):  # 5 samples per movement
            file_path = os.path.join(folder, f'{movement}_{i}.npy')
            matrix = np.load(file_path)  # Shape (114, 100)
            data.append(matrix)
            labels.append(np.full(matrix.shape[1], label))
    X_train = np.hstack(data)
    Labels = np.concatenate(labels)
    return np.array(X_train), np.array(Labels)

X_train, L_train = load_data(train_folder)
print(np.shape(X_train))
print(X_train)
print(L_train)

pca = PCA()
X_train_pca = pca.fit(X_train.T)
PC1 = X_train_pca.components_[0, :]
PC2 = X_train_pca.components_[1, :]
PC3 = X_train_pca.components_[2, :]

print(np.mean(X_train_pca.components_))

# Plot 2D PCA projection
plt.figure()
movement = [0, 1, 2]
color = ['r', 'g', 'b']
for i in range(3):
    plt.scatter(np.dot(PC1, X_train[:, i*500:(i+1)*500]), np.dot(PC2, X_train[:, i*500:(i+1)*500]), 
    label=movement[i], c=color[i])
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("2D PCA Projection")
plt.legend()
plt.show()

# Plot 3D PCA projection
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
movement = [0, 1, 2]
color = ['r', 'g', 'b']
for i in range(3):
    ax.scatter(np.dot(PC1, X_train[:, i*500:(i+1)*500]), np.dot(PC2, X_train[:, i*500:(i+1)*500]),
               np.dot(PC3, X_train[:, i*500:(i+1)*500]),
    label=movement[i], c=color[i])
ax.set_xlabel("PC1")
ax.set_ylabel("PC2")
ax.set_zlabel("PC3")
ax.set_title("3D PCA Projection")
ax.legend()
plt.show()

# Compute centroids
def compute_centroids(X_pca_k, L_train):
    centroids = {}
    for movement in [0, 1, 2]:
        indices = np.where(L_train == movement)[0]
        centroids[movement] = np.mean(X_pca_k[:, indices], axis=1)
    return centroids


# Test classifier accuracy for different k values
k_values = range(1, 144)
print(np.shape(k_values))
accuracies = []

for k in k_values:
    X_pca_k = np.dot(X_train_pca.components_[:,:k].T, X_train)
    centroids = compute_centroids(X_pca_k, L_train)
    trained_labels = []
    for i in range(np.shape(X_train)[1]):
        sample = np.dot(X_train_pca.components_[:,:k].T, X_train[:, i])
        distances = [np.linalg.norm(sample - centroids[label]) for label in centroids]
        predicted_label = np.argmin(distances)
        trained_labels.append(predicted_label)
    trained_labels = np.array(trained_labels)

    # Compute accuracy
    accuracy = accuracy_score(L_train, trained_labels)
    accuracies.append(accuracy)
    # print(f"Accuracy for k={k}: {accuracy:.2f}")
print(np.shape(accuracies))
# Analyze the optimal k based on accuracy results
optimal_k = k_values[np.argmax(accuracies)]
print(f"Optimal k for highest accuracy: {optimal_k}\nAccuracy: {accuracies[optimal_k-1]}")

# Plot accuracy results
plt.figure()
plt.plot(k_values, [accuracies[k-1] for k in k_values], marker='o', linestyle='-', color='purple')
plt.xlabel("Number of PCA Modes (k)")
plt.ylabel("Classifier Accuracy")
plt.title("Accuracy vs. Number of PCA Modes")
plt.grid()
plt.show()

def load_test_data(folder):
    test_data = []
    test_labels = []
    label_map = {'walking': 0, 'jumping': 1, 'running': 2}
    for movement, label in label_map.items():
        file_path = os.path.join(folder, f'{movement}_1t.npy')
        matrix = np.load(file_path)  # Shape (144, 100)
        test_data.append(matrix)
        test_labels.append(np.full(matrix.shape[1], label))
    X_test = np.hstack(test_data)
    L_test = np.concatenate(test_labels)
    return np.array(X_test), np.array(L_test)

# Load test data
test_folder = "/Users/hjy/AMATH/AMATH 582/HW 2/hw2data/test/"
X_test, label_t = load_test_data(test_folder)
print(np.shape(X_test))

pca = PCA()
X_test_pca = pca.fit(X_test.T)

PC1t = X_test_pca.components_[0, :]
PC2t = X_test_pca.components_[1, :]
PC3t = X_test_pca.components_[2, :]

# print(np.mean(X_train_pca.components_))

# Plot 2D PCA projection
plt.figure()
movement = [0, 1, 2]
color = ['r', 'g', 'b']
for i in range(3):
    plt.scatter(np.dot(PC1t, X_test[:, i*500:(i+1)*500]), np.dot(PC2t, X_test[:, i*500:(i+1)*500]), 
    label=movement[i], c=color[i])
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("2D PCA Projection")
plt.legend()
plt.show()

# Plot 3D PCA projection
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
movement = [0, 1, 2]
color = ['r', 'g', 'b']
for i in range(3):
    ax.scatter(np.dot(PC1t, X_test[:, i*500:(i+1)*500]), np.dot(PC2t, X_test[:, i*500:(i+1)*500]),
               np.dot(PC3t, X_test[:, i*500:(i+1)*500]),
    label=movement[i], c=color[i])
ax.set_xlabel("PC1")
ax.set_ylabel("PC2")
ax.set_zlabel("PC3")
ax.set_title("3D PCA Projection")
ax.legend()
plt.show()

accuraciest = []
for k in k_values:
    X_pca_k = np.dot(X_train_pca.components_[:,:k].T, X_train)
    X_pca_test = np.dot(X_test_pca.components_[:,:k].T, X_test)
    centroids = compute_centroids(X_pca_k, L_train)
    
    test_predict = []
    for i in range(np.shape(X_test)[1]):
        sample = np.dot(X_train_pca.components_[:,:k].T, X_test[:, i])
        distances = [np.linalg.norm(sample - centroids[label]) for label in centroids]
        predicted_label = np.argmin(distances)
        test_predict.append(predicted_label)
    test_predict = np.array(test_predict)

    # Compute accuracy
    accuracy = accuracy_score(label_t, test_predict)
    accuraciest.append(accuracy)
    # print(f"Accuracy for k={k}: {accuracy:.2f}")
print(np.shape(accuracies))

# Plot accuracy results
plt.figure()
plt.plot(k_values, [accuraciest[k-1] for k in k_values], marker='o', linestyle='-', color='purple')
plt.xlabel("Number of PCA Modes (k)")
plt.ylabel("Classifier Accuracy")
plt.title("Accuracy vs. Number of PCA Modes")
plt.grid()
plt.show()