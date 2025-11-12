import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
import os

def pca_svd(X, k):
    X_mean = np.mean(X, axis=0, keepdims=True)
    X_centered = X - X_mean
    U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
    # k principal components
    components = Vt[:k, :]
    # Project data onto k principal components
    X_pca = np.dot(X_centered, components.T)  # Shape (n_samples, k)
    return X_pca, components

def load_data(folder):
    data = []
    labels = []
    label_map = {'walking': 0, 'jumping': 1, 'running': 2}
    for movement, label in label_map.items():
        for i in range (1,6):
            file_path = os.path.join(folder, f'{movement}_{i}.npy')
            matrix = np.load(file_path)  # Shape (144, 100)
            data.append(matrix)
            labels.append(np.full(matrix.shape[1], label))
    X_train = np.hstack(data)
    L_train = np.concatenate(labels)
    return np.array(X_train), np.array(L_train)


#filename and folder
train_folder = "/Users/hjy/AMATH/AMATH 582/HW 2/hw2data/train/"
X_train, L_train = load_data(train_folder)

k = 3
X_train_pca, pt_components = pca_svd(X_train.T, k)


# Plot 2D PCA projection
plt.figure()
movement = [0, 1, 2]
color = ['r', 'g', 'b']
for i in range(3):
    plt.scatter(X_train_pca[i*500:(i+1)*500, 0], X_train_pca[i*500:(i+1)*500, 1], 
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
    ax.scatter(X_train_pca[i*500:(i+1)*500, 0], X_train_pca[i*500:(i+1)*500, 1], X_train_pca[i*500:(i+1)*500, 2],
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
        centroids[movement] = np.mean(X_pca_k[indices, :], axis=0)
    return centroids

# Centroids for k=3
X_pca_k, _ = pca_svd(X_train.T, 3)
centroids = compute_centroids(X_pca_k, L_train)
for label, centroid in centroids.items():
    movement = ["walking", "jumping", "running"][label]
    print(f"Centroid for {movement}({label}): {centroid}")


# Test classifier accuracy for different k values
k_values = range(0, 115)
accuracies = []

for k in k_values:
    X_train_pca_k, train_components_k = pca_svd(X_train.T, k)
    centroids = compute_centroids(X_train_pca_k, L_train)
    
    trained_labels = []
    for sample in X_train_pca_k:
        distances = [np.linalg.norm(sample - centroids[label]) for label in centroids]
        predicted_label = np.argmin(distances)
        trained_labels.append(predicted_label)
    trained_labels = np.array(trained_labels)

    # Compute accuracy
    accuracy = accuracy_score(L_train, trained_labels)
    accuracies.append(accuracy)


# Analyze the optimal k based on accuracy results
optimal_k = k_values[np.argmax(accuracies)]
print(f"Optimal k for highest accuracy: {optimal_k}\nAccuracy: {accuracies[optimal_k]}")

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
X_test, L_test = load_test_data(test_folder)
# print(np.shape(X_test))
# print(L_test)


def pca_svd_t(X, X_train, k):
    X_mean = np.mean(X_train, axis=0, keepdims=True)
    X_centered = X_train - X_mean
    X_mean_t = np.mean(X, axis=0, keepdims=True)
    X_centered_t = X - X_mean_t
    U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
    # k principal components
    components = Vt[:k, :]
    # Project data onto k principal components
    X_pca = np.dot(X_centered_t, components.T)  # Shape (n_samples, k)
    return X_pca, components

X_test_pca, pt_components = pca_svd_t(X_test.T, X_train.T, k)

test_accuracies = []
for k in k_values:
    X_train_pca_k, train_components_k = pca_svd(X_train.T, k)
    X_test_pca_k, test_components_k = pca_svd_t(X_test.T, X_train.T, k)
    # Compute centroids in k-PCA space
    centroids = compute_centroids(X_train_pca_k, L_train)
    # Predict labels for test samples
    test_predict = []
    for sample in X_test_pca_k:
        distances = [np.linalg.norm(sample - centroids[label]) for label in centroids]
        predicted_label = np.argmin(distances)
        test_predict.append(predicted_label)
    test_predict = np.array(test_predict)

    # Compute accuracy on test set
    test_accuracy = accuracy_score(L_test, test_predict)
    test_accuracies.append(test_accuracy)
    

# Analyze and compare
best_k = k_values[np.argmax(test_accuracies)]
print(f"Best k for test accuracy: {best_k}")
print(f"Test Accuracy = {test_accuracy:.4f}")

# Compare accuracy
plt.figure()
plt.plot(k_values, [accuracies[k-1] for k in k_values], marker='o', linestyle='-', label='Train Accuracy', color='purple')
plt.plot(k_values, [test_accuracies[k-1] for k in k_values], marker='s', linestyle='--', label='Test Accuracy', color='g')
plt.xlabel("Number of PCA Modes (k)")
plt.ylabel("Accuracy")
plt.title("Train vs Test Accuracy")
plt.legend()
plt.grid()
plt.show()

'''
print(\n)
print(np.shape(X_train))
print(X_train)
print(np.shape(L_train))
print(L_train)
print(np.shape(k_values))
print(np.shape(accuracies))
# Plot 2D PCA projection (Test Data)
plt.figure()
movement = [0, 1, 2]
color = ['r', 'g', 'b']
for i in range(3):
    plt.scatter(X_test_pca[i*100:(i+1)*100, 0], X_test_pca[i*100:(i+1)*100, 1], 
    label=movement[i], c=color[i])
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("2D PCA Projection (Test Data)")
plt.legend()
plt.show()

# Plot 3D PCA projection (Test Data)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
movement = [0, 1, 2]
color = ['r', 'g', 'b']
for i in range(3):
    ax.scatter(X_test_pca[i*100:(i+1)*100, 0], X_test_pca[i*100:(i+1)*100, 1], X_test_pca[i*100:(i+1)*100, 2],
    label=movement[i], c=color[i])
ax.set_xlabel("PC1")
ax.set_ylabel("PC2")
ax.set_zlabel("PC3")
ax.set_title("3D PCA Projection (Test Data)")
ax.legend()
plt.show()
'''

