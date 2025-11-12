import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
import os

#filename and folder
fname= "walking_1"
train_folder = "/Users/hjy/AMATH/AMATH 582/HW 2/documents/hw2data/train/"

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

# Apply PCA
pca = PCA()
X_pca = pca.fit(X_train.T)
X_pca_ft = pca.fit_transform(X_train.T)
# number of PCA modes
explained_var = np.cumsum(pca.explained_variance_ratio_)
variance = [0.70, 0.80, 0.90, 0.95]
colors = ['r', 'g', 'c', 'purple']
num_modes_list = []
for var in variance:
    num_modes = np.argmax(explained_var >= var) + 1
    print(f"Number of PCA modes needed ({var*100}% variance): {num_modes}")
    num_modes_list.append(num_modes)

# Plot cumulative energy
plt.figure()
plt.plot(range(1, len(explained_var) + 1), explained_var, marker='o', color='blue')
for var, num_modes, color in zip(variance, num_modes_list, colors):
    plt.axhline(y=var, color=color, linestyle='--', label=f"{var*100}% variance; Number of modes: {num_modes}")
plt.xlabel("Number of PCA Spatial Modes", fontdict=dict(size=15))
plt.ylabel("Cumulative Explained Variance", fontdict=dict(size=15))
plt.title("PCA Cumulative Energy", fontdict=dict(size=20))
plt.legend(fontsize=13)
plt.savefig('PCA Cumulative Energy.pdf')
plt.show()


X_pca_2D = X_pca_ft[:, :2]
print(np.shape(X_pca_2D))
X_pca_3D = X_pca_ft[:, :3]

# Plot 2D PCA projection
plt.figure()
label = [0, 1, 2]
movements = ["Walking", "Jumping", "Running"]
color = ['r', 'g', 'b']
for i, movement in zip(label, movements):
    plt.scatter(X_pca_2D[i*500:(i+1)*500, 0], X_pca_2D[i*500:(i+1)*500, 1], 
    label=f'{movement} ({i})', c=color[i])
plt.xlabel("PC1", fontdict=dict(size=15))
plt.ylabel("PC2", fontdict=dict(size=15))
plt.title("2D PCA Projection", fontdict=dict(size=20))
plt.legend(fontsize=13)
plt.savefig('2D PCA Projection.pdf')
plt.show()

# Plot 3D PCA projection
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
for i, movement in zip(label, movements):
    ax.scatter(X_pca_3D[i*500:(i+1)*500, 0], X_pca_3D[i*500:(i+1)*500, 1], X_pca_3D[i*500:(i+1)*500, 2],
    label=f'{movement} ({i})', c=color[i])
ax.set_xlabel("PC1", fontdict=dict(size=15))
ax.set_ylabel("PC2", fontdict=dict(size=15))
ax.set_zlabel("PC3", fontdict=dict(size=15))
ax.set_title("3D PCA Projection", fontdict=dict(size=20))
ax.legend(fontsize=13)
plt.savefig('3D PCA Projection.pdf')
plt.show()


# Compute centroids
def compute_centroids(X_pca_k, L_train):
    centroids = []
    for movement in [0, 1, 2]:
        indices = np.where(L_train == movement)[0]
        centroid = np.mean(X_pca_k[indices, :], axis=0)
        centroids.append(centroid)
    return np.array(centroids)

# Centriods in 3D
centroids_3D = compute_centroids(X_pca_3D, L_train)
labels = [0, 1, 2]
for label, centroid in zip(labels, centroids_3D):
    movement = ["walking", "jumping", "running"][label]
    print(f"Centroid for {movement}({label}): {centroid}")


# Test classifier accuracy for different k values
k_values = range(0, 115)
accuracies = []

for k in k_values:
    # Apply PCA
    pca_k = PCA(n_components=k)
    X_pca_k = pca_k.fit_transform(X_train.T)  # Truncate to k-modes
    centroids = compute_centroids(X_pca_k, L_train)
    
    trained_labels = []
    for sample in X_pca_k:
        distances = [np.linalg.norm(sample - centroid) for centroid in centroids]
        predicted_label = np.argmin(distances)
        trained_labels.append(predicted_label)
    trained_labels = np.array(trained_labels)

    # Compute accuracy
    accuracy = accuracy_score(L_train, trained_labels)
    accuracies.append(accuracy)

# Analyze the optimal k based on accuracy results
optimal_k = k_values[np.argmax(accuracies)]
print(f"Optimal k for highest train accuracy: {optimal_k}")
print(f"Train accuracy: {accuracies[optimal_k]}")

# Plot accuracy results
plt.figure()
plt.plot(k_values[:60], accuracies[:60], marker='o', linestyle='-', color='purple')
plt.axhline(y=accuracy, color='c', linestyle='--', linewidth=2, label=f"Train Accuracy: {accuracy:.4f}")
plt.scatter(optimal_k, accuracies[optimal_k], color='red', s=150, label="Optimal k")
plt.annotate(f" Optimal k = {optimal_k}", (optimal_k, accuracies[optimal_k]),
             textcoords="offset points", xytext=(10,-30),
             ha='left', fontsize=15, color='blue')
plt.xlabel("Number of PCA Modes (k)", fontdict=dict(size=15))
plt.ylabel("Classifier Accuracy", fontdict=dict(size=15))
plt.title("Trained Classifier Accuracy", fontdict=dict(size=20))
plt.legend(fontsize=13)
plt.grid()
plt.savefig('Trained Classifier Accuracy.pdf')
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


# Initialize results storage
test_accuracies = []
for k in k_values:
    # Fit PCA on the training and test data
    pca_k = PCA(n_components=k)
    X_train_pca_k = pca_k.fit_transform(X_train.T)  # Shape (1500, k)
    X_test_pca_k = pca_k.fit(X_train.T).transform(X_test.T)    # (300, k)
    # Compute centroids in k-PCA space
    centroids = compute_centroids(X_train_pca_k, L_train)
    
    # Predict labels for test samples
    test_predict = []
    for sample in X_test_pca_k:
        distances = [np.linalg.norm(sample - centroid) for centroid in centroids]
        predicted_label = np.argmin(distances)
        test_predict.append(predicted_label)
    test_predict = np.array(test_predict)

    # Compute accuracy on test set
    test_accuracy = accuracy_score(L_test, test_predict)
    test_accuracies.append(test_accuracy)
    

# Analyze and compare
best_k = k_values[np.argmax(test_accuracies)]
print(f"Best k for highest test accuracy: {best_k}")
print(f"Test Accuracy: {test_accuracy}")

# Compare accuracy
plt.figure()
plt.plot(k_values[:60], accuracies[:60], marker='o', linestyle='-', label='Train Accuracy', color='purple')
plt.plot(k_values[:60], test_accuracies[:60], marker='o', linestyle='--', label='Test Accuracy', color='g')
plt.axhline(y=test_accuracy, color='c', linestyle='--', linewidth=2, label=f"Test Accuracy: {test_accuracy:.4f}")
plt.axhline(y=accuracy, color='r', linestyle='--', linewidth=2, label=f"Train Accuracy: {accuracy:.4f}")
plt.xlabel("Number of PCA Modes (k)", fontdict=dict(size=15))
plt.ylabel("Accuracy", fontdict=dict(size=15))
plt.title("Train vs Test Accuracy", fontdict=dict(size=20))
plt.legend(fontsize=13)
plt.grid()
plt.savefig('Trained vs Test Accuracy.pdf')
plt.show()


# k-NN (number of neighbors)
k_neighbors = range(1, 115)
test_accuracies_knn = []

for k in k_neighbors:
    # Fit PCA on training data
    pca_k = PCA(n_components=k)
    X_train_pca_k = pca_k.fit_transform(X_train.T)  # Shape (1500, k)
    X_test_pca_k = pca_k.transform(X_test.T)

    # ----- k-NN Classifier -----
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train_pca_k, L_train)  # Train k-NN
    test_predict_knn = knn.predict(X_test_pca_k)  # Predict test labels
    test_accuracy_knn = accuracy_score(L_test, test_predict_knn)  # Compute accuracy
    test_accuracies_knn.append(test_accuracy_knn)

# Find best k values
best_k_knn = k_neighbors[np.argmax(test_accuracies_knn)]

# Plot Accuracy Comparison
plt.figure(figsize=(8, 5))
plt.plot(k_values[1:], test_accuracies[1:], marker='o', linestyle='-', color='purple', label=f'Centroid Classifier (k=1,2,...,114)')
plt.plot(k_neighbors, test_accuracies_knn, marker='o', linestyle='--', color='green', label=f'k-NN Classifier (k=1,2,...,114)')

plt.axhline(y=test_accuracy, color='r', linestyle='--', linewidth=2, label=f"Centroid Accuracy: {test_accuracy:.4f}")
plt.axhline(y=max(test_accuracies_knn), color='c', linestyle='--', linewidth=2, label=f"Best k-NN Accuracy: {max(test_accuracies_knn):.4f}")

plt.xlabel("Number of PCA Modes (k)", fontdict=dict(size=14))
plt.ylabel("Test Accuracy", fontdict=dict(size=14))
plt.title("Comparison of Centroid and k-NN Classifiers", fontdict=dict(size=16))
plt.legend(fontsize=12)
plt.grid()
plt.savefig('Comparison of Centroid and k-NN Classifiers.pdf')
plt.show()


'''
# Plot 2D PCA projection of test data
plt.figure()
movement = [0, 1, 2]
color = ['r', 'g', 'b']
for i in range(3):
    plt.scatter(X_test_pca_k[i*100:(i+1)*100, 0], X_test_pca_k[i*100:(i+1)*100, 1], 
    label=movement[i], c=color[i])
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("2D PCA Projection")
plt.legend()
plt.show()


print("\n")
print("X_train shape: ", np.shape(X_train))
print("train label shape: ", np.shape(L_train))
print("train label: ", L_train)
print("shape of train accuracies: ", np.shape(accuracies))s
print("X_test shape: ", np.shape(X_test))
print("test label: ", L_test)
print("X_pca_2D shape: ", np.shape(X_pca_2D))
print("X_pca_3D shape: ", np.shape(X_pca_3D))
print("k_values: ", np.shape(k_values))
'''
