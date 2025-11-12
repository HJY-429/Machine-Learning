import numpy as np
import struct
import time
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score


def load_data(filename): 
    with open(filename,'rb') as f:
        _, size = struct.unpack(">II", f.read(8))
        nrows, ncols = struct.unpack(">II", f.read(8))
        data = np.fromfile(f, dtype=np.dtype(np.uint8).newbyteorder('>'))
        data_T = np.transpose(data.reshape((size, nrows*ncols)))
    return data_T

def load_label(filename):
    with open(filename,'rb') as f:
        _, size = struct.unpack(">II", f.read(8))
        data = np.fromfile(f, dtype=np.dtype(np.uint8).newbyteorder('>'))
        labels = data.reshape((size,)) 
    return labels

def plot_digits(XX, N, title):  #(784, :)
    fig, ax = plt.subplots(N, N, figsize=(8, 8))
    for i in range(N):
      for j in range(N):
        ax[i,j].imshow(XX[:,(N)*i+j].reshape((28, 28)), cmap="Greys")
        ax[i,j].axis("off")
    fig.suptitle(title, fontsize=24)
    plt.savefig(f'{title}.pdf')
    plt.show()

Xtraindata = load_data(filename='data/train-images.idx3-ubyte')
Xtestdata = load_data(filename='data/t10k-images.idx3-ubyte')
ytrainlabels = load_label(filename='data/train-labels.idx1-ubyte')
ytestlabels = load_label(filename='data/t10k-labels.idx1-ubyte')

plot_digits(Xtraindata, 8, "First 64 Training Images" )
# traindata_imgs =  np.transpose(Xtraindata).reshape((60000,28,28))

print(Xtraindata.shape)
print(ytrainlabels.shape)
print(Xtestdata.shape)
print(ytestlabels.shape)

# ____________________________________________________

# Apply PCA
pca_16 = PCA(n_components=16)
pca = PCA()

# Plot the first 16 PC modes
X_pca_16 = pca_16.fit(Xtraindata.T)
PC_modes_16 = X_pca_16.components_.T
print(np.shape(PC_modes_16))
plot_digits(PC_modes_16, 4, "First 16 PC Modes")


X_pca = pca.fit(Xtraindata.T)
cumulative_var = np.cumsum(pca.explained_variance_ratio_)
expected_var = 0.85
k_modes = np.argmax(cumulative_var >= expected_var) + 1
print(f"Number of PCA modes needed ({expected_var*100}% variance): {k_modes}")

# Plot cumulative energy
plt.figure(figsize=(8, 5))
plt.plot(cumulative_var, label="Cumulative Energy", marker='o', color='purple')
plt.axhline(y = 0.85, color='b', linestyle='--', label="85% Variance")
plt.axvline(x = k_modes, color='g', linestyle='--', label=f"k = {k_modes}")
plt.xlabel("Number of Principal Components", fontdict=dict(size=17))
plt.ylabel("Cumulative Energy", fontdict=dict(size=17))
plt.title("Cumulative Energy of Singular Values", fontdict=dict(size=20))
plt.legend(fontsize=17)
plt.savefig('Cumulative Energy of Singular Values.pdf')
plt.show()


# Reconstruct images using the first k principal components
pca_k = PCA(n_components=k_modes)
X_train_pcaft = pca_k.fit_transform(Xtraindata.T)
X_inv = pca_k.inverse_transform(X_train_pcaft)

# Transpose (784, num_samples)
X_inv_T = X_inv.T

# Plot reconstructed images
plot_digits(X_inv_T, 8, f"Reconstructed Images Using {k_modes} PC Modes")


def subdigits(X_train, y_train, X_test, y_test, dig1, dig2):
   # Select train subset
   subdigits1 = (y_train == dig1) | (y_train == dig2)
   X_subtrain = X_train[:, subdigits1]
   y_subtrain = y_train[subdigits1]
   # Select test subset
   subdigits2 = (y_test == dig1) | (y_test == dig2)
   X_subtest = X_test[:, subdigits2]
   y_subtest = y_test[subdigits2]

   return X_subtrain, y_subtrain, X_subtest, y_subtest

# Select digits
X_train18, y_train18, X_test18, y_test18 = subdigits(Xtraindata, ytrainlabels, Xtestdata, ytestlabels, 1, 8)
X_train38, y_train38, X_test38, y_test38 = subdigits(Xtraindata, ytrainlabels, Xtestdata, ytestlabels, 3, 8)
X_train27, y_train27, X_test27, y_test27 = subdigits(Xtraindata, ytrainlabels, Xtestdata, ytestlabels, 2, 7)

def Ridge_Classifier_CV(X_train, y_train, X_test, y_test):
    # project subtrain data onto k-PC modes
    X_train_pca = pca_k.fit_transform(X_train.T)
    X_test_pca = pca_k.transform(X_test.T)
    
    # Classifier with Cross-Validation
    ridge = RidgeClassifier(alpha=1.0)
    # alphas_val = np.linspace(0.01, 50, 200)
    # ridge_cv = RidgeCV(alphas=alphas_val, store_cv_results=True)
    ridge.fit(X_train_pca, y_train)

    # Evaluate on the subdata (train and test)
    cv_scores = cross_val_score(ridge, X_train_pca, y_train, cv=5)  # 5-fold CV
    train_scores = ridge.score(X_train_pca, y_train)
    test_scores = ridge.score(X_test_pca, y_test)
    return cv_scores, train_scores, test_scores


cv_scores18, train_scores18, test_scores18 = Ridge_Classifier_CV(X_train18, y_train18, X_test18, y_test18)
cv_scores38, train_scores38, test_scores38 = Ridge_Classifier_CV(X_train38, y_train38, X_test38, y_test38)
cv_scores27, train_scores27, test_scores27 = Ridge_Classifier_CV(X_train27, y_train27, X_test27, y_test27)

print("\n")
print(f"Cross-validation accuracy (1, 8): {np.mean(cv_scores18):.4f} ± {np.std(cv_scores18):.4f}")
print(f"\t Train Accuracy(1, 8): {train_scores18:.4f}\n \t Test accuracy (1, 8): {test_scores18:.4f}")
print(f"Cross-validation accuracy (3, 8): {np.mean(cv_scores38):.4f} ± {np.std(cv_scores38):.4f}")
print(f"\t Train Accuracy(3, 8): {train_scores38:.4f}\n \t Test accuracy (3, 8): {test_scores38:.4f}")
print(f"Cross-validation accuracy (2, 7): {np.mean(cv_scores27):.4f} ± {np.std(cv_scores27):.4f}")
print(f"\t Train Accuracy(2, 7): {train_scores27:.4f}\n \t Test accuracy (2, 7): {test_scores27:.4f}")


import seaborn as sns

X_train18_pca = pca_k.fit_transform(X_train18.T)
X_train18_pca_2D = X_train18_pca[:, :2]
X_train38_pca = pca_k.fit_transform(X_train38.T)
X_train38_pca_2D = X_train38_pca[:, :2]
X_train27_pca = pca_k.fit_transform(X_train27.T)
X_train27_pca_2D = X_train27_pca[:, :2]

# scatter data in 2D
def plot2D_PCA(X, y_train, sub1, sub2):
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=X[:, 0], y=X[:, 1], hue=y_train, palette=['red', 'blue'], alpha=0.7)
    plt.xlabel("PC 1", fontdict=dict(size=15))
    plt.ylabel("PC 2", fontdict=dict(size=15))
    plt.title(f"2D PCA Projection of Digits {sub1} and {sub2}", fontdict=dict(size=20))
    plt.legend(title="Digit", fontsize=16, title_fontsize=16)
    plt.grid(True)
    plt.savefig(f'2D PCA Projection of Digits {sub1} and {sub2}.pdf')
    plt.show()

plot2D_PCA(X_train18_pca_2D, y_train18, 1, 8)
plot2D_PCA(X_train38_pca_2D, y_train38, 3, 8)
plot2D_PCA(X_train27_pca_2D, y_train27, 2, 7)

solve_times = {}
Ridge = RidgeClassifier(alpha=1.0)
Knn = KNeighborsClassifier(n_neighbors=5)
Lda = LinearDiscriminantAnalysis()
Svm = SVC(kernel='rbf', C=10, gamma='scale')

def Classifier_CV(X_train_pca, y_train, X_test_pca, y_test, classifier):
    # Classifier with Cross-Validation
    classifier.fit(X_train_pca, y_train)
    # Evaluate on the subdata (train and test)
    cv_scores = cross_val_score(classifier, X_train_pca, y_train, cv=5)  # 5-fold CV
    train_scores = classifier.score(X_train_pca, y_train)
    test_scores = classifier.score(X_test_pca, y_test)
    return cv_scores, train_scores, test_scores

X_train_pca = pca_k.fit_transform(Xtraindata.T)
X_test_pca = pca_k.transform(Xtestdata.T)

start_ridge = time.time()
Ridge_CV, Ridge_train_score, Ridge_test_score = Classifier_CV(X_train_pca, ytrainlabels, X_test_pca, ytestlabels, Ridge)
end_ridge = time.time()

start_lda = time.time()
Lda_CV, Lda_train_score, Lda_test_score = Classifier_CV(X_train_pca, ytrainlabels, X_test_pca, ytestlabels, Lda)
end_lda = time.time()

start_knn = time.time()
Knn_CV, Knn_train_score, Knn_test_score = Classifier_CV(X_train_pca, ytrainlabels, X_test_pca, ytestlabels, Knn)
end_knn = time.time()


print("\n")
print(f"Ridge Classifier CV Accuracy: {Ridge_CV.mean():.4f} ± {Ridge_CV.std():.4f}") 
print(f"\t Train Accuracy: {Ridge_train_score:.4f}\n \t Test Accuracy: {Ridge_test_score:.4f}")
print(f"KNN Classifier CV Accuracy: {Knn_CV.mean():.4f} ± {Knn_CV.std():.4f}")
print(f"\t Train Accuracy: {Knn_train_score:.4f}\n \t Test Accuracy: {Knn_test_score:.4f}")
print(f"LDA Classifier CV Accuracy: {Lda_CV.mean():.4f} ± {Lda_CV.std():.4f}")
print(f"\t Train Accuracy: {Lda_train_score:.4f}\n \t Test Accuracy: {Lda_test_score:.4f}")

start_svm = time.time()
Svm_CV, Svm_train_score, Svm_test_score = Classifier_CV(X_train_pca, ytrainlabels, X_test_pca, ytestlabels, Svm)
end_svm = time.time()
print(f"SVM Classifier CV Accuracy: {Svm_CV.mean():.4f} ± {Svm_CV.std():.4f}")
print(f"\t Train Accuracy: {Svm_train_score:.4f}\n \t Test Accuracy: {Svm_test_score:.4f}")

solve_times['Ridge'] = end_ridge - start_ridge
solve_times['KNN'] = end_knn - start_knn
solve_times['LDA'] = end_lda - start_lda
solve_times['SVM'] = end_svm - start_svm

for method, time_taken in solve_times.items():
    print(f"{method} solve time: {time_taken:.4f} seconds")

