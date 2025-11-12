from typing import Tuple, Union

import matplotlib.pyplot as plt
import numpy as np

from utils import load_dataset, problem


def f_true(x: np.ndarray) -> np.ndarray:
    """True function, which was used to generate data.
    Should be used for plotting.

    Args:
        x (np.ndarray): A (n,) array. Input.

    Returns:
        np.ndarray: A (n,) array.
    """
    return 6 * np.sin(np.pi * x) * np.cos(4 * np.pi * x ** 2)


@problem.tag("hw3-A")
def poly_kernel(x_i: np.ndarray, x_j: np.ndarray, d: int) -> np.ndarray:
    """Polynomial kernel.

    Given two indices a and b it should calculate:
    K[a, b] = (x_i[a] * x_j[b] + 1)^d

    Args:
        x_i (np.ndarray): An (n,) array. Observations (Might be different from x_j).
        x_j (np.ndarray): An (m,) array. Observations (Might be different from x_i).
        d (int): Degree of polynomial.

    Returns:
        np.ndarray: A (n, m) matrix, where each element is as described above (see equation for K[a, b])

    Note:
        - It is crucial for this function to be vectorized, and not contain for-loops.
            It will be called a lot, and it has to be fast for reasonable run-time.
        - You might find .outer functions useful for this function.
            They apply an operation similar to xx^T (if x is a vector), but not necessarily with multiplication.
            To use it simply append .outer to function. For example: np.add.outer, np.divide.outer
    """
    return (np.outer(x_i, x_j) + 1) ** d


@problem.tag("hw3-A")
def rbf_kernel(x_i: np.ndarray, x_j: np.ndarray, gamma: float) -> np.ndarray:
    """Radial Basis Function (RBF) kernel.

    Given two indices a and b it should calculate:
    K[a, b] = exp(-gamma*(x_i[a] - x_j[b])^2)

    Args:
        x_i (np.ndarray): An (n,) array. Observations (Might be different from x_j).
        x_j (np.ndarray): An (m,) array. Observations (Might be different from x_i).
        gamma (float): Gamma parameter for RBF kernel. (Inverse of standard deviation)

    Returns:
        np.ndarray: A (n, m) matrix, where each element is as described above (see equation for K[a, b])

    Note:
        - It is crucial for this function to be vectorized, and not contain for-loops.
            It will be called a lot, and it has to be fast for reasonable run-time.
        - You might find .outer functions useful for this function.
            They apply an operation similar to xx^T (if x is a vector), but not necessarily with multiplication.
            To use it simply append .outer to function. For example: np.add.outer, np.divide.outer
    """
    square = np.subtract.outer(x_i, x_j) ** 2
    return np.exp(-gamma * square)


@problem.tag("hw3-A")
def train(
    x: np.ndarray,
    y: np.ndarray,
    kernel_function: Union[poly_kernel, rbf_kernel],  # type: ignore
    kernel_param: Union[int, float],
    _lambda: float,
) -> np.ndarray:
    """Trains and returns an alpha vector, that can be used to make predictions.

    Args:
        x (np.ndarray): Array of shape (n,). Observations.
        y (np.ndarray): Array of shape (n,). Targets.
        kernel_function (Union[poly_kernel, rbf_kernel]): Either poly_kernel or rbf_kernel functions.
        kernel_param (Union[int, float]): Gamma (if kernel_function is rbf_kernel) or d (if kernel_function is poly_kernel).
        _lambda (float): Regularization constant.

    Returns:
        np.ndarray: Array of shape (n,) containing alpha hat as described in the pdf.
    """
    n = len(x)

    K = kernel_function(x, x, kernel_param)
    # \alpha = (K + \lambda I)^{-1} y
    alpha = np.linalg.solve(K + _lambda * np.eye(n), y)
    return alpha


@problem.tag("hw3-A", start_line=1)
def cross_validation(
    x: np.ndarray,
    y: np.ndarray,
    kernel_function: Union[poly_kernel, rbf_kernel],  # type: ignore
    kernel_param: Union[int, float],
    _lambda: float,
    num_folds: int,
) -> float:
    """Performs cross validation.

    In a for loop over folds:
        1. Set current fold to be validation, and set all other folds as training set.
        2, Train a function on training set, and then get mean squared error on current fold (validation set).
    Return validation loss averaged over all folds.

    Args:
        x (np.ndarray): Array of shape (n,). Observations.
        y (np.ndarray): Array of shape (n,). Targets.
        kernel_function (Union[poly_kernel, rbf_kernel]): Either poly_kernel or rbf_kernel functions.
        kernel_param (Union[int, float]): Gamma (if kernel_function is rbf_kernel) or d (if kernel_function is poly_kernel).
        _lambda (float): Regularization constant.
        num_folds (int): Number of folds. It should be either len(x) for LOO, or 10 for 10-fold CV.

    Returns:
        float: Average loss of trained function on validation sets across all folds.
    """
    n = len(x)
    fold_size = n // num_folds
    indices = np.random.permutation(n)

    loss = 0.0
    
    for fold in range(num_folds):
        # validation
        val_start = fold * fold_size
        val_end = (val_start + fold_size) if fold < num_folds-1 else n
        val_indice = indices[val_start : val_end]
        train_indice = np.concatenate([indices[:val_start], indices[val_end:]])

        # data
        x_train, y_train = x[train_indice], y[train_indice]
        x_val, y_val = x[val_indice], y[val_indice]

        # train
        alpha = train(x_train, y_train, kernel_function, kernel_param, _lambda)

        # predict
        K_val = kernel_function(x_val, x_train, kernel_param)
        y_pred = np.dot(K_val, alpha)

        mse = np.mean((y_val - y_pred) ** 2)
        loss += mse

        return loss / num_folds
    


@problem.tag("hw3-A")
def rbf_param_search(
    x: np.ndarray, y: np.ndarray, num_folds: int
) -> Tuple[float, float]:
    """
    Parameter search for RBF kernel.

    There are two possible approaches:
        - Grid Search - Fix possible values for lambda, loop over them and record value with the lowest loss.
        - Random Search - Fix number of iterations, during each iteration sample lambda from some distribution and record value with the lowest loss.

    Args:
        x (np.ndarray): Array of shape (n,). Observations.
        y (np.ndarray): Array of shape (n,). Targets.
        num_folds (int): Number of folds. It should be len(x) for LOO.

    Returns:
        Tuple[float, float]: Tuple containing best performing lambda and gamma pair.

    Note:
        - You do not really need to search over gamma. 1 / (median(dist(x_i, x_j)^2) for all unique pairs x_i, x_j in x
            should be sufficient for this problem (where unique means i != j). That being said you are more than welcome to do so.
        - If using random search we recommend sampling lambda from distribution 10**i, where i~Unif(-5, -1)
        - If using grid search we recommend choosing possible lambdas to 10**i, where i=linspace(-5, -1)
    """
    n = len(x)
    distance = []
    for i in range(n):
        for j in range(i+1, n):
            distance.append((x[i] - x[j]) ** 2)
    gamma = float(1.0 / np.median(distance))

    # grid search
    best_lambda = 0
    best = float('inf')
    i = np.linspace(-5, -1, 50)
    lambdas = 10 ** i
    for l in lambdas:
        loss = cross_validation(x, y, rbf_kernel, gamma, l, num_folds)
        if loss < best:
            best = loss
            best_lambda = l
    
    return best_lambda, gamma




@problem.tag("hw3-A")
def poly_param_search(
    x: np.ndarray, y: np.ndarray, num_folds: int
) -> Tuple[float, int]:
    """
    Parameter search for Poly kernel.

    There are two possible approaches:
        - Grid Search - Fix possible values for lambdas and ds.
            Have nested loop over all possibilities and record value with the lowest loss.
        - Random Search - Fix number of iterations, during each iteration sample lambda, d from some distributions and record value with the lowest loss.

    Args:
        x (np.ndarray): Array of shape (n,). Observations.
        y (np.ndarray): Array of shape (n,). Targets.
        num_folds (int): Number of folds. It should be either len(x) for LOO, or 10 for 10-fold CV.

    Returns:
        Tuple[float, int]: Tuple containing best performing lambda and d pair.

    Note:
        - If using random search we recommend sampling lambda from distribution 10**i, where i~Unif(-5, -1)
            and d from distribution [5, 6, ..., 24, 25]
        - If using grid search we recommend choosing possible lambdas to 10**i, where i=linspace(-5, -1)
            and possible ds to [5, 6, ..., 24, 25]
    """

    # grid search
    best_lambda = 0
    best_d = 0
    best = float('inf') # best loss
    i = np.linspace(-5, -1, 50)
    lambdas = 10 ** i
    ds = range(5, 26)
    for l in lambdas:
        for d in ds:
            loss = cross_validation(x, y, poly_kernel, d, l, num_folds)
            if loss < best:
                best = loss
                best_lambda = l
                best_d = d
    
    return best_lambda, best_d
    

@problem.tag("hw3-A", start_line=1)
def main():
    """
    Main function of the problem

    It should:
        A. Using x_30, y_30, rbf_param_search and poly_param_search report optimal values for lambda (for rbf), gamma, lambda (for poly) and d.
            Note that x_30, y_30 has been loaded in for you. You do not need to use (x_300, y_300) or (x_1000, y_1000).
        B. For both rbf and poly kernels, train a function using x_30, y_30 and plot predictions on a fine grid

    Note:
        - In part b fine grid can be defined as np.linspace(0, 1, num=100)
        - When plotting you might find that your predictions go into hundreds, causing majority of the plot to look like a flat line.
            To avoid this call plt.ylim(-6, 6).
    """
    (x_30, y_30), (x_300, y_300), (x_1000, y_1000) = load_dataset("kernel_bootstrap")
    
    # rbf_param_search
    lambda_rbf, gamma = rbf_param_search(x_30, y_30, len(x_30))
    print("RBF search:")
    print(f"Best lambda: {lambda_rbf:.6f}\tBest gamma: {gamma:.6f}\n")

    # poly_param_search
    lambda_poly, d = poly_param_search(x_30, y_30, len(x_30))
    print("Poly search:")
    print(f"Best lambda: {lambda_poly:.6f}\tBest d: {d}\n")

    # train a function using x_30, y_30 and plot predictions
    x_grid = np.linspace(0, 1, 100)
    alpha_rbf = train(x_30, y_30, rbf_kernel, gamma, lambda_rbf)
    Ktest_rbf = rbf_kernel(x_grid, x_30, gamma)
    ypred_rbf = np.dot(Ktest_rbf, alpha_rbf)

    alpha_poly = train(x_30, y_30, poly_kernel, d, lambda_poly)
    Ktest_poly = poly_kernel(x_grid, x_30, d)
    ypred_poly = np.dot(Ktest_poly, alpha_poly)

    y_true = f_true(x_grid)

    # plots
    plt.figure()
    plt.scatter(x_30, y_30, alpha=0.7, label='Data points')
    plt.plot(x_grid, y_true, 'g-', label='True function', linewidth=2)
    plt.plot(x_grid, ypred_rbf, 'b--', label='RBF prediction', linewidth=2)
    plt.title(f'RBF Kernel ($\lambda={lambda_rbf:.6f}, \gamma={gamma:.6f}$)', fontdict=dict(size=16))
    plt.xlabel('x', fontdict=dict(size=12))
    plt.ylabel('y', fontdict=dict(size=12))
    plt.legend(fontsize=10)
    plt.ylim(-6, 6)
    plt.grid(True, alpha=0.3)
    plt.savefig("Kernel_RBF.pdf")
    plt.show()
    
    # Polynomial plot
    plt.figure()
    plt.scatter(x_30, y_30, alpha=0.7, label='Data points')
    plt.plot(x_grid, y_true, 'g-', label='True function', linewidth=2)
    plt.plot(x_grid, ypred_poly, 'b--', label='Poly prediction', linewidth=2)
    plt.title(f'Polynomial Kernel ($\lambda={lambda_poly:.6f}, d={d}$)', fontdict=dict(size=16))
    plt.xlabel('x', fontdict=dict(size=12))
    plt.ylabel('y', fontdict=dict(size=12))
    plt.legend(fontsize=10)
    plt.ylim(-6, 6)
    plt.grid(True, alpha=0.3)
    plt.savefig("Kernel_poly.pdf")
    plt.show()


if __name__ == "__main__":
    main()
