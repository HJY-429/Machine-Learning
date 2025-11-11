from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

from utils import problem


@problem.tag("hw2-A")
def step(
    X: np.ndarray, y: np.ndarray, weight: np.ndarray, bias: float, _lambda: float, eta: float
) -> Tuple[np.ndarray, float]:
    """Single step in ISTA algorithm.
    It should update every entry in weight, and then return an updated version of weight along with calculated bias on input weight!

    Args:
        X (np.ndarray): An (n x d) matrix, with n observations each with d features.
        y (np.ndarray): An (n, ) array, with n observations of targets.
        weight (np.ndarray): An (d,) array. Weight returned from the step before.
        bias (float): Bias returned from the step before.
        _lambda (float): Regularization constant. Determines when weight is updated to 0, and when to other values.
        eta (float): Step-size. Determines how far the ISTA iteration moves for each step.

    Returns:
        Tuple[np.ndarray, float]: Tuple with 2 entries. First represents updated weight vector, second represents bias.
    
    """
    # compute error
    error = np.dot(X, weight) + bias - y
    sum_error = 2 * np.sum(error)
    # compute b'
    bias_ = bias - eta * sum_error

    # compute w_k'
    sum_weight = 2 * np.dot(X.T, error)
    w = weight - eta * sum_weight
    # w_k'
    weight_ = np.zeros_like(w)

    threshold = 2 * eta * _lambda
    for k in range(len(w)):
        if w[k] < -threshold:
            weight_[k] = w[k] + threshold
        elif w[k] > threshold:
            weight_[k] = w[k] - threshold
        else:
            weight_[k] = 0

    return weight_, bias_


@problem.tag("hw2-A")
def loss(
    X: np.ndarray, y: np.ndarray, weight: np.ndarray, bias: float, _lambda: float
) -> float:
    """L-1 (Lasso) regularized SSE loss.

    Args:
        X (np.ndarray): An (n x d) matrix, with n observations each with d features.
        y (np.ndarray): An (n, ) array, with n observations of targets.
        weight (np.ndarray): An (d,) array. Currently predicted weights.
        bias (float): Currently predicted bias.
        _lambda (float): Regularization constant. Should be used along with L1 norm of weight.

    Returns:
        float: value of the loss function
    """
    # compute error
    error = np.dot(X, weight) + bias - y
    sse = np.sum(error ** 2)
    penalty = _lambda * np.sum(np.abs(weight))
    loss = sse + penalty
    return loss


@problem.tag("hw2-A", start_line=5)
def train(
    X: np.ndarray,
    y: np.ndarray,
    _lambda: float = 0.01,
    eta: float = 0.00001,
    convergence_delta: float = 1e-4,
    start_weight: np.ndarray = None,
    start_bias: float = None
) -> Tuple[np.ndarray, float]:
    """Trains a model and returns predicted weight and bias.

    Args:
        X (np.ndarray): An (n x d) matrix, with n observations each with d features.
        y (np.ndarray): An (n, ) array, with n observations of targets.
        _lambda (float): Regularization constant. Should be used for both step and loss.
        eta (float): Step size.
        convergence_delta (float, optional): Defines when to stop training algorithm.
            The smaller the value the longer algorithm will train.
            Defaults to 1e-4.
        start_weight (np.ndarray, optional): Weight for hot-starting model.
            If None, defaults to array of zeros. Defaults to None.
            It can be useful when testing for multiple values of lambda.
        start_bias (float, optional): Bias for hot-starting model.
            If None, defaults to zero. Defaults to None.
            It can be useful when testing for multiple values of lambda.

    Returns:
        Tuple[np.ndarray, float]: A tuple with first item being array of shape (d,) representing predicted weights,
            and second item being a float representing the bias.

    Note:
        - You will have to keep an old copy of weights for convergence criterion function.
            Please use `np.copy(...)` function, since numpy might sometimes copy by reference,
            instead of by value leading to bugs.
        - You will also have to keep an old copy of bias for convergence criterion function.
        - You might wonder why do we also return bias here, if we don't need it for this problem.
            There are two reasons for it:
                - Model is fully specified only with bias and weight.
                    Otherwise you would not be able to make predictions.
                    Training function that does not return a fully usable model is just weird.
                - You will use bias in next problem.
    """
    if start_weight is None:
        start_weight = np.zeros(X.shape[1])
    if start_bias is None:
        start_bias = 0.0
    old_w = None
    old_b = None
    weight = np.copy(start_weight)
    bias = start_bias

    while True:
        old_w = np.copy(weight)
        old_b = bias
        
        weight, bias = step(X, y, weight, bias, _lambda, eta)
        
        # check convergence
        if convergence_criterion(weight, old_w, bias, old_b, convergence_delta):
            break
    
    return weight, bias


@problem.tag("hw2-A")
def convergence_criterion(
    weight: np.ndarray, old_w: np.ndarray, bias: float, old_b: float, convergence_delta: float
) -> bool:
    """Function determining whether weight and bias has converged or not.
    It should calculate the maximum absolute change between weight and old_w vector, and compare it to convergence delta.
    It should also calculate the maximum absolute change between the bias and old_b, and compare it to convergence delta.

    Args:
        weight (np.ndarray): Weight from current iteration of gradient descent.
        old_w (np.ndarray): Weight from previous iteration of gradient descent.
        bias (float): Bias from current iteration of gradient descent.
        old_b (float): Bias from previous iteration of gradient descent.
        convergence_delta (float): Aggressiveness of the check.

    Returns:
        bool: False, if weight and bias has not converged yet. True otherwise.
    """
    if old_w is None or old_b is None:
        return False
        
    max_change_w = np.max(np.abs(weight - old_w))
    max_change_b = np.abs(bias - old_b)
    result = (max_change_w <= convergence_delta and max_change_b <= convergence_delta)
    
    return result


@problem.tag("hw2-A")
def main():
    """
    Use all of the functions above to make plots.
    """
    np.random.seed(42)

    n = 500
    d = 1000
    k = 100
    sigma = 1.0

    # weights w_j
    weights = np.zeros(d)
    for j in range(k):
        weights[j] = (j+1) / k
    
    # X normal distribution
    X = np.random.randn(n, d)
    X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)

    # get eps and y
    eps = np.random.normal(0, sigma, n)
    y = np.dot(X, weights) + eps

    # compute lambda
    lambda_max = 0
    for j in range(d):
        current = 2 * np.abs(np.sum(X[:, j] * (y - np.mean(y))))
        if current > lambda_max:
            lambda_max = current

    current_lambda = lambda_max
    lambdas = []
    while current_lambda > 1e-4:
        lambdas.append(current_lambda)
        current_lambda /= 2
    # lambdas = list(reversed(lambdas))

    # store results
    non_zeros = []
    FDRs = []
    TPRs = []
    current_weight, current_bias = None, None
    
    # compute non_zeros FDR TPR
    for i, _lambda in enumerate(lambdas):
        print(f"lambda = {_lambda:.8f}\t")
        weight, bias = train(X, y, _lambda, eta=1e-5, convergence_delta=1e-4, 
                             start_weight=current_weight, start_bias=current_bias)
        current_weight = np.copy(weight)
        current_bias = bias

        # count non_zeros
        not_zeros = np.sum(weight != 0)
        non_zeros.append(not_zeros)

        true_nonzero = (weights != 0)
        pred_nonzero = (weight != 0)
        
        # correct non_zeros
        correct = np.sum(pred_nonzero & true_nonzero)
        
        # incorrect non_zeros
        incorrect = np.sum(pred_nonzero & ~true_nonzero)
        
        # total non_zeros
        total_nonzero = np.sum(pred_nonzero)
        
        # FDR TPR
        if total_nonzero > 0:
            fdr = incorrect / total_nonzero
        else:
            fdr = 0
            
        tpr = correct / k
        
        FDRs.append(fdr)
        TPRs.append(tpr)

        print(f"#nonzeros={not_zeros}, FDR={fdr:.3f}, TPR={tpr:.3f}\n")

    # plot 1
    plt.figure(figsize=(10, 6))
    plt.plot(lambdas, non_zeros, 'bo-', linewidth=2, markersize=6)
    plt.xscale('log')
    plt.xlabel('lambda', fontdict=dict(size=15))
    plt.ylabel('Number of non-zeros', fontdict=dict(size=15))
    plt.title('Number of Non-zeros vs Lambda', fontdict=dict(size=20))
    plt.grid(True)
    plt.savefig('non_zeros_vs_lambda.pdf')
    plt.show()
    # plt.cla()
    # plt.clf()

    # plot 2
    plt.figure(figsize=(10, 6))
    plt.plot(FDRs, TPRs, 'ro-', linewidth=2, markersize=6)
    # plt.plot((d-k)/d, 1, 'g*', markersize=15, label='Ideal')
    # plt.legend()
    plt.xlabel('FDR', fontdict=dict(size=15))
    plt.ylabel('TPR', fontdict=dict(size=15))
    plt.title('TPR vs FDR', fontdict=dict(size=20))
    plt.grid(True)
    plt.savefig('fdr_vs_tpr.pdf')
    plt.show()
    # plt.cla()
    # plt.clf()


if __name__ == "__main__":
    main()
