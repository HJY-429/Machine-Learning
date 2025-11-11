"""
    Template for polynomial regression
    AUTHOR Eric Eaton, Xiaoxiang Hu
"""

from typing import Tuple

import numpy as np

from utils import problem


class PolynomialRegression:
    @problem.tag("hw1-A", start_line=5)
    def __init__(self, degree: int = 1, reg_lambda: float = 1e-8):
        """Constructor
        """
        self.degree: int = degree
        self.reg_lambda: float = reg_lambda
        # Fill in with matrix with the correct shape
        self.weight: np.ndarray = None  # type: ignore
        # You can add additional fields
        self.mean = None
        self.std = None

    @staticmethod
    @problem.tag("hw1-A")
    def polyfeatures(X: np.ndarray, degree: int) -> np.ndarray:
        """
        Expands the given X into an (n, degree) array of polynomial features of degree degree.

        Args:
            X (np.ndarray): Array of shape (n, 1).
            degree (int): Positive integer defining maximum power to include.

        Returns:
            np.ndarray: A (n, degree) numpy array, with each row comprising of
                X, X * X, X ** 3, ... up to the degree^th power of X.
                Note that the returned matrix will not include the zero-th power.

        """
        n = X.shape[0]
        feature_matrix = np.zeros((n, degree))
        for i in range(1, degree+1):
            feature_matrix[:, i-1] = (X**i).flatten()
        return feature_matrix

    @problem.tag("hw1-A")
    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Trains the model, and saves learned weight in self.weight

        Args:
            X (np.ndarray): Array of shape (n, 1) with observations.
            y (np.ndarray): Array of shape (n, 1) with targets.

        Note:
            You will need to apply polynomial expansion and data standardization first.
        """
        # get feature matrix from polyfeatures()
        Xfeature = self.polyfeatures(X, self.degree)

        # add x0
        n = Xfeature.shape[0]
        Xfeature = np.c_[np.ones((n, 1)), Xfeature]

        # data standardization
        if n > 1:
            self.mean = np.mean(Xfeature[:, 1:], axis=0)
            self.std = np.std(Xfeature[:, 1:], axis=0)
            self.std[self.std == 0] = 1
            Xfeature_std = Xfeature.copy()
            Xfeature_std[:, 1:] = (Xfeature[:, 1:] - self.mean) / self.std
        else:
            self.mean = np.zeros(Xfeature.shape[1] - 1)
            self.std = np.ones(Xfeature.shape[1] - 1)
            Xfeature_std = Xfeature

        # regularization
        n = Xfeature_std.shape[1]
        reg_matrix = self.reg_lambda * np.eye(n)
        reg_matrix[0, 0] = 0

        # save weight
        self.weight = np.linalg.solve(np.dot(Xfeature_std.T, Xfeature_std) + reg_matrix, np.dot(Xfeature_std.T, y))

    @problem.tag("hw1-A")
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Use the trained model to predict values for each instance in X.

        Args:
            X (np.ndarray): Array of shape (n, 1) with observations.

        Returns:
            np.ndarray: Array of shape (n, 1) with predictions.
        """
        Xfeature = self.polyfeatures(X, self.degree)

        # add x0
        n = Xfeature.shape[0]
        Xfeature = np.c_[np.ones((n, 1)), Xfeature]

        # data standardization
        Xfeature_std = Xfeature.copy()
        Xfeature_std[:, 1:] = (Xfeature_std[:, 1:] - self.mean) / self.std

        # predict with weight
        Xpredict = np.dot(Xfeature_std, self.weight)

        return Xpredict


@problem.tag("hw1-A")
def mean_squared_error(a: np.ndarray, b: np.ndarray) -> float:
    """Given two arrays: a and b, both of shape (n, 1) calculate a mean squared error.

    Args:
        a (np.ndarray): Array of shape (n, 1)
        b (np.ndarray): Array of shape (n, 1)

    Returns:
        float: mean squared error between a and b.
    """
    return float(np.mean((a-b) ** 2))


@problem.tag("hw1-A", start_line=5)
def learningCurve(
    Xtrain: np.ndarray,
    Ytrain: np.ndarray,
    Xtest: np.ndarray,
    Ytest: np.ndarray,
    reg_lambda: float,
    degree: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute learning curves.

    Args:
        Xtrain (np.ndarray): Training observations, shape: (n, 1)
        Ytrain (np.ndarray): Training targets, shape: (n, 1)
        Xtest (np.ndarray): Testing observations, shape: (n, 1)
        Ytest (np.ndarray): Testing targets, shape: (n, 1)
        reg_lambda (float): Regularization factor
        degree (int): Polynomial degree

    Returns:
        Tuple[np.ndarray, np.ndarray]: Tuple containing:
            1. errorTrain -- errorTrain[i] is the training mean squared error using model trained by Xtrain[0:(i+1)]
            2. errorTest -- errorTest[i] is the testing mean squared error using model trained by Xtrain[0:(i+1)]

    Note:
        - For errorTrain[i] only calculate error on Xtrain[0:(i+1)], since this is the data used for training.
            THIS DOES NOT APPLY TO errorTest.
        - errorTrain[0:1] and errorTest[0:1] won't actually matter, since we start displaying the learning curve at n = 2 (or higher)
    """
    n = len(Xtrain)

    errorTrain = np.zeros(n)
    errorTest = np.zeros(n)
    # Fill in errorTrain and errorTest arrays
    for i in range(1, n+1):
        Xt = Xtrain[:i]
        Yt = Ytrain[:i]
        try:
            # train model
            model = PolynomialRegression(degree, reg_lambda)
            model.fit(Xt, Yt)

            # errors
            Ypredict_train = model.predict(Xt)
            errorTrain[i-1] = mean_squared_error(Yt, Ypredict_train)

            Ypredict_test = model.predict(Xtest)
            errorTest[i-1] = mean_squared_error(Ytest, Ypredict_test)
            
        except (np.linalg.LinAlgError, ValueError) as e:
            # if singular matrix, set errors high
            errorTrain[i-1] = 1e10
            errorTest[i-1] = 1e10


    return errorTrain, errorTest
