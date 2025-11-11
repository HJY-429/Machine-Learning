if __name__ == "__main__":
    from ISTA import train  # type: ignore
else:
    from .ISTA import train

import matplotlib.pyplot as plt
import numpy as np

from utils import load_dataset, problem


@problem.tag("hw2-A", start_line=3)
def main():
    # df_train and df_test are pandas dataframes.
    # Make sure you split them into observations and targets
    df_train, df_test = load_dataset("crime")

    # split into features and target
    y_train = df_train["ViolentCrimesPerPop"].values
    X_train = df_train.drop("ViolentCrimesPerPop", axis = 1).values
    y_test = df_test["ViolentCrimesPerPop"].values
    X_test = df_test.drop("ViolentCrimesPerPop", axis = 1).values
    feature_names = df_train.drop("ViolentCrimesPerPop", axis=1).columns.tolist()

    # standardize features
    X_train = (X_train - np.mean(X_train, axis=0)) / np.std(X_train, axis=0)
    X_test = (X_test - np.mean(X_test, axis=0)) / np.std(X_test, axis=0)

    # compute lambda_max
    lambda_max = 0
    for j in range(X_train.shape[1]):
        current = 2 * np.abs(np.sum(X_train[:, j] * (y_train - np.mean(y_train))))
        if current > lambda_max:
            lambda_max = current

    print(f"lambda_max: {lambda_max}\n")

    # regularization paths
    current_lambda = lambda_max
    lambdas = []
    while current_lambda > 0.01:
        lambdas.append(current_lambda)
        current_lambda /= 2

    lambdas = list(reversed(lambdas))
    
    # init results storage
    non_zeros = []
    errors_train = []
    errors_test = []
    weights = []

    # get variables for part d
    input_var = ['agePct12t29', 'pctWSocSec', 'pctUrban', 'agePct65up', 'householdsize']
    input_index = [feature_names.index(i) for i in input_var]
    input_paths = {path: [] for path in input_var}
    
    # train data
    current_weight, current_bias = None, None
    
    # compute non_zeros FDR TPR
    for i, _lambda in enumerate(lambdas):
        print(f"lambda = {_lambda:.8f}")
        weight, bias = train(X_train, y_train, _lambda, eta=1e-5, convergence_delta=1e-4, 
                             start_weight=current_weight, start_bias=current_bias)
        current_weight = np.copy(weight)
        current_bias = bias

        # count non_zeros
        not_zeros = np.sum(weight != 0)
        non_zeros.append(not_zeros)
        print(f"#nonzeros={not_zeros}\n")

        # compute error
        train_pred = np.dot(X_train, weight) + bias
        test_pred = np.dot(X_test, weight) + bias

        error_train = np.mean((train_pred - y_train) ** 2)
        error_test = np.mean((test_pred - y_test) ** 2)
        errors_train.append(error_train)
        errors_test.append(error_test)
        weights.append(weight)

        for var, idx in zip(input_var, input_index):
            input_paths[var].append(weight[idx])

    # plot c
    plt.figure(figsize=(10, 6))
    plt.plot(lambdas, non_zeros, 'bo-', linewidth=2, markersize=6)
    plt.xscale('log')
    plt.xlabel('lambda', fontdict=dict(size=15))
    plt.ylabel('number of non-zero weights', fontdict=dict(size=15))
    plt.title('Number of Non-zero Weights vs Lambda', fontdict=dict(size=20))
    plt.grid(True)
    plt.savefig('(c)non_zeros_vs_lambda.pdf')
    plt.show()

    # plot d
    plt.figure(figsize=(10, 6))
    for var in input_var:
        plt.plot(lambdas, input_paths[var], 'o-', linewidth=2, markersize=6, label=var)
    plt.xscale('log')
    plt.xlabel('lambda', fontdict=dict(size=15))
    plt.ylabel('coefficents', fontdict=dict(size=15))
    plt.title('Regularization Paths for Input Variables', fontdict=dict(size=20))
    plt.legend(fontsize=12)
    plt.grid(True)
    plt.savefig('(d)crime_feature_paths.pdf')
    plt.show()

    # plot e
    plt.figure(figsize=(10, 6))
    plt.plot(lambdas, errors_train, 'go-', linewidth=2, markersize=6, label='Training error')
    plt.plot(lambdas, errors_test, 'bo-', linewidth=2, markersize=6, label='Test error')    
    plt.xscale('log')
    plt.xlabel('lambda', fontdict=dict(size=15))
    plt.ylabel('train/test mean squared error', fontdict=dict(size=15))
    plt.title('Training and Test Errors for Different Lambdas', fontdict=dict(size=20))
    plt.legend(fontsize=12)
    plt.grid(True)
    plt.savefig('(e)crime_errors_vs_lambda.pdf')
    plt.show()

    # retrain lambda=30
    print(f"lambda = 30")
    weight_30, bias_30 = train(X_train, y_train, _lambda=30, eta=1e-5, convergence_delta=1e-4, 
                            start_weight=current_weight, start_bias=current_bias)
    # print(weight_30)

    # check non_zeros
    weight_30_nonzero = (weight_30 != 0)

    non_zero_features = [(feature_names[i], weight_30[i]) 
                        for i in range(len(weight_30)) if weight_30_nonzero[i]]
    
    # sort by coefficient value
    non_zero_features.sort(key=lambda x: x[1])

    print(f"\n# nonzero features: {len(non_zero_features)}")
    print(f"Most positive feature: {non_zero_features[-1]}\n")
    print(f"Most negative feature: {non_zero_features[0]}\n")


    # print(non_zero_features)

if __name__ == "__main__":
    main()
