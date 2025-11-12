import matplotlib.pyplot as plt
import numpy as np

from utils import load_dataset

if __name__ == "__main__":
    from polyreg import PolynomialRegression  # type: ignore
else:
    from .polyreg import PolynomialRegression

if __name__ == "__main__":
    """
        Main function to test polynomial regression
    """

    # load the data
    allData = load_dataset("polyreg")

    X = allData[:, [0]]
    y = allData[:, [1]]

    # regression with degree = d
    d = 8
    model = PolynomialRegression(degree=d, reg_lambda=0)
    model.fit(X, y)

    # output predictions
    xpoints = np.linspace(np.max(X), np.min(X), 100).reshape(-1, 1)
    ypoints = model.predict(xpoints)

    # plot curve
    plt.figure()
    plt.plot(X, y, "rx")
    plt.title(f"PolyRegression with d = {d}, $\lambda$ = 0", fontdict=dict(size=16))
    plt.plot(xpoints, ypoints, "b-")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.savefig(f"PolyRegression original.pdf")
    plt.show()

    # different regularization parameters
    lambdas = [0, 0.05, 0.2, 0.5, 1, 2]
    degree = 8

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for i, reg_lambda in enumerate(lambdas):
        model = PolynomialRegression(degree=degree, reg_lambda=reg_lambda)
        model.fit(X, y)
        ypoints = model.predict(xpoints)
        
        # Plot
        axes[i].plot(X, y, "rx", markersize=8, label='Data points')
        axes[i].plot(xpoints, ypoints, "b-", linewidth=2, label='Fitted curve')
        axes[i].set_title(f"d={degree}, $\lambda$={reg_lambda}")
        axes[i].set_xlabel("X")
        axes[i].set_ylabel("Y")
        axes[i].grid(True, alpha=0.3)
        axes[i].legend()
    plt.suptitle("PolyRegression with Different Regularization Params", fontsize=20)
    plt.tight_layout()
    plt.savefig(f"PolyRegression Different Params.pdf")
    plt.show()
