if __name__ == "__main__":
    from layers import LinearLayer, ReLULayer, SigmoidLayer
    from losses import MSELossLayer
    from optimizers import SGDOptimizer
    from train import plot_model_guesses, train
else:
    from .layers import LinearLayer, ReLULayer, SigmoidLayer
    from .optimizers import SGDOptimizer
    from .losses import MSELossLayer
    from .train import plot_model_guesses, train


from typing import Any, Dict

import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from utils import load_dataset, problem

RNG = torch.Generator()
RNG.manual_seed(446)


@problem.tag("hw3-A")
def accuracy_score(model: nn.Module, dataloader: DataLoader) -> float:
    """Calculates accuracy of model on dataloader. Returns it as a fraction.

    Args:
        model (nn.Module): Model to evaluate.
        dataloader (DataLoader): Dataloader for MSE.
            Each example is a tuple consiting of (observation, target).
            Observation is a 2-d vector of floats.
            Target is also a 2-d vector of floats, but specifically with one being 1.0, while other is 0.0.
            Index of 1.0 in target corresponds to the true class.

    Returns:
        float: Vanilla python float resprenting accuracy of the model on given dataset/dataloader.
            In range [0, 1].

    Note:
        - For a single-element tensor you can use .item() to cast it to a float.
        - This is similar to CrossEntropy accuracy_score function,
            but there will be differences due to slightly different targets in dataloaders.
    """

    model.eval()
    correct = 0
    total = 0.0

    with torch.no_grad():
        for x, y in dataloader:
            predict = model(x)
            class_pred = torch.argmax(predict, dim=1)
            class_true = torch.argmax(y, dim=1)
            correct += torch.sum((class_pred == class_true)).item()
            total += len(x)
    
    accuracy = correct / total
    return accuracy


@problem.tag("hw3-A")
def mse_parameter_search(
    dataset_train: TensorDataset, dataset_val: TensorDataset
) -> Dict[str, Any]:
    """
    Main subroutine of the MSE problem.
    It's goal is to perform a search over hyperparameters, and return a dictionary containing training history of models, as well as models themselves.

    Models to check (please try them in this order):
        - Linear Regression Model
        - Network with one hidden layer of size 2 and sigmoid activation function after the hidden layer
        - Network with one hidden layer of size 2 and ReLU activation function after the hidden layer
        - Network with two hidden layers (each with size 2)
            and Sigmoid, ReLU activation function after corresponding hidden layers
        - Network with two hidden layers (each with size 2)
            and ReLU, Sigmoid activation function after corresponding hidden layers
        - Network with two hidden layers (each with size 2)
            and ReLU activation function after each hidden layer

    Notes:
        - When choosing the number of epochs, consider effect of other hyperparameters on it.
            For example as learning rate gets smaller you will need more epochs to converge.

    Args:
        dataset_train (TensorDataset): Training dataset.
        dataset_val (TensorDataset): Validation dataset.

    Returns:
        Dict[str, Any]: Dictionary/Map containing history of training of all models.
            You are free to employ any structure of this dictionary, but we suggest the following:
            {
                name_of_model: {
                    "train": Per epoch losses of model on train set,
                    "val": Per epoch losses of model on validation set,
                    "model": Actual PyTorch model (type: nn.Module),
                }
            }
    """
    models = {
        "Linear": nn.Sequential(
            LinearLayer(2, 2, generator=RNG)
        ),
        "HL1_size2_Sigmoid": nn.Sequential(
            LinearLayer(2, 2, generator=RNG),
            SigmoidLayer(),
            LinearLayer(2, 2, generator=RNG)
        ),
        "HL1_size2_ReLU": nn.Sequential(
            LinearLayer(2, 2, generator=RNG),
            ReLULayer(),
            LinearLayer(2, 2, generator=RNG)
        ),
        "HL2_size2_Sigmoid_ReLU": nn.Sequential(
            LinearLayer(2, 2, generator=RNG),
            SigmoidLayer(),
            LinearLayer(2, 2, generator=RNG),
            ReLULayer(),
            LinearLayer(2, 2, generator=RNG)
        ),
        "HL2_size2_ReLU_Sigmoid": nn.Sequential(
            LinearLayer(2, 2, generator=RNG),
            ReLULayer(),
            LinearLayer(2, 2, generator=RNG),
            SigmoidLayer(),
            LinearLayer(2, 2, generator=RNG)
        ),
        "HL2_size2_ReLU_ReLU": nn.Sequential(
            LinearLayer(2, 2, generator=RNG),
            ReLULayer(),
            LinearLayer(2, 2, generator=RNG),
            ReLULayer(),
            LinearLayer(2, 2, generator=RNG)
        )
    }

    # train parameters
    batches = 32
    epochs = 150
    lr = 0.01
    train_data = DataLoader(dataset_train, batches, shuffle=True)
    val_data = DataLoader(dataset_val, batches)

    # return a dictionary as results
    results = {}

    for name, model in models.items():
        print(f"\nTraining Model {name}")
        optm = SGDOptimizer(model.parameters(), lr=lr)
        loss = MSELossLayer()

        history = train(train_data, model, loss, optm, val_data, epochs)

        results[name] = {
            "train": history["train"],
            "val": history["val"],
            "model": model,
        }
    
    return results

    


@problem.tag("hw3-A", start_line=11)
def main():
    """
    Main function of the MSE problem.
    It should:
        1. Call mse_parameter_search routine and get dictionary for each model architecture/configuration.
        2. Plot Train and Validation losses for each model all on single plot (it should be 12 lines total).
            x-axis should be epochs, y-axis should me MSE loss, REMEMBER to add legend
        3. Choose and report the best model configuration based on validation losses.
            In particular you should choose a model that achieved the lowest validation loss at ANY point during the training.
        4. Plot best model guesses on test set (using plot_model_guesses function from train file)
        5. Report accuracy of the model on test set.

    Starter code loads dataset, converts it into PyTorch Datasets, and those into DataLoaders.
    You should use these dataloaders, for the best experience with PyTorch.
    """
    (x, y), (x_val, y_val), (x_test, y_test) = load_dataset("xor")

    dataset_train = TensorDataset(torch.from_numpy(x).float(), torch.from_numpy(to_one_hot(y)))
    dataset_val = TensorDataset(
        torch.from_numpy(x_val).float(), torch.from_numpy(to_one_hot(y_val))
    )
    dataset_test = TensorDataset(
        torch.from_numpy(x_test).float(), torch.from_numpy(to_one_hot(y_test))
    )

    # 1. Call mse_parameter_search routine and get dictionary
    mse_configs = mse_parameter_search(dataset_train, dataset_val)
    
    # 2. Plot Train and Validation losses for each model all on single plot
    plt.figure(figsize=(10, 6))
    for name, config in mse_configs.items():
        plt.plot(config["train"], '-o', markersize=3, linewidth=2, label=f"{name} (Train)")
        plt.plot(config["val"], '--o', markersize=3, linewidth=2, label=f"{name} (Val)")
    plt.xlabel("epochs", fontdict=dict(size=12))
    plt.ylabel("MSE loss", fontdict=dict(size=12))
    plt.title("MSE Loss for Different Model Architectures", fontdict=dict(size=16))
    plt.legend(fontsize=10)
    plt.grid(True)
    plt.savefig("MSE Loss for Different Model Architectures.pdf")
    plt.show()

    # 3. Choose and report the best model configuration based on validation losses
    best_model = "Linear"   # default
    valloss_best = float('inf')
    for name, config in mse_configs.items():
        # print(f"Model: {name}")
        # print(f"Type of val losses: {type(config['val'][0])}")
        # print(f"First few val losses: {config['val'][:5]}")
        valloss_min = min(config["val"])
        if valloss_min < valloss_best:
            valloss_best = valloss_min
            best_model = name
    print(f"Best model: {best_model} with validation loss: {valloss_best}")

    # 4. Plot best model guesses on test set
    best_guesses = mse_configs[best_model]["model"]
    test_data = DataLoader(dataset_test, len(dataset_test))
    plot_model_guesses(test_data, best_guesses, f"Best MSE Model Guesses: {best_model}")

    # 5. Report accuracy of the model on test set
    accuracy = accuracy_score(best_guesses, test_data)
    print(f"Accuracy of the best model on test set: {accuracy:.4f}\n")




def to_one_hot(a: np.ndarray) -> np.ndarray:
    """Helper function. Converts data from categorical to one-hot encoded.

    Args:
        a (np.ndarray): Input array of integers with shape (n,).

    Returns:
        np.ndarray: Array with shape (n, c), where c is maximal element of a.
            Each element of a, has a corresponding one-hot encoded vector of length c.
    """
    r = np.zeros((len(a), 2))
    r[np.arange(len(a)), a] = 1
    return r


if __name__ == "__main__":
    main()
