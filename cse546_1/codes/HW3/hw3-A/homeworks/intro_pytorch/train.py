from typing import Dict, List, Optional

import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import nn, optim
from torch.utils.data import DataLoader

from utils import problem


@problem.tag("hw3-A")
def train(
    train_loader: DataLoader,
    model: nn.Module,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    val_loader: Optional[DataLoader] = None,
    epochs: int = 100,
) -> Dict[str, List[float]]:
    """Performs training of a provided model and provided dataset.

    Args:
        train_loader (DataLoader): DataLoader for training set.
        model (nn.Module): Model to train.
        criterion (nn.Module): Callable instance of loss function, that can be used to calculate loss for each batch.
        optimizer (optim.Optimizer): Optimizer used for updating parameters of the model.
        val_loader (Optional[DataLoader], optional): DataLoader for validation set.
            If defined, if should be used to calculate loss on validation set, after each epoch.
            Defaults to None.
        epochs (int, optional): Number of epochs (passes through dataset/dataloader) to train for.
            Defaults to 100.

    Returns:
        Dict[str, List[float]]: Dictionary with history of training.
            It should have have two keys: "train" and "val",
            each pointing to a list of floats representing loss at each epoch for corresponding dataset.
            If val_loader is undefined, "val" can point at an empty list.

    Note:
        - Calculating training loss might expensive if you do it seperately from training a model.
            Using a running loss approach is advised.
            In this case you will just use the loss that you called .backward() on add sum them up across batches.
            Then you can divide by length of train_loader, and you will have an average loss for each batch.
        - You will be iterating over multiple models in main function.
            Make sure the optimizer is defined for proper model.
        - Make use of pytorch documentation: https://pytorch.org/docs/stable/index.html
            You might find some examples/tutorials useful.
            Also make sure to check out torch.no_grad function. It might be useful!
        - Make sure to load the model parameters corresponding to model with the best validation loss (if val_loader is provided).
            You might want to look into state_dict: https://pytorch.org/tutorials/beginner/saving_loading_models.html
    """
    # init history dict
    history = {"train": [], "val": []}
    val_loss_best = float('inf')

    # dict to store the best performing
    best_model = None

    for e in range(epochs):
        if e % (epochs // 10) == 0:
            print(f"Running epoch {e+1}-{e+(epochs // 10)}")
        model.train()
        train_runloss = 0.0
        batch_num = 0
        for x_batch, y_batch in train_loader:
            optimizer.zero_grad()

            # forward & backward
            y_pred = model(x_batch)
            loss = criterion(y_pred, y_batch)

            loss.backward()
            optimizer.step()

            # sum
            train_runloss += loss.item()
            batch_num += 1
        
        avg_trainloss = train_runloss / batch_num
        history["train"].append(avg_trainloss)

        if val_loader is not None:
            model.eval()
            val_runloss = 0.0
            val_num = 0

            with torch.no_grad():
                for x_val, y_val in val_loader:
                    val_pred = model(x_val)
                    val_loss = criterion(val_pred, y_val)

                    # sum
                    val_runloss += val_loss.item()
                    val_num += 1
            avg_valloss = val_runloss / val_num
            history["val"].append(avg_valloss)

            # record best performing
            if avg_valloss < val_loss_best:
                val_loss_best = avg_valloss
                best_model = model.state_dict().copy()
    if val_loader is not None and best_model is not None:
        model.load_state_dict(best_model)
    
    return history






def plot_model_guesses(
    dataloader: DataLoader, model: nn.Module, title: Optional[str] = None
):
    """Helper function!
    Given data and model plots model predictions, and groups them into:
        - True positives
        - False positives
        - True negatives
        - False negatives

    Args:
        dataloader (DataLoader): Data to plot.
        model (nn.Module): Model to make predictions.
        title (Optional[str], optional): Optional title of the plot.
            Might be useful for distinguishing between MSE and CrossEntropy.
            Defaults to None.
    """
    with torch.no_grad():
        list_xs = []
        list_ys_pred = []
        list_ys_batch = []
        for x_batch, y_batch in dataloader:
            y_pred = model(x_batch)
            list_xs.extend(x_batch.numpy())
            list_ys_batch.extend(y_batch.numpy())
            list_ys_pred.extend(torch.argmax(y_pred, dim=1).numpy())

        xs = np.array(list_xs)
        ys_pred = np.array(list_ys_pred)
        ys_batch = np.array(list_ys_batch)

        # True positive
        if len(ys_batch.shape) == 2 and ys_batch.shape[1] == 2:
            # MSE fix
            ys_batch = np.argmax(ys_batch, axis=1)
        idxs = np.logical_and(ys_batch, ys_pred)
        plt.figure(figsize=(10, 6))
        plt.scatter(
            xs[idxs, 0], xs[idxs, 1], marker="o", c="green", label="True Positive"
        )
        # False positive
        idxs = np.logical_and(1 - ys_batch, ys_pred)
        plt.scatter(
            xs[idxs, 0], xs[idxs, 1], marker="o", c="red", label="False Positive"
        )
        # True negative
        idxs = np.logical_and(1 - ys_batch, 1 - ys_pred)
        plt.scatter(
            xs[idxs, 0], xs[idxs, 1], marker="x", c="green", label="True Negative"
        )
        # False negative
        idxs = np.logical_and(ys_batch, 1 - ys_pred)
        plt.scatter(
            xs[idxs, 0], xs[idxs, 1], marker="x", c="red", label="False Negative"
        )

        if title:
            plt.title(title, fontdict=dict(size=16))
        plt.xlabel("x0", fontdict=dict(size=12))
        plt.ylabel("x1", fontdict=dict(size=12))
        plt.legend(fontsize=12)
        plt.savefig(f"{title}.pdf")
        plt.show()
