# When taking sqrt for initialization you might want to use math package,
# since torch.sqrt requires a tensor, and math.sqrt is ok with integer
import math
from typing import List

import matplotlib.pyplot as plt
import torch
from torch.distributions import Uniform
from torch.nn import Module
from torch.nn.functional import cross_entropy, relu
from torch.nn.parameter import Parameter
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset

from utils import load_dataset, problem


class F1(Module):
    @problem.tag("hw3-A", start_line=1)
    def __init__(self, h: int, d: int, k: int):
        """Create a F1 model as described in pdf.

        Args:
            h (int): Hidden dimension.
            d (int): Input dimension/number of features.
            k (int): Output dimension/number of classes.
        """
        super().__init__()
        
        # init
        alpha0 = 1.0 / math.sqrt(d)
        alpha1 = 1.0 / math.sqrt(k)

        # W0, b0
        self.W0 = Parameter(Uniform(-alpha0, alpha0).sample((h, d)))
        self.b0 = Parameter(Uniform(-alpha0, alpha0).sample((h, )))

        # W1, b1
        self.W1 = Parameter(Uniform(-alpha1, alpha1).sample((k, h)))
        self.b1 = Parameter(Uniform(-alpha1, alpha1).sample((k, )))


    @problem.tag("hw3-A")
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Pass input through F1 model.

        It should perform operation:
        W_1(sigma(W_0*x + b_0)) + b_1

        Note that in this coding assignment, we use the same convention as previous
        assignments where a linear module is of the form xW + b. This differs from the 
        general forward pass operation defined above, which assumes the form Wx + b.
        When implementing the forward pass, make sure that the correct matrices and
        transpositions are used.

        Args:
            x (torch.Tensor): FloatTensor of shape (n, d). Input data.

        Returns:
            torch.Tensor: FloatTensor of shape (n, k). Prediction.
        """
        # x: (n, d), W0: (h, d)
        z0 = x @ self.W0.T + self.b0
        a0 = relu(z0)

        # a0: (n, h), W1: (k, h)
        out = a0 @ self.W1.T + self.b1

        return out  # (n, k)


class F2(Module):
    @problem.tag("hw3-A", start_line=1)
    def __init__(self, h0: int, h1: int, d: int, k: int):
        """Create a F2 model as described in pdf.

        Args:
            h0 (int): First hidden dimension (between first and second layer).
            h1 (int): Second hidden dimension (between second and third layer).
            d (int): Input dimension/number of features.
            k (int): Output dimension/number of classes.
        """
        super().__init__()
        
        # init
        alpha0 = 1.0 / math.sqrt(d)
        alpha1 = 1.0 / math.sqrt(h0)
        alpha2 = 1.0 / math.sqrt(h1)

        # W0, b0
        self.W0 = Parameter(Uniform(-alpha0, alpha0).sample((h0, d)))
        self.b0 = Parameter(Uniform(-alpha0, alpha0).sample((h0, )))

        # W1, b1
        self.W1 = Parameter(Uniform(-alpha1, alpha1).sample((h1, h0)))
        self.b1 = Parameter(Uniform(-alpha1, alpha1).sample((h1, )))

        # W2, b2
        self.W2 = Parameter(Uniform(-alpha2, alpha2).sample((k, h1)))
        self.b2 = Parameter(Uniform(-alpha2, alpha2).sample((k, )))


    @problem.tag("hw3-A")
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Pass input through F2 model.

        It should perform operation:
        W_2(sigma(W_1(sigma(W_0*x + b_0)) + b_1) + b_2)

        Note that in this coding assignment, we use the same convention as previous
        assignments where a linear module is of the form xW + b. This differs from the 
        general forward pass operation defined above, which assumes the form Wx + b.
        When implementing the forward pass, make sure that the correct matrices and
        transpositions are used.

        Args:
            x (torch.Tensor): FloatTensor of shape (n, d). Input data.

        Returns:
            torch.Tensor: FloatTensor of shape (n, k). Prediction.
        """
        # x: (n, d), W0: (h0, d)
        z0 = x @ self.W0.T + self.b0
        a0 = relu(z0)

        # a0: (n, h0), W1: (h1, h0)
        z1 = a0 @ self.W1.T + self.b1
        a1 = relu(z1)

        # a1: (n, h1), W1: (k, h1)
        out = a1 @ self.W2.T + self.b2

        return out  # (n, k)


@problem.tag("hw3-A")
def train(model: Module, optimizer: Adam, train_loader: DataLoader) -> List[float]:
    """
    Train a model until it reaches 99% accuracy on train set, and return list of training crossentropy losses for each epochs.

    Args:
        model (Module): Model to train. Either F1, or F2 in this problem.
        optimizer (Adam): Optimizer that will adjust parameters of the model.
        train_loader (DataLoader): DataLoader with training data.
            You can iterate over it like a list, and it will produce tuples (x, y),
            where x is FloatTensor of shape (n, d) and y is LongTensor of shape (n,).
            Note that y contains the classes as integers.

    Returns:
        List[float]: List containing average loss for each epoch.
    """
    model.train()
    losses = []

    max_iter = 30
    for i in range(max_iter):
        loss_t = 0.0
        correct = 0
        total = 0
        for x_train, y_train in train_loader:
            optimizer.zero_grad()
            out = model(x_train)
            loss = cross_entropy(out, y_train)

            loss.backward()
            optimizer.step()
            loss_t += loss.item() * x_train.size(0)

            # accuracy
            magic, pred = torch.max(out.data, 1)
            total += y_train.size(0)
            correct += (pred == y_train).sum().item()

        loss_avg = loss_t / total
        acc = correct / total
        losses.append(loss_avg)

        print(f"Epoch {len(losses)}")
        print(f"loss: {loss_avg:.4f}, accuracy: {acc*100:.2f} %\n")

        if acc > 0.99:
            break

    return losses



@problem.tag("hw3-A", start_line=5)
def main():
    """
    Main function of this problem.
    For both F1 and F2 models it should:
        1. Train a model
        2. Plot per epoch losses
        3. Report accuracy and loss on test set
        4. Report total number of parameters for each network

    Note that we provided you with code that loads MNIST and changes x's and y's to correct type of tensors.
    We strongly advise that you use torch functionality such as datasets, but as mentioned in the pdf you cannot use anything from torch.nn other than what is imported here.
    """
    (x, y), (x_test, y_test) = load_dataset("mnist")
    x = torch.from_numpy(x).float()
    y = torch.from_numpy(y).long()
    x_test = torch.from_numpy(x_test).float()
    y_test = torch.from_numpy(y_test).long()
    
    # dataset -> loader
    train_data = TensorDataset(x, y)
    test_data = TensorDataset(x_test, y_test)

    train_loader = DataLoader(train_data, 64, shuffle=True)
    test_loader = DataLoader(test_data, 64, shuffle=True)

    d = 784
    k = 10

    # F1 model
    model_f1 = F1(64, d, k)
    optm_f1 = Adam(model_f1.parameters(), 0.001)
    losses_f1 = train(model_f1, optm_f1, train_loader)

    # F1 evaluate
    model_f1.eval()
    f1_true = 0
    f1_loss = 0.0
    f1_total = 0
    with torch.no_grad():
        for x_test, y_test in test_loader:
            out = model_f1(x_test)
            loss = cross_entropy(out, y_test)
            f1_loss += loss.item() * x_test.size(0)

            # accuracy
            magic, pred = torch.max(out.data, 1)
            f1_total += y_test.size(0)
            f1_true += (pred == y_test).sum().item()

    f1_loss /= f1_total
    f1_testacc = f1_true / f1_total

    f1_params = sum(p.numel() for p in model_f1.parameters())


    # F2 model
    model_f2 = F2(32, 32, d, k)
    optm_f2 = Adam(model_f2.parameters(), 0.001)
    losses_f2 = train(model_f2, optm_f2, train_loader)

    # F2 evaluate
    model_f2.eval()
    f2_true = 0
    f2_loss = 0.0
    f2_total = 0
    with torch.no_grad():
        for x_test, y_test in test_loader:
            out = model_f2(x_test)
            loss = cross_entropy(out, y_test)
            f2_loss += loss.item() * x_test.size(0)

            # accuracy
            magic, pred = torch.max(out.data, 1)
            f2_total += y_test.size(0)
            f2_true += (pred == y_test).sum().item()

    f2_loss /= f2_total
    f2_testacc = f2_true / f2_total

    f2_params = sum(p.numel() for p in model_f2.parameters())

    # plots
    plt.figure()
    plt.plot(losses_f1, 'bo-', linewidth=2)
    plt.title('Training Loss vs Epoch (F1)', fontdict=dict(size=16))
    plt.xlabel('epoch', fontdict=dict(size=12))
    plt.ylabel('loss', fontdict=dict(size=12))
    plt.grid(True)
    plt.savefig('f1_losses.pdf')
    plt.show()

    plt.figure()
    plt.plot(losses_f2, 'bo-', linewidth=2)
    plt.title('Training Loss vs Epoch (F2)', fontdict=dict(size=16))
    plt.xlabel('epoch', fontdict=dict(size=12))
    plt.ylabel('loss', fontdict=dict(size=12))
    plt.grid(True)
    plt.savefig('f2_losses.pdf')
    plt.show()


    # results
    print("---------------------------------")
    print(f"F1 model test results:")
    print(f"loss: {f1_loss:.4f}, accuracy: {f1_testacc*100:.2f} %")
    print(f"total parameters: {f1_params}\n")

    print(f"F2 model test results:")
    print(f"loss: {f2_loss:.4f}, accuracy: {f2_testacc*100:.2f} %")
    print(f"total parameters: {f2_params}\n")


if __name__ == "__main__":
    main()
