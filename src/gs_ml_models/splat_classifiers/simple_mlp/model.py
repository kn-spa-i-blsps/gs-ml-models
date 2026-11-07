"""
A small fully connected network (MLP) that takes an image and predicts four
values in [0, 1]. It exposes the API required by the Splat Classifier README:
fit / train / predict / save / load.

"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset

from gs_ml_models import TORCH_DEVICE


class Net(nn.Module):
    """MLP: image -> downscale -> flatten -> hidden layers -> 4 values in [0, 1]."""

    def __init__(self, hidden_sizes=(128, 128), pooled_size=(480, 640)) -> None:
        """
        Args:
            hidden_sizes: Number of neurons in each hidden layer.
            pooled_size: Size (height, width) the image is downscaled to
                before it enters the first linear layer.
        """
        super().__init__()
        self.pooled_size = tuple(pooled_size)

        sizes = [3 * self.pooled_size[0] * self.pooled_size[1], *hidden_sizes]

        self.hidden = nn.ModuleList(
            nn.Linear(n_in, n_out) for n_in, n_out in zip(sizes[:-1], sizes[1:])
        )
        self.out = nn.Linear(sizes[-1], 4)

    def forward(self, x: Tensor) -> Tensor:
        """
        Input: shape (B, 3, 480, 640), values in [0, 1].
        Output: shape (B, 4), values in [0, 1].
        """
        x = F.adaptive_avg_pool2d(x, self.pooled_size)
        x = torch.flatten(x, start_dim=1)
        for layer in self.hidden:
            x = F.relu(layer(x))

        return torch.sigmoid(self.out(x))


class SimpleMLPClassifier:
    """
    Wrapper exposing the API from the README (fit / train / predict / save / load).
    """

    def __init__(
            self,
            hidden_sizes=(128, 128),
            pooled_size=(480, 640),
            lr: float = 1e-3,
            epochs: int = 5,
            batch_size: int = 16,
            device=TORCH_DEVICE,
    ) -> None:
        """
        Args:
            hidden_sizes: Number of neurons in each hidden layer.
            pooled_size: Size (height, width) images are downscaled to.
            lr: Learning rate of the optimizer.
            epochs: Number of passes over the whole dataset per train() call.
            batch_size: Number of images processed at once.
            device: Device to run on (CPU or CUDA).
        """
        self.hidden_sizes = tuple(hidden_sizes)
        self.pooled_size = tuple(pooled_size)
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.device = device
        self._reset()

    def _reset(self) -> None:
        """Create a fresh network (random weights) and a new optimizer for it."""
        self.net = Net(self.hidden_sizes, self.pooled_size).to(self.device)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.lr)

    def fit(self, X, y) -> None:
        """Train from scratch (new weights and a new optimizer).

        Args:
            X: Images, shape (B, 3, 480, 640), values in [0, 1].
            y: Labels, shape (B, 4), values in [0, 1].
        """
        self._reset()
        self.train(X, y)

    def train(self, X, y) -> None:
        """Continue training without resetting the weights or the optimizer.

        Args:
            X: Images, shape (B, 3, 480, 640), values in [0, 1].
            y: Labels, shape (B, 4), values in [0, 1].
        """
        dataset = TensorDataset(
            torch.as_tensor(X, dtype=torch.float32),
            torch.as_tensor(y, dtype=torch.float32),
        )
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        self.net.train()
        for epoch in range(1, self.epochs + 1):
            total_loss = 0.0
            for data, target in loader:
                data, target = data.to(self.device), target.to(self.device)

                self.optimizer.zero_grad()
                output = self.net(data)
                loss = F.mse_loss(output, target)
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item() * len(data)
            print(f"Epoch {epoch}/{self.epochs}, loss: {total_loss / len(dataset):.4f}")

    def predict(self, X) -> np.ndarray:
        """Predict the four values for each image.

        Args:
            X: Images, shape (B, 3, 480, 640), values in [0, 1].

        Returns:
            NumPy array of shape (B, 4) with values in [0, 1].
        """
        X = torch.as_tensor(X, dtype=torch.float32)
        self.net.eval()
        outputs = []
        with torch.no_grad():
            for i in range(0, len(X), self.batch_size):
                batch = X[i: i + self.batch_size].to(self.device)
                outputs.append(self.net(batch).cpu())
        return torch.cat(outputs).numpy()

    def save(self, filename) -> None:
        """Save the weights and the settings needed to rebuild the model.

        The saved file can be restored with load().
        """
        torch.save(
            {
                "config": {
                    "hidden_sizes": list(self.hidden_sizes),
                    "pooled_size": list(self.pooled_size),
                    "lr": self.lr,
                    "epochs": self.epochs,
                    "batch_size": self.batch_size,
                },
                "state_dict": self.net.state_dict(),
            },
            filename,
        )

    @classmethod
    def load(cls, filename, device=TORCH_DEVICE) -> "SimpleMLPClassifier":
        """Return a model restored from a file produced by save()."""
        checkpoint = torch.load(filename, map_location=device, weights_only=True)
        model = cls(**checkpoint["config"], device=device)
        model.net.load_state_dict(checkpoint["state_dict"])
        return model
