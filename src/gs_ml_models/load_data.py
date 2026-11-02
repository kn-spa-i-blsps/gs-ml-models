import numpy as np
import pandas as pd
import torch

# TODO: cpu fallback for pytorch


def load_labels_csv(path="data/splatdata.csv") -> pd.DataFrame:
    """Loads the labels file, might potentially do more later so
    it's left as a call to basic pandas right now"""
    df = pd.read_csv(path)
    return df


# TODO: reasonable way to load labels corresponding to images


# TODO: impl. Note there should be some reasonable order to the images
def load_images_np(path="data/classification_data") -> np.ndarray:
    pass


def load_images_grouped_np(path="data/classification_data") -> list[np.ndarray]:
    """Loads the images into a list of tensors"""
    return []


def load_images_grouped_torch(path="data/classification_data") -> list[torch.Tensor]:
    data = load_images_grouped_np(path)
    return [torch.from_numpy(d).to("cuda") for d in data]


def load_images_torch(path="data/classification_data"):
    data = load_images_np(path)
    return torch.from_numpy(data).to("cuda")
