# %%
# ruff: noqa: E402 F401
"""
https://www.kaggle.com/c/dogs-vs-cats/overview
"""
import shutil
from pathlib import Path

import torch
from fastai.callback.schedule import fine_tune
from fastai.data.external import URLs, untar_data
from fastai.data.transforms import get_image_files
from fastai.learner import Learner, export, load_learner
from fastai.metrics import error_rate
from fastai.vision.augment import Resize
from fastai.vision.core import PILImage
from fastai.vision.data import ImageDataLoaders
from fastai.vision.learner import vision_learner
from fastai.vision.models import resnet34
from loguru import logger

logger.info(f"{torch.cuda.is_available()=}")

url = URLs.DOGS
# Download and unpack dataset
path = untar_data(url, base="/ml/data")

model_file = Path("/ml/models/model.pkl")


# %%
# Define label function
def is_cat(image_file_name: str) -> bool:
    return image_file_name.startswith("cat")


# %%
# Define data loader
dls = ImageDataLoaders.from_name_func(
    path,
    get_image_files(path),
    valid_pct=0.2,
    seed=42,
    label_func=is_cat,
    item_tfms=Resize(224),
)

# Display examples
dls.show_batch(max_n=6)

# %%
# Load model if already trained
learn = None
if model_file.is_file():
    # model_file.unlink()
    logger.info("Model exists, load model")
    learn = load_learner(model_file)

# Train model if it doesn't exist
if learn is None:
    logger.info("Model does not exist, train model")

    learn = vision_learner(dls, resnet34, metrics=error_rate)
    learn.fine_tune(1)

    logger.info("Done training model")
    # learn.save() and load_model() are only for checkpoints in the training process
    learn.export("model.pkl")
    shutil.copy("/root/.fastai/data/dogscats/model.pkl", model_file)
