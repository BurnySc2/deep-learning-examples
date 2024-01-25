# %%
# ruff: noqa: E402 F401
"""
https://www.kaggle.com/c/dogs-vs-cats/overview
"""
import shutil
from pathlib import Path
import sys

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

url = URLs.DOGS
# Download and unpack dataset
path = untar_data(url, base=Path(__file__).parent / "fastai")

model_file = Path(__file__).parent / "models/model.pkl"


# %%
# Define label function
def is_cat(image_file_name: str) -> bool:
    return image_file_name.startswith("cat")


# %%
# Define data loader
# dls = ImageDataLoaders.from_name_func(
#     path,
#     get_image_files(path),
#     valid_pct=0.2,
#     seed=42,
#     label_func=is_cat,
#     item_tfms=Resize(224),
# )

# # Display examples
# dls.show_batch(max_n=6)

# %%
# Load model if already trained
learn = None
if model_file.is_file():
    # model_file.unlink()
    logger.info("Model exists, load model")
    learn = load_learner(model_file)

if learn is None:
    # Model if it doesn't exist
    logger.info("Model does not exist, exiting")
    sys.exit(1)

# %%
# Predict result from a test image
logger.info("Predict an image")
image = get_image_files(path / "test1")[2]
result = learn.predict(image)
logger.info(f"Result for image {image} is: {result}")
# Show a preview of the image on notebook
PILImage.create(image).to_thumb(192)
