# Build images
# docker build -t example_01 -f Dockerfile .
docker build -t example_01 -f Dockerfile2 .

# Update packages
# docker run --rm --name update_dependencies \
#     -v $(pwd):/ml \
#     example_01 \
#     poetry update --lock

# Train model 
docker run --gpus=all --rm --name example_01_train_model \
    -v $(pwd)/models:/ml/models \
    -v $(pwd)/data:/ml/data \
    -v $(pwd)/fastai:/root/.fastai \
    example_01 \
    python train_model.py

# Predict image 
# docker run --rm --name example_01_predict \
#     -v ./models:/ml/models \
#     -v ./data:/ml/data \
#     example_01 \
#     poetry run python predict.py
