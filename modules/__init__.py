import os
import sys
from datetime import datetime
import torch
from .logging import logger
from .data_prep import load_data, transform_data, split_data
from .training import train
from .prediction import predict
from .evaluation import score
from .create_model import create_model_loss_optim


__all__ = [
    'load_data',
    'transform_data',
    'split_data',
    'train',
    'predict',
    'score',
    'create_model_loss_optim',
]

data_dir = 'D:/WorldQuantUniversity/Projects/PlantDiseaseDetection/data/color'

def execute_pipeline():
    try:
        # Loading and transforming data
        dataset = load_data(data_dir)

        transformed_dataset, dataset_loader = transform_data(dataset)

        num_classes = len(dataset.classes)

        logger.info(f"Data loaded and transformed successfully with {num_classes} classes.")
    except Exception as e:
        logger.error(f"Error loading or transforming data: {e}")
        sys.exit(1)
    
    try:
        # splitting data
        train_dataset, val_dataset, train_loader, val_loader = split_data(transformed_dataset)
        logger.info("Data splitted into training and validation sets successfully.")

    except Exception as e:
        logger.error(f"Error splitting data: {e}")
        sys.exit(1)

    try:
        # Creating model, loss function, and optimizer
        model, loss_fn, optimizer = create_model_loss_optim(num_classes)

        logger.info("Model, loss function, and optimizer created successfully!")
    except Exception as e:
        logger.error(f"Error creating model, loss function, or optimizer: {e}")
        sys.exit(1)

    try:
        # Training for 4 epochs
        train(
            model,
            optimizer,
            loss_fn,
            train_loader,
            val_loader,
            epochs=4,
            device="cpu",
            use_train_accuracy=True,
        )

        logger.info("Training completed successfully!")
    except Exception as e:
        logger.error(f"Error during training: {e}")
        sys.exit(1)

    try:
        # Saving the trained model
        torch.save(model.state_dict(), "../artifacts/ModelPlus.pth")

        logger.info("Model saved successfully!")
    except Exception as e:
        logger.error(f"Error saving the model: {e}")
        sys.exit(1)



if __name__ == "__main__":
    try:
        start_time = datetime.now()
        logger.info(f"\nStarting the machine learning pipeline at {start_time}...\n")

        execute_pipeline()

        logger.info(f"Finished the machine learning pipeline at {datetime.now()}.\n")

        print(f"Total time taken: {datetime.now() - start_time}")
    except Exception as e:
        logger.error(f"Pipeline execution failed: {e}")
        sys.exit(1)