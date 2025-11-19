import os
import torch
from tqdm.notebook import tqdm
from .logging import logger
from .evaluation import score


def train_epoch(model, optimizer, loss_fn, data_loader, device="cpu"):
    training_loss = 0.0
    model.train()

    # Iterating over all batches in the training set to complete one epoch
    for inputs, targets in tqdm(data_loader, desc="Training", leave=False):
        optimizer.zero_grad()
        inputs = inputs.to(device)
        targets = targets.to(device)

        output = model(inputs)
        loss = loss_fn(output, targets)

        loss.backward()
        optimizer.step()
        training_loss += loss.data.item() * inputs.size(0)

    return training_loss / len(data_loader.dataset)


def train(
    model,
    optimizer,
    loss_fn,
    train_loader,
    val_loader,
    epochs=5,
    device="cpu",
    use_train_accuracy=True,
    checkpoint_dir="D:/WorldQuantUniversity/Projects/PlantDiseaseDetection/artifacts/checkpoints",
):
    # model progress over epochs
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []

    # Ensuring the checkpoint directory exists
    os.makedirs(checkpoint_dir, exist_ok=True)
    logger.info(f"Checkpoint directory available at: {checkpoint_dir}")

    for epoch in range(1, epochs + 1):
        # Training one epoch
        training_loss = train_epoch(model, optimizer, loss_fn, train_loader, device)

        # Evaluating training results
        if use_train_accuracy:
            train_loss, train_accuracy = score(model, train_loader, loss_fn, device)
        else:
            train_loss = training_loss
            train_accuracy = 0
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)

        # Testing on validation set
        validation_loss, validation_accuracy = score(model, val_loader, loss_fn, device)
        val_losses.append(validation_loss)
        val_accuracies.append(validation_accuracy)

        print(f"Epoch: {epoch}")
        if use_train_accuracy:
            print(f"    Training accuracy: {train_accuracy:.2f}")
        print(f"    Validation accuracy: {validation_accuracy:.2f}")
        
        logger.info(
            f"\nEpoch {epoch}: Train Loss={train_loss:.4f},\n"
            f"Train Acc={train_accuracy:.4f},\n"
            f"Val Loss={validation_loss:.4f},\n"
            f"Val Acc={validation_accuracy:.4f}"
        )

        # Checkpointing
        try:
            checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch}.pth")
            
            # Saving the model state, optimizer state, and current metrics
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': validation_loss,
                'accuracy': validation_accuracy,
            }, checkpoint_path)
            
            logger.info(f"Checkpoint saved successfully for Epoch {epoch} to {checkpoint_path}")

        except Exception as e:
            logger.error(f"Error saving checkpoint for Epoch {epoch}: {e}")

    return train_losses, val_losses, train_accuracies, val_accuracies