import torch
import torch.nn as nn
import torch.optim as optim
from .logging import logger


def create_model_loss_optim(num_classes):
    model = torch.nn.Sequential()

    # Convolutional layer 1 (sees 3x224x224 image tensor)
    conv1 = torch.nn.Conv2d(in_channels=3, out_channels=16, kernel_size=(3, 3), padding=1)
    model.append(conv1)
    max_pool1 = nn.MaxPool2d(2, 2)
    model.append(torch.nn.ReLU())
    model.append(max_pool1)

    # Convolutional layer 2 (sees 16x112x112 image tensor)
    conv2 = torch.nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3), padding=1)
    model.append(conv2)
    max_pool2 = nn.MaxPool2d(2, 2)
    model.append(torch.nn.ReLU())
    model.append(max_pool2)

    # Convolutional layer 3 (sees 32x56x56 tensor)
    conv3 = torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), padding=1)
    max_pool3 = nn.MaxPool2d(2, 2)
    model.append(conv3)
    model.append(torch.nn.ReLU())
    model.append(max_pool3)

    # Flatten layer
    model.append(torch.nn.Flatten())
    model.append(nn.Dropout(0.5))

    # Linear layer (64 * 28 * 28 -> 500)
    linear1 = torch.nn.Linear(in_features=50176, out_features=500)
    model.append(linear1)
    model.append(torch.nn.ReLU())
    model.append(torch.nn.Dropout())

    # Linear layer (500 -> 23)
    output_layer = torch.nn.Linear(in_features=500, out_features=num_classes)
    model.append(output_layer)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # logger.info("Model, loss function, and optimizer created.")
    return model, loss_fn, optimizer