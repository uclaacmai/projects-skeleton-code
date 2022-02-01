import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter


def evaluate(val_loader, model, loss_fn, dimensions):
    model.eval()
    total, correct = 0, 0

    for data in iter(val_loader):
        inputs, labels = data
        inputs = torch.reshape(inputs, dimensions)
        predictions = model(inputs).argmax(axis=1)
        total += len(labels)
        correct += (predictions==labels).sum().item()
    
    print(f"{100 * correct / total}%")



def starting_train(train_dataset, val_dataset, dimensions, model, hyperparameters, n_eval):
    """
    Trains and evaluates a model.

    Args:
        train_dataset:   PyTorch dataset containing training data.
        val_dataset:     PyTorch dataset containing validation data.
        model:           PyTorch model to be trained.
        hyperparameters: Dictionary containing hyperparameters.
        n_eval:          Interval at which we evaluate our model.
    """
    writer = SummaryWriter()
    # Get keyword arguments
    batch_size, epochs = hyperparameters["batch_size"], hyperparameters["epochs"]

    # Initialize dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=True
    )

    # Initalize optimizer (for gradient descent) and loss function
    optimizer = optim.Adam(model.parameters())
    loss_fn = nn.CrossEntropyLoss()

    step = 0
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1} of {epochs}")

        # Loop over each batch in the dataset
        for data in tqdm(train_loader):
            # TODO: Backpropagation and gradient descent determine which lines of code to use
            # outputs = model(images)

            # loss = loss_fn(outputs,labels)
            # loss.backward()
            # optimizer.step()
            # optimizer.zero_grad()

            batch_inputs, batch_labels = data

            batch_inputs = torch.reshape(batch_inputs, dimensions)
            optimizer.zero_grad()
            predictions = model(batch_inputs)
            current_loss = loss_fn(predictions, batch_labels)
            
            # Periodically evaluate our model + log to Tensorboard
            if step % n_eval == 0:
                # TODO:
                # Compute training loss and accuracy.
                # Log the results to Tensorboard.
                writer.add_scalar("Train Accuracy", compute_accuracy(predictions, batch_labels))
                writer.add_scalar("Train Loss", current_loss)
                # go to http://localhost:6006/ to view the Tensorboard

                # TODO:
                # Compute validation loss and accuracy.
                # Log the results to Tensorboard.
                # Don't forget to turn off gradient calculations!
                pass
                
            step += 1
            current_loss.backward()
            optimizer.step()
        
        evaluate(val_loader, model, loss_fn, dimensions)

        print()

    writer.flush()
    writer.close()


def compute_accuracy(outputs, labels):
    """
    Computes the accuracy of a model's predictions.

    Example input:
        outputs: [0.7, 0.9, 0.3, 0.2]
        labels:  [1, 1, 0, 1]

    Example output:
        0.75
    """

    n_correct = (torch.round(outputs) == labels).sum().item()
    n_total = len(outputs)
    return n_correct / n_total