import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import constants

def starting_train( train_dataset, val_dataset, model, hyperparameters, n_eval):
    # Use GPU
    if torch.cuda.is_available():  # Check if GPU is available
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    # Move the model to the GPU
    model = model.to(device)
    """
    Trains and evaluates a model.

    Args:
        train_dataset:   PyTorch dataset containing training data.
        val_dataset:     PyTorch dataset containing validation data.
        model:           PyTorch model to be trained.
        hyperparameters: Dictionary containing hyperparameters.
        n_eval:          Interval at which we evaluate our model.
    """

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
        for batch in tqdm(train_loader):
            # TODO: Backpropagation and gradient descent
            model.train()
            batch_inputs, batch_labels = batch
            batch_inputs = batch_inputs.to(device)
            batch_labels = batch_labels.to(device)
            predictions = model(batch_inputs)
            loss = loss_fn(predictions, batch_labels)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
    # Periodically evaluate our model + log to Tensorboard
            if step % n_eval == 0:
                # TODO:
                # Compute training loss and accuracy.
                # Log the results to Tensorboard.
                model.eval()
                # pass
                print('Training Loss: ', loss.item())

                # for data in iter(train_loader):
                batch_inputs, batch_labels = batch
                batch_inputs = batch_inputs.to(device)
                batch_labels = batch_labels.to(device)
                predictions = model(batch_inputs).argmax(axis=1)
                accuracy = 100 * compute_accuracy(predictions, batch_labels)
                print(accuracy, "%")
                # TODO:
                # Compute validation loss and accuracy.
                # Log the results to Tensorboard.
                # Don't forget to turn off gradient calculations!
                evaluate(val_loader, model, loss_fn)
                model.train()
            step += 1

        print(step)


def compute_accuracy(outputs, labels):
    """
    Computes the accuracy of a model's predictions.

    Example input:
        outputs: [0.7, 0.9, 0.3, 0.2]
        labels:  [1, 1, 0, 1]

    Example output:
        0.75
    """

    n_correct = (torch.round(outputs.float()) == labels).sum().item()
    n_total = len(outputs)
    return n_correct / n_total


def evaluate(val_loader, model, loss_fn):
    """
    Computes the loss and accuracy of a model on the validation dataset.

    TODO!
    """
    # pass
    if torch.cuda.is_available():  # Check if GPU is available
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

        # Move the model to the GPU
    model = model.to(device)
    model.eval()
    loss_fn = nn.CrossEntropyLoss()

    for batch in tqdm(val_loader):
        batch_inputs, batch_labels = batch
        batch_inputs = batch_inputs.to(device)
        batch_labels = batch_labels.to(device)
        predictions = model(batch_inputs)

        loss = loss_fn(predictions, batch_labels)
        print('Validation Loss: ', loss.item())



    for data in iter(val_loader):
        batch_inputs, batch_labels = data
        batch_inputs = batch_inputs.to(device)
        batch_labels = batch_labels.to(device)
        predictions = model(batch_inputs).argmax(axis=1)

    accuracy = 100 * compute_accuracy(predictions, batch_labels)
    print(accuracy, "%")
