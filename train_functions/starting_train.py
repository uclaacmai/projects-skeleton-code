import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.tensorboard


def starting_train(
    train_dataset, val_dataset, model, hyperparameters, n_eval, summary_path
):
    """
    Trains and evaluates a model.

    Args:
        train_dataset:   PyTorch dataset containing training data.
        val_dataset:     PyTorch dataset containing validation data.
        model:           PyTorch model to be trained.
        hyperparameters: Dictionary containing hyperparameters.
        n_eval:          Interval at which we evaluate our model.
        summary_path:    Path where Tensorboard summaries are located.
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

    # Initialize summary writer (for logging)
    writer = torch.utils.tensorboard.SummaryWriter(summary_path)

    step = 0
    model.train()
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1} of {epochs}")

        # Loop over each batch in the dataset
        for i, batch in enumerate(train_loader):
            print(f"\rIteration {i + 1} of {len(train_loader)} ...", end="")

            # TODO: Backpropagation and gradient descent
            # * Backprop should be done
            # model.train()

            # print(batch)
            batch_inputs, batch_labels = batch
            # batch_labels = [x for x in batch_labels]
            # batch_inputs = torch.tensor([batch_inputs])
            # batch_labels = torch.tensor([batch_labels])
            # print("batch_labels is:",batch_labels)
            optimizer.zero_grad()
            predictions = model.forward(batch_inputs)
            print("Predictions size:",predictions.size())
            print("Batch Labels size:",batch_labels.size())
            current_loss = loss_fn(predictions, batch_labels)
            current_loss.backward()
            optimizer.step()

            step +=1
            # Periodically evaluate our model + log to Tensorboard
            if step % n_eval == 0:
                # TODO:
                # Compute training loss and accuracy.
                # Log the results to Tensorboard.
                writer.add_scalar("train_loss", current_loss, global_step = step)
                writer.add_scalar("train_accuracy", compute_accuracy(predictions, batch_labels), global_step = step)
                # TODO:
                # Compute validation loss and accuracy.
                # Log the results to Tensorboard.
                # Don't forget to turn off gradient calculations!
                eval_accuracy, eval_loss = evaluate(val_loader, model, loss_fn)
                writer.add_scalar("eval_accuracy",eval_accuracy,global_step=step)
                writer.add_scalar("eval_loss",eval_loss, global_step=step)
            # step += 1

        print()


def compute_accuracy(outputs, labels):
    """
    Computes the accuracy of a model's predictions.

    Example input:
        outputs: [0.7, 0.9, 0.3, 0.2]
        labels:  [1, 1, 0, 1]

    Example output:
        0.75
    """  
    n_correct = (torch.argmax(outputs) == labels).sum().item()
    n_total = len(outputs)
    return n_correct / n_total


def evaluate(val_loader, model, loss_fn):
    """
    Computes the loss and accuracy of a model on the validation dataset.

    TODO!
    """
    model.eval()
    correct = 0
    total = 0
    for i, batch in enumerate(val_loader):
        # batch_inputs, batch_labels = batch_inputs.to(device), batch_labels.to(device)
        batch_inputs, batch_labels = batch
        batch_inputs = torch.squeeze(batch_inputs)
        print("Batch inputs is ",batch_inputs.size())
        predictions = model.forward(batch_inputs) #16x10, axis 0 is batch size, axis 1 is output dim from the model
        # predictions = model.forward(batch_inputs).argmax(axis=1)
        loss = loss_fn(predictions, batch_labels)
        total += len(batch_labels)
        correct += (torch.argmax(predictions) == batch_labels).sum().item()
    print(100*correct/total,"%")
    return (100*correct/total), loss
