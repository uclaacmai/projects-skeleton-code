import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.tensorboard


def starting_train(
    train_dataset, val_dataset, model, hyperparameters, n_eval, summary_path, device
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

    model.to(device)
    loss_fn.to(device)

    # Initialize summary writer (for logging)
    writer = torch.utils.tensorboard.SummaryWriter(summary_path)

    step = 0
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1} of {epochs}")

#convolutional
# for epoch in range(EPOCH):
#     for i, data in enumerate(train_dataloader):
#         input_data, labels = data

#         optimizer.zero_grad()
        
#         predictions = conv_network.forward(input_data)
#         loss = loss_fn(predictions, labels)
        
#         step += 1
#         train_summary.add_scalar("train_loss", loss, global_step = step)
        
#         loss.backward()
#         optimizer.step()
    
#     print("Epoch ", epoch, "  Loss ", loss.item())

        # Loop over each batch in the dataset
        for i, batch in enumerate(train_loader):
            print(f"\rIteration {i + 1} of {len(train_loader)} ...", end="")

            # TODO: Backpropagation and gradient descent

            img, labels = batch
            
            img, labels = img.to(device), labels.to(device)

            optimizer.zero_grad()
            
            predictions = model.forward(img)

            loss = loss_fn(predictions, labels)

            # Periodically evaluate our model + log to Tensorboard
            if step % n_eval == 0:
                # TODO:
                # Compute training loss and accuracy.
                #print(labels.shape)
                #print(predictions.shape)
                #n_correct += (predictions.argmax(axis=1) == labels).sum().item()
                #n_total += len(predictions.argmax(axis=1))
                accuracy = compute_accuracy(predictions.argmax(axis=1), labels)
                print(f"Accuracy: {accuracy}")

                # Log the results to Tensorboard.
                writer.add_scalar("train_loss", loss.item(), global_step = step)
                
                # TODO:
                # Compute validation loss and accuracy.
                # Log the results to Tensorboard.
                # Don't forget to turn off gradient calculations!
                model.eval()
                with torch.no_grad():
                    evaluate(val_loader, model, loss_fn, device)
                writer.add_scalar("validation_loss", loss.item(), global_step = step)
   
            model.train()
            step += 1
            
            loss.backward()
            optimizer.step()
            
        print("Epoch ", epoch, "Loss ", loss.item())


def compute_accuracy(outputs, labels):
    """
    Computes the accuracy of a model's predictions.

    Example input:
        outputs: [0.7, 0.9, 0.3, 0.2]
        labels:  [1, 1, 0, 1]

    Example output:
        0.75
    """

    #print(outputs.shape)
    #print(labels.shape)

    n_correct = (outputs == labels).sum().item()
    n_total = len(outputs)
    return n_correct / n_total


def evaluate(val_loader, model, loss_fn, device):
    """
    Computes the loss and accuracy sof a model on the validation dataset.
    """
    model.eval() #eval mode so network doesn't learn from test dataset
    
    n_correct = 0
    n_total = 0

    for i, data in enumerate(val_loader):
        input_data, labels = data
        input_data, labels = input_data.to(device), labels.to(device)
        predictions = model.forward(input_data)
        n_correct += (predictions.argmax(axis=1) == labels).sum().item()
        n_total += len(predictions.argmax(axis=1))        
        #accuracy = compute_accuracy(predictions.argmax(axis=1), labels)
        loss = loss_fn(predictions, labels)
        
    print(f"Validation Accuracy: {n_correct/n_total} Loss: {loss}")

