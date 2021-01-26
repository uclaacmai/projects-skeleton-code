import torch


class StartingDataset(torch.utils.data.Dataset):
    def __init__(self, statements, truth):
        self.statements = statements
        self.truth = truth

    def __getitem__(self, index):
        item = self.statements[index]
        label = self.truth[index]

        return item, label

    def __len__(self):
        return len(self.statements)
