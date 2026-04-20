import torch.nn as nn


class MLP(nn.Module):
    """Feedforward neural network for genre classification."""

    def __init__(self, input_size=518, hidden_sizes=[512, 256], num_classes=16, dropout=0.3):
        super(MLP, self).__init__()
        layers = []
        prev_size = input_size
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_size = hidden_size
        layers.append(nn.Linear(prev_size, num_classes))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


def get_model(params=None):
    if params is None:
        params = {
            "input_size": 518,
            "hidden_sizes": [512, 256],
            "num_classes": 16,
            "dropout": 0.3,
        }
    return MLP(**params)
