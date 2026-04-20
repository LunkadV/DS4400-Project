import torch.nn as nn
import torch.nn.functional as F


SPEC_HEIGHT = 128
SPEC_WIDTH = 1296
NUM_CLASSES = 16


class CNNClassifier(nn.Module):
    """
    CNN classifier for mel spectrogram genre classification.
    Convolutional encoder followed by a fully connected classification head.
    """

    def __init__(self, num_layers=3, embedding_size=256, num_classes=16, dropout=0.3):
        super(CNNClassifier, self).__init__()

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        input_channels = 1
        for i in range(num_layers):
            output_channels = 16 * (2 ** i)
            self.convs.append(
                nn.Conv2d(input_channels, output_channels, kernel_size=3, stride=2, padding=1)
            )
            self.bns.append(nn.BatchNorm2d(output_channels))
            self.dropouts.append(nn.Dropout2d(dropout))
            input_channels = output_channels

        self.h = SPEC_HEIGHT // (2 ** num_layers)
        self.w = SPEC_WIDTH // (2 ** num_layers)
        self.flatten_size = output_channels * self.h * self.w

        self.fc = nn.Linear(self.flatten_size, embedding_size)
        self.fc_dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(embedding_size, num_classes)

    def forward(self, x):
        for conv, bn, dropout in zip(self.convs, self.bns, self.dropouts):
            x = dropout(F.relu(bn(conv(x))))
        x = x.view(x.size(0), -1)
        x = self.fc_dropout(F.relu(self.fc(x)))
        x = self.classifier(x)
        return x


def get_model(params=None):
    if params is None:
        params = {
            "num_layers": 3,
            "embedding_size": 256,
            "num_classes": 16,
            "dropout": 0.3,
        }
    return CNNClassifier(**params)
