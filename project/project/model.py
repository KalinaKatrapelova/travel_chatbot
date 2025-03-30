import torch
import torch.nn as nn


class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, dropout_rate=0.2):
        super(NeuralNet, self).__init__()
        # Increase hidden size (parameter is passed from training script)

        # First hidden layer
        self.l1 = nn.Linear(input_size, hidden_size)
        self.leaky_relu = nn.LeakyReLU(0.1)  # Leaky ReLU with 0.1 negative slope
        self.dropout1 = nn.Dropout(dropout_rate)

        # Second hidden layer (same size as first)
        self.l2 = nn.Linear(hidden_size, hidden_size)
        self.dropout2 = nn.Dropout(dropout_rate)

        # Additional hidden layer with reduced size
        self.l3 = nn.Linear(hidden_size, hidden_size // 2)
        self.dropout3 = nn.Dropout(dropout_rate)

        # Output layer
        self.l4 = nn.Linear(hidden_size // 2, num_classes)

        # Batch normalization layers for stability
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.bn2 = nn.BatchNorm1d(hidden_size)
        self.bn3 = nn.BatchNorm1d(hidden_size // 2)

    def forward(self, x):
        # Check if we're dealing with a single sample (for inference)
        is_single_sample = (x.dim() == 1)
        if is_single_sample:
            x = x.unsqueeze(0)  # Add batch dimension

        # First layer
        out = self.l1(x)
        out = self.bn1(out)
        out = self.leaky_relu(out)
        out = self.dropout1(out)

        # Second layer
        out = self.l2(out)
        out = self.bn2(out)
        out = self.leaky_relu(out)
        out = self.dropout2(out)

        # Third layer
        out = self.l3(out)
        out = self.bn3(out)
        out = self.leaky_relu(out)
        out = self.dropout3(out)

        # Output layer
        out = self.l4(out)

        if is_single_sample:
            out = out.squeeze(0)  # Remove batch dimension if it was a single sample

        return out