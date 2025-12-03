import torch
from torch import nn
import torch.nn.functional as F

class ObjectClassifier(nn.Module):
    def __init__(self, input_size=500, hidden_size=256, output_size=50, dropout=0.2):
        """
        A constructor that defines the network's architecture.

        Args:
            input_size (int, optional): The amount of units from the input layer. Should be equivalent to the number of features. Defaults to 500.
            hidden_size (int, optional): The amount of units from the hidden layer. Defaults to 256.
            output_size (int, optional): The amount of units in the output layer. Defaults to 50.
            dropout (float, optional): The amount of dropout applied to the hidden layers. Defaults to 0.2.
        """
        super(ObjectClassifier, self).__init__()

        # Explicitly define each layer
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.bn2 = nn.BatchNorm1d(hidden_size // 2)
        self.fc3 = nn.Linear(hidden_size // 2, hidden_size // 4)
        self.bn3 = nn.BatchNorm1d(hidden_size // 4)
        self.output = nn.Linear(hidden_size // 4, output_size)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout / 2)
        self.dropout3 = nn.Dropout(dropout / 4)

    def forward(self, x):
        """
        A method that will do a forward pass through the network using input x.

        Args:
            x (torch.Tensor): Input

        Returns:
            out (torch.Tensor): The network’s output after the final layer.
        """
        x = x.view(x.size(0), -1)  # flatten
        x = F.gelu(self.bn1(self.fc1(x)))
        x = self.dropout1(x)
        x = F.gelu(self.bn2(self.fc2(x)))
        x = self.dropout2(x)
        x = F.gelu(self.bn3(self.fc3(x)))
        x = self.dropout3(x)
        out = self.output(x)
        return out

    def initialize_weights(self, d_type="xavier"):
        """
        This method will use Xavier initialization by default, unless specified otherwise.

        Args:
            d_type (str, optional): The initialization type. Defaults to "xavier".

        Raises:
            ValueError: If we do not get xavier or kaiming as d_type, this error is raised.
        """
        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                if d_type == "xavier":
                    nn.init.xavier_uniform_(layer.weight)
                elif d_type == "kaiming":
                    nn.init.kaiming_uniform_(layer.weight, nonlinearity='relu')
                else:
                    raise ValueError("Specify either d_type='xavier' or d_type='kaiming'.")

                nn.init.zeros_(layer.bias)  # zero bias