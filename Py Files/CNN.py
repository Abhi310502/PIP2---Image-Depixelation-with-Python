import torch
import torch.nn as nn


class SimpleCNN(nn.Module):

    def __init__(
            self,
            input_channels: int,
            hidden_channels: int,
            num_hidden_layers: int,
            use_batchnormalization: bool,
            num_classes: int,
            kernel_size: int = 3,
            activation_function: nn.Module = torch.nn.ReLU()
            ):
        super().__init__()
        hidden_layers = []
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        for _ in range(num_hidden_layers):
            layer = nn.Conv2d(in_channels=input_channels, out_channels=hidden_channels, kernel_size=kernel_size,
                              padding='same', padding_mode='zeros')
            hidden_layers.append(layer)
            if use_batchnormalization:
                hidden_layers.append(nn.BatchNorm2d(hidden_channels))
            hidden_layers.append(activation_function)
            input_channels = hidden_channels
        self.hidden_layers = nn.Sequential(*hidden_layers)
        self.output_layer = nn.Conv2d(in_channels=input_channels, out_channels=1, kernel_size=kernel_size,
                              padding='same', padding_mode='zeros')

    def forward(self, input_images: torch.Tensor):
        x = input_images.to(device=self.device)
        x = self.hidden_layers(x)
        x = (self.output_layer(x))
        return x


target_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = SimpleCNN(1, 16, 5, True, 1).to(target_device)
pytorch_total_params = sum(p.numel() for p in model.parameters())
pytorch_train_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(pytorch_total_params, pytorch_train_params)