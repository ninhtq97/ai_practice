import torch
import torch.nn as nn
from torchsummary import summary  # For part (b) and model inspection

# Import for visualization
from torchviz import make_dot  # Added for network visualization


# Basic Convolution Block
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ConvBlock, self).__init__()
        # Structure of ConvBlock as in the image
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Skip connection
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = self.shortcut(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += identity  # Add with skip connection
        out = self.relu(out)
        return out


# DownBlock (Uses ConvBlock with stride 2 for dimension reduction)
class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=2):
        super(DownBlock, self).__init__()
        # DownBlock is also a type of ConvBlock with stride = 2 for conv1 and shortcut
        # Ensure skip connection also reduces dimensions
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        identity = self.shortcut(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += identity
        out = self.relu(out)
        return out


# ResNet-18 Architecture
class ResNet18(nn.Module):
    def __init__(self, num_classes=10):  # Assuming 10 output classes for example purposes
        super(ResNet18, self).__init__()
        self.in_channels = 64

        # Feature Extractor
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # ConvBlock and DownBlock layers
        # As in the image: 2x ConvBlock 64/64, 2x DownBlock/ConvBlock 128/128, etc.
        self.layer1 = self._make_layer(ConvBlock, 64, 2)  # 2 ConvBlock 64/64
        self.layer2 = self._make_layer(ConvBlock, 128, 2, stride=2)  # 1 DownBlock (stride=2), 1 ConvBlock 128/128
        self.layer3 = self._make_layer(ConvBlock, 256, 2, stride=2)  # 1 DownBlock (stride=2), 1 ConvBlock 256/256
        self.layer4 = self._make_layer(ConvBlock, 512, 2, stride=2)  # 1 DownBlock (stride=2), 1 ConvBlock 512/512

        # Classifier
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # Global Average Pooling
        self.dropout = nn.Dropout(0.7)  # Dropout 0.7 as in the image
        self.fc1 = nn.Linear(512, 128)  # Linear 512/128
        self.bn_fc1 = nn.BatchNorm1d(128)  # BatchNorm for the first Linear layer
        self.fc2 = nn.Linear(128, num_classes)  # Linear 128/num_classes

    def _make_layer(self, block, out_channels, num_blocks, stride=1):
        layers = []
        # If stride=2, the first block will be a DownBlock
        if stride != 1 or self.in_channels != out_channels:
            layers.append(DownBlock(self.in_channels, out_channels, stride))
        else:
            layers.append(block(self.in_channels, out_channels, stride))

        self.in_channels = out_channels
        for _ in range(1, num_blocks):
            layers.append(block(self.in_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        # Feature Extractor
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # Layers
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # Classifier
        x = self.avgpool(x)
        x = torch.flatten(x, 1)  # Flatten tensor
        x = self.dropout(x)
        x = self.fc1(x)
        x = self.bn_fc1(x)  # Apply BatchNorm
        x = self.relu(x)
        x = self.fc2(x)
        return x


if __name__ == '__main__':
    # (c) Test the model with dummy input data
    batch_size = 3
    input_channels = 3
    input_height = 224
    input_width = 224
    dummy_input = torch.randn(batch_size, input_channels, input_height, input_width)

    model = ResNet18(num_classes=10)  # Initialize model with 10 output classes

    print("ResNet-18 Model Architecture:")
    # (b) Display architecture using torchsummary
    # Check output of each layer and display summary
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    summary(model, (input_channels, input_height, input_width))

    # Run forward pass with dummy data and display output size of each layer
    print("\nChecking output of each layer with dummy data:")
    x = dummy_input
    print(f"Input: {x.shape}")

    x = model.conv1(x)
    x = model.bn1(x)
    x = model.relu(x)
    print(f"After conv1, bn1, relu: {x.shape}")
    x = model.maxpool(x)
    print(f"After maxpool: {x.shape}")

    x = model.layer1(x)
    print(f"After layer1 (2x ConvBlock 64/64): {x.shape}")
    x = model.layer2(x)
    print(f"After layer2 (DownBlock/ConvBlock 128/128): {x.shape}")
    x = model.layer3(x)
    print(f"After layer3 (DownBlock/ConvBlock 256/256): {x.shape}")
    x = model.layer4(x)
    print(f"After layer4 (DownBlock/ConvBlock 512/512): {x.shape}")

    x = model.avgpool(x)
    print(f"After avgpool: {x.shape}")
    x = torch.flatten(x, 1)
    print(f"After flatten: {x.shape}")
    x = model.dropout(x)
    print(f"After dropout: {x.shape}")
    x = model.fc1(x)
    x = model.bn_fc1(x)
    x = model.relu(x)
    print(f"After fc1, bn_fc1, relu: {x.shape}")
    x = model.fc2(x)
    print(f"Final output: {x.shape}")

    # (b) Visualize the network architecture and flow using torchviz
    # To use this, uncomment the following lines and ensure graphviz is installed
    G = make_dot(model(dummy_input), params=dict(model.named_parameters()))
    G.render("resnet18_graph", view=True)  # Saves as resnet18_graph.pdf and opens it
    print("\nNetwork graph saved as resnet18_graph.pdf")
