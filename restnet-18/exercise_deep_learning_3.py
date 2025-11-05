import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm  # For progress bar


# Basic Convolution Block (unchanged from previous version)
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
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
        out += identity
        out = self.relu(out)
        return out


# DownBlock (unchanged from previous version)
class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=2):
        super(DownBlock, self).__init__()
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


# ResNet-18 Architecture (unchanged from previous version, but now with num_classes for Imagenette)
class ResNet18(nn.Module):
    def __init__(self, num_classes=10):  # Imagenette has 10 classes
        super(ResNet18, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(ConvBlock, 64, 2)
        self.layer2 = self._make_layer(ConvBlock, 128, 2, stride=2)
        self.layer3 = self._make_layer(ConvBlock, 256, 2, stride=2)
        self.layer4 = self._make_layer(ConvBlock, 512, 2, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.7)
        self.fc1 = nn.Linear(512, 128)
        self.bn_fc1 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, num_classes)

    def _make_layer(self, block, out_channels, num_blocks, stride=1):
        layers = []
        if stride != 1 or self.in_channels != out_channels:
            layers.append(DownBlock(self.in_channels, out_channels, stride))
        else:
            layers.append(block(self.in_channels, out_channels, stride))
        self.in_channels = out_channels
        for _ in range(1, num_blocks):
            layers.append(block(self.in_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc1(x)
        x = self.bn_fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


# --- Huấn luyện mô hình ---
def train_model():
    # 1. Tải dữ liệu và tiền xử lý
    # Đặt đường dẫn đến thư mục chứa Imagenette2-320
    # Ví dụ: data_path = 'path/to/imagenette2-320'
    # Đảm bảo cấu trúc thư mục là data_path/train/class_name/images.jpg
    # và data_path/val/class_name/images.jpg
    data_path = 'data/imagenette2-320'  # Cập nhật đường dẫn này nếu cần

    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize ảnh về 224x224
        transforms.ToTensor(),  # Chuyển ảnh sang Tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Chuẩn hóa ImageNet
    ])

    # Tạo dataset và dataloader
    try:
        train_dataset = datasets.ImageFolder(root=f'{data_path}/train', transform=transform)
        val_dataset = datasets.ImageFolder(root=f'{data_path}/val', transform=transform)
    except Exception as e:
        print(f"Lỗi khi tải dữ liệu: {e}")
        print(f"Đảm bảo thư mục '{data_path}' tồn tại và chứa thư mục 'train' và 'val' với cấu trúc ảnh phù hợp.")
        print("Ví dụ: '{data_path}/train/n01440764/ILSVRC2012_val_00000293.JPEG'")
        return

    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    num_classes = len(train_dataset.classes)
    model = ResNet18(num_classes=num_classes)

    # 2. Lựa chọn loss function và hàm tối ưu Adam
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)  # learning rate = 1e-3

    # Thiết lập device (GPU nếu có)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"Sử dụng thiết bị: {device}")

    # Danh sách để lưu trữ kết quả huấn luyện
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []

    # 3. Huấn luyện mô hình trong tối thiểu 10 epochs
    num_epochs = 10
    print(f"\nBắt đầu huấn luyện mô hình trong {num_epochs} epochs...")
    for epoch in range(num_epochs):
        # Huấn luyện
        model.train()  # Đặt mô hình ở chế độ huấn luyện
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        # Sử dụng tqdm để hiển thị thanh tiến trình
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs} [Train]"):
            images, labels = images.to(device), labels.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

        epoch_train_loss = running_loss / len(train_loader.dataset)
        epoch_train_acc = correct_train / total_train
        train_losses.append(epoch_train_loss)
        train_accuracies.append(epoch_train_acc)

        # Kiểm thử (Validation)
        model.eval()  # Đặt mô hình ở chế độ đánh giá
        running_val_loss = 0.0
        correct_val = 0
        total_val = 0
        with torch.no_grad():  # Tắt tính toán gradient trong quá trình kiểm thử
            for images, labels in tqdm(val_loader, desc=f"Epoch {epoch + 1}/{num_epochs} [Val]"):
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)

                running_val_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()

        epoch_val_loss = running_val_loss / len(val_loader.dataset)
        epoch_val_acc = correct_val / total_val
        val_losses.append(epoch_val_loss)
        val_accuracies.append(epoch_val_acc)

        print(f"Epoch {epoch + 1}/{num_epochs}:")
        print(f"  Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc:.4f}")
        print(f"  Val Loss: {epoch_val_loss:.4f}, Val Acc: {epoch_val_acc:.4f}")

    print("\nHuấn luyện hoàn tất!")

    # 4. Vẽ đồ thị biểu diễn sự thay đổi của accuracy và loss
    epochs_range = range(1, num_epochs + 1)

    plt.figure(figsize=(12, 5))

    # Đồ thị Loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, train_losses, label='Train Loss')
    plt.plot(epochs_range, val_losses, label='Validation Loss')
    plt.title('Loss qua các Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    # Đồ thị Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, train_accuracies, label='Train Accuracy')
    plt.plot(epochs_range, val_accuracies, label='Validation Accuracy')
    plt.title('Accuracy qua các Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

    print("\nNhận xét về hiệu quả của việc huấn luyện:")
    print("Quan sát đồ thị Loss và Accuracy để đánh giá:")
    print(
        "- Hiện tượng hồi tụ (Convergence): Cả loss và accuracy trên tập huấn luyện và kiểm thử có xu hướng ổn định sau một số epochs.")
    print(
        "- Overfitting: Nếu Train Loss tiếp tục giảm nhưng Validation Loss bắt đầu tăng trở lại, và Train Accuracy cao hơn đáng kể so với Validation Accuracy, đó là dấu hiệu của overfitting.")
    print(
        "- Underfitting: Nếu cả Train Loss và Validation Loss đều cao, và cả hai accuracy đều thấp, có thể mô hình chưa đủ phức tạp hoặc chưa được huấn luyện đủ lâu.")
    print(
        "- Siêu tham số (Hyperparameters): Quan sát xem các đường cong có đạt được kết quả tốt nhất hay không. Nếu không, có thể cần điều chỉnh learning rate, batch size, số lượng epochs, hoặc các tham số của mô hình.")


if __name__ == '__main__':
    # (c) Chạy thử mô hình với dữ liệu đầu vào giả (giữ lại để kiểm tra cấu trúc)
    # Phần này độc lập với phần huấn luyện chính
    batch_size_dummy = 4
    input_channels_dummy = 3
    input_height_dummy = 224
    input_width_dummy = 224
    dummy_input = torch.randn(batch_size_dummy, input_channels_dummy, input_height_dummy, input_width_dummy)

    model_dummy_test = ResNet18(num_classes=10)  # Using 10 classes as an example for dummy test

    print("--- Kiểm tra cấu trúc mô hình với dữ liệu giả ---")
    print(f"Input for dummy test: {dummy_input.shape}")

    # Run forward pass through the model and print shape at each stage
    x_dummy = model_dummy_test.conv1(dummy_input)
    x_dummy = model_dummy_test.bn1(x_dummy)
    x_dummy = model_dummy_test.relu(x_dummy)
    print(f"After conv1, bn1, relu (dummy test): {x_dummy.shape}")
    x_dummy = model_dummy_test.maxpool(x_dummy)
    print(f"After maxpool (dummy test): {x_dummy.shape}")

    x_dummy = model_dummy_test.layer1(x_dummy)
    print(f"After layer1 (dummy test): {x_dummy.shape}")
    x_dummy = model_dummy_test.layer2(x_dummy)
    print(f"After layer2 (dummy test): {x_dummy.shape}")
    x_dummy = model_dummy_test.layer3(x_dummy)
    print(f"After layer3 (dummy test): {x_dummy.shape}")
    x_dummy = model_dummy_test.layer4(x_dummy)
    print(f"After layer4 (dummy test): {x_dummy.shape}")

    x_dummy = model_dummy_test.avgpool(x_dummy)
    print(f"After avgpool (dummy test): {x_dummy.shape}")
    x_dummy = torch.flatten(x_dummy, 1)
    print(f"After flatten (dummy test): {x_dummy.shape}")
    x_dummy = model_dummy_test.dropout(x_dummy)
    print(f"After dropout (dummy test): {x_dummy.shape}")
    x_dummy = model_dummy_test.fc1(x_dummy)
    x_dummy = model_dummy_test.bn_fc1(x_dummy)
    x_dummy = model_dummy_test.relu(x_dummy)
    print(f"After fc1, bn_fc1, relu (dummy test): {x_dummy.shape}")
    x_dummy = model_dummy_test.fc2(x_dummy)
    print(f"Final output (dummy test): {x_dummy.shape}")

    print("\n--- Bắt đầu quá trình huấn luyện chính ---")
    train_model()

    # (b) Visualize the network architecture and flow using torchviz (optional)
    # Uncomment the following lines and ensure graphviz is installed
    # from torchviz import make_dot
    # G = make_dot(model_dummy_test(dummy_input), params=dict(model_dummy_test.named_parameters()))
    # G.render("resnet18_graph", view=True)
    # print("\nNetwork graph saved as resnet18_graph.pdf")
