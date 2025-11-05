import os  # Dùng để lấy tên file từ đường dẫn
import random  # Dùng để chọn ngẫu nhiên ảnh
from collections import Counter  # Dùng để đếm số lượng mẫu mỗi lớp
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from PIL import Image  # Dùng để mở ảnh khi hiển thị ảnh gốc cho augmentation
from torch.utils.data import DataLoader

# --- 1. CÁC THAM SỐ CẤU HÌNH ---
# Thư mục gốc để chứa dữ liệu
DATA_DIR = Path("data")
# Đường dẫn đến thư mục dữ liệu sau khi giải nén
# LƯU Ý: Hãy chắc chắn rằng bạn đã giải nén bộ dữ liệu vào đúng đường dẫn này: "data/imagenette2-320"
DATASET_PATH = DATA_DIR / "imagenette2-320"
# Kích thước ảnh đầu vào cho mô hình (ví dụ: 224x224 cho ResNet)
IMAGE_SIZE = 224
# Số lượng ảnh trong mỗi batch
BATCH_SIZE = 32
# Số luồng song song để tải dữ liệu. Tăng giá trị này nếu CPU của bạn mạnh.
# Đặt NUM_WORKERS=0 nếu vẫn gặp lỗi hoặc đang debug để đơn giản hóa.
NUM_WORKERS = 2

# --- 2. ĐỊNH NGHĨA CÁC PHÉP BIẾN ĐỔI (TRANSFORMS) ---

# Các giá trị trung bình (mean) và độ lệch chuẩn (std) của bộ ImageNet.
# Việc chuẩn hóa giúp mô hình hội tụ nhanh và ổn định hơn.
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

# Pipeline các phép biến đổi và tăng cường dữ liệu cho tập huấn luyện (train)
train_transforms = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),  # 1. Thay đổi kích thước ảnh về kích thước chuẩn
    transforms.RandomHorizontalFlip(),  # 2. Lật ngang ảnh ngẫu nhiên (p=0.5)
    transforms.RandomRotation(15),  # 3. Xoay ảnh ngẫu nhiên trong khoảng [-15, 15] độ
    transforms.ColorJitter(brightness=0.2, contrast=0.2),  # 4. Thay đổi độ sáng và tương phản ngẫu nhiên
    transforms.ToTensor(),  # 5. Chuyển ảnh (PIL/numpy) thành Tensor PyTorch
    normalize,  # 6. Chuẩn hóa Tensor ảnh
])

# Pipeline các phép biến đổi cho tập kiểm định (validation)
# Không áp dụng các phép tăng cường ngẫu nhiên để kết quả đánh giá được nhất quán
val_transforms = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    normalize,
])

# --- 3. TẠO DATASET VÀ DATALOADER ---

# Kiểm tra xem đường dẫn dữ liệu có tồn tại không
if not DATASET_PATH.exists():
    print(f"Lỗi: Không tìm thấy thư mục dữ liệu tại '{DATASET_PATH}'")
    print("Vui lòng tải và giải nén bộ dữ liệu Imagenette2 vào thư mục 'data/imagenette2-320' trước khi chạy.")
    # Thoát chương trình nếu không có dữ liệu
    exit()

# Sử dụng ImageFolder để tạo Dataset từ cấu trúc thư mục
# PyTorch sẽ tự động gán nhãn (label) là chỉ số của thư mục con (ví dụ: n01440764 -> 0, n02102040 -> 1, ...)
train_dataset = torchvision.datasets.ImageFolder(
    root=str(DATASET_PATH / "train"),
    transform=train_transforms
)

val_dataset = torchvision.datasets.ImageFolder(
    root=str(DATASET_PATH / "val"),
    transform=val_transforms
)

# Tạo DataLoader để quản lý việc nạp dữ liệu theo từng batch
train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,  # Xáo trộn dữ liệu ở mỗi epoch, rất quan trọng cho việc huấn luyện
    num_workers=NUM_WORKERS,  # Sử dụng các tiến trình con để tải dữ liệu, giúp tăng tốc
    # pin_memory=True  # Giúp tăng tốc độ chuyển dữ liệu từ CPU sang GPU (nếu có)
)

val_loader = DataLoader(
    dataset=val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,  # Không cần xáo trộn ở tập validation
    num_workers=NUM_WORKERS,
    # pin_memory=True
)

print("\n--- TÓM TẮT THÔNG TIN DATASET VÀ DATALOADER ---")
print(f"Số lượng ảnh trong tập train: {len(train_dataset)}")
print(f"Số lượng ảnh trong tập validation: {len(val_dataset)}")
print(f"Số lượng batch trong train_loader: {len(train_loader)}")
print(f"Số lượng batch trong val_loader: {len(val_loader)}")

# Lấy thông tin về các lớp và ánh xạ từ chỉ số sang tên lớp
# Tên lớp chính là tên thư mục (WNID)
class_names = train_dataset.classes
print(f"\nTìm thấy {len(class_names)} lớp. Các WNID tương ứng:")
print(class_names)


# --- 4. CÁC HÀM TRỰC QUAN HÓA DỮ LIỆU ---

def imshow(inp: torch.Tensor, title: str = None, ax=None, fontsize=10):
    """Hàm để hiển thị một batch ảnh tensor."""
    # Dữ liệu trong DataLoader đã được chuẩn hóa, cần "hoàn tác" lại để hiển thị đúng
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)  # Giới hạn giá trị pixel trong khoảng [0, 1]

    if ax is None:
        # Giữ nguyên việc tạo figure mới nếu ax không được truyền vào,
        # điều này thường chỉ xảy ra khi imshow được gọi riêng lẻ
        plt.figure(figsize=(6, 8))
        plt.imshow(inp)
        if title is not None:
            plt.title(title, fontsize=fontsize)
        plt.axis('off')
        plt.show()
    else:
        ax.imshow(inp)
        if title is not None:
            ax.set_title(title, fontsize=fontsize)
        ax.axis('off')


# Đảm bảo mã chạy trong tiến trình chính khi sử dụng num_workers > 0
if __name__ == '__main__':
    # --- 4.1. TRỰC QUAN HÓA: THỐNG KÊ SỐ LƯỢNG MẪU MỖI LỚP ---
    print("\n--- TRỰC QUAN HÓA: THỐNG KÊ SỐ LƯỢNG MẪU MỖI LỚP ---")

    # Đếm số lượng mẫu cho mỗi lớp trong tập huấn luyện
    class_counts = Counter(train_dataset.targets)
    # Sắp xếp theo thứ tự nhãn
    sorted_class_counts = sorted(class_counts.items())

    # Tách nhãn (chỉ số) và số lượng
    labels_idx = [item[0] for item in sorted_class_counts]
    counts = [item[1] for item in sorted_class_counts]

    # Ánh xạ chỉ số nhãn sang tên lớp thực tế
    class_names_readable = [train_dataset.classes[idx] for idx in labels_idx]

    plt.figure(figsize=(10, 6))  # Giảm chiều rộng một chút để vừa hơn
    plt.bar(class_names_readable, counts, color='skyblue')
    plt.xlabel("Tên lớp (WNID)", fontsize=12)
    plt.ylabel("Số lượng ảnh", fontsize=12)
    plt.title("Số lượng ảnh trên mỗi lớp trong tập huấn luyện", fontsize=14)
    plt.xticks(rotation=45, ha='right', fontsize=10)  # Xoay nhãn trục x để dễ đọc
    plt.yticks(fontsize=10)
    plt.tight_layout()  # Điều chỉnh layout để tránh chồng chéo
    plt.show()

    print("\nNhận xét về số lượng mẫu mỗi lớp:")
    print("- Biểu đồ cho thấy phân phối số lượng ảnh giữa các lớp là tương đối đồng đều.")
    print(
        "- Điều này là lý tưởng cho việc huấn luyện mô hình, giúp tránh tình trạng mô hình thiên vị về các lớp có nhiều dữ liệu hơn.")
    print("- Mỗi lớp có khoảng hơn 900 ảnh, cung cấp đủ dữ liệu để mô hình học các đặc trưng riêng biệt cho từng loại.")

    # --- 4.2. TRỰC QUAN HÓA: HIỂN THỊ 3 ẢNH MẪU CỦA MỖI LỚP ---
    print("\n--- TRỰC QUAN HÓA: HIỂN THỊ 3 ẢNH MẪU CỦA MỖI LỚP ---")
    plt.close('all')  # Đóng tất cả các figure hiện có để tránh việc hiển thị trùng lặp

    # Tạo một dictionary để lưu trữ đường dẫn ảnh theo từng lớp
    class_image_paths = {class_idx: [] for class_idx in range(len(train_dataset.classes))}
    for path, class_idx in train_dataset.samples:
        class_image_paths[class_idx].append(path)

    num_images_per_class_to_show = 3
    num_classes = len(train_dataset.classes)

    # Giảm figsize.height và y để thu gọn khoảng trống phía trên
    # Tăng wspace và hspace để tạo khoảng trống giữa các ảnh và hàng ảnh
    # Giảm left để tạo thêm không gian cho nhãn bên trái
    fig, axes = plt.subplots(num_classes, num_images_per_class_to_show, figsize=(10, num_classes * 1.1))

    # Điều chỉnh khoảng cách giữa các subplot và thêm padding bên trái cho nhãn
    # Đã điều chỉnh 'left' xuống 0.05 để nhãn có thể sát hơn với ảnh
    plt.subplots_adjust(left=0.05, right=0.95, top=0.92, bottom=0.05, wspace=0.2, hspace=0.2)

    # Tiêu đề chung cho toàn bộ figure
    fig.suptitle('3 Ảnh mẫu từ mỗi lớp trong tập huấn luyện', fontsize=16, y=0.97)

    for i, (class_idx, image_paths) in enumerate(class_image_paths.items()):
        # Chọn ngẫu nhiên 3 ảnh (hoặc ít hơn nếu không đủ)
        selected_paths = random.sample(image_paths, min(num_images_per_class_to_show, len(image_paths)))

        # Lấy tên file của các ảnh được chọn để đưa vào tiêu đề
        selected_filenames = [os.path.basename(p) for p in selected_paths]

        # Thêm WNID và tên file ảnh làm nhãn hàng ngang
        # Tính toán vị trí x và y cho nhãn hàng:
        # x_text_position: Lấy x0 của subplot đầu tiên trong hàng, sau đó trừ đi một offset nhỏ.
        #                  max(0.005, ...) để đảm bảo không đi quá mép trái của figure.
        x_text_position = max(0.005, axes[i, 0].get_position().x0 - 0.02)
        y_text_position = (axes[i, 0].get_position().y0 + axes[i, 0].get_position().y1) / 2

        fig.text(
            x_text_position,  # Vị trí x đã được điều chỉnh để ngay cạnh ảnh đầu tiên
            y_text_position,  # Vị trí y (đặt giữa hàng ảnh)
            f"{train_dataset.classes[class_idx]}:",
            fontsize=7,
            verticalalignment='center',  # Căn giữa theo chiều dọc
            horizontalalignment='left',  # Căn lề trái
            transform=fig.transFigure  # Tọa độ tương đối với figure
        )

        for j, img_path in enumerate(selected_paths):
            image = Image.open(img_path).convert('RGB')
            # Áp dụng val_transforms để chuẩn hóa và chuyển thành tensor để hiển thị
            processed_image = val_transforms(image)

            current_ax = axes[i, j]
            imshow(processed_image, ax=current_ax, fontsize=7)

    plt.show()

    # --- 4.3. TRỰC QUAN HÓA: HIỂN THỊ KẾT QUẢ TĂNG CƯỜNG DỮ LIỆU (AUGMENTATION) ---
    print("\n--- TRỰC QUAN HÓA: HIỂN THỊ KẾT QUẢ TĂNG CƯỜNG DỮ LIỆU (AUGMENTATION) ---")

    # Chọn một ảnh ngẫu nhiên từ tập dữ liệu gốc để minh họa
    random_idx = random.randint(0, len(train_dataset.samples) - 1)
    original_image_path, original_label_idx = train_dataset.samples[random_idx]
    original_pil_image = Image.open(original_image_path).convert('RGB')

    # Định nghĩa các phép biến đổi riêng lẻ để dễ dàng minh họa
    transform_resize_tensor_normalize = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        normalize
    ])

    transform_horizontal_flip = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(p=1.0),  # Luôn lật ngang
        transforms.ToTensor(),
        normalize
    ])

    transform_random_rotation = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomRotation(degrees=15),
        transforms.ToTensor(),
        normalize
    ])

    transform_color_jitter = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ColorJitter(brightness=0.8, contrast=0.8, saturation=0.8, hue=0.1),  # Thay đổi mạnh hơn để dễ thấy
        transforms.ToTensor(),
        normalize
    ])

    # Tạo danh sách các ảnh để hiển thị: ảnh gốc và các phiên bản đã tăng cường
    images_to_display = []
    titles_to_display = []

    # Ảnh gốc (chỉ thay đổi kích thước và chuẩn hóa để hiển thị)
    images_to_display.append(transform_resize_tensor_normalize(original_pil_image))
    titles_to_display.append("Ảnh gốc")

    # Lật ngang
    images_to_display.append(transform_horizontal_flip(original_pil_image))
    titles_to_display.append("Lật ngang")

    # Xoay ngẫu nhiên
    images_to_display.append(transform_random_rotation(original_pil_image))
    titles_to_display.append("Xoay ngẫu nhiên (15 độ)")

    # Thay đổi độ sáng/tương phản
    images_to_display.append(transform_color_jitter(original_pil_image))
    titles_to_display.append("Độ sáng/tương phản")

    # Hiển thị tất cả ảnh trong một lưới
    fig, axes = plt.subplots(1, len(images_to_display), figsize=(11, 3.5))  # Giảm kích thước tổng thể
    fig.suptitle(f"Minh họa các kỹ thuật tăng cường dữ liệu cho lớp: {train_dataset.classes[original_label_idx]}",
                 fontsize=14)  # Giảm fontsize tiêu đề chính

    # Điều chỉnh khoảng cách giữa các subplot
    plt.subplots_adjust(wspace=0.1, hspace=0.1)

    for i, img_tensor in enumerate(images_to_display):
        imshow(img_tensor, title=titles_to_display[i], ax=axes[i], fontsize=9)  # Giảm fontsize tiêu đề ảnh

    plt.show()

    # --- 4.4. TRỰC QUAN HÓA: MỘT BATCH ẢNH TỪ TRAIN LOADER (ĐÃ QUA AUGMENTATION) ---
    print("\n--- TRỰC QUAN HÓA: MỘT BATCH ẢNH TỪ TRAIN LOADER ---")
    try:
        # iter() tạo một iterator, next() lấy phần tử tiếp theo (là một batch)
        images, labels = next(iter(train_loader))

        # Tạo một lưới ảnh để hiển thị (chỉ lấy 16 ảnh đầu tiên trong batch)
        out = torchvision.utils.make_grid(images[:16], nrow=4)

        # Lấy tên các lớp tương ứng với 16 ảnh này
        class_titles = [class_names[x] for x in labels[:16]]

        # Định dạng lại title để hiển thị trên nhiều dòng
        formatted_title = "Một batch ảnh từ tập Train (đã qua Augmentation)\n"
        for i in range(0, 16, 4):
            formatted_title += f"{[class_names[labels[j]] for j in range(i, min(i + 4, len(labels)))]}\n"

        print("\nĐang hiển thị một batch ảnh mẫu...")
        imshow(out, title=formatted_title, fontsize=12)  # Giảm fontsize tiêu đề ảnh batch

    except Exception as e:
        print(f"\nLỗi khi trực quan hóa dữ liệu: {e}")
        print("Môi trường hiện tại có thể không hỗ trợ hiển thị đồ họa (GUI).")
        print("Hãy thử chạy mã này trong môi trường như Jupyter Notebook hoặc Google Colab.")
