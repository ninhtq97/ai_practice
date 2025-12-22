"""
Object Detection using SURF (Speeded-Up Robust Features)
Phát hiện đặc trưng sử dụng SURF theo OpenCV docs
"""

import argparse
import cv2
import numpy as np
from pathlib import Path


def detect_surf_features(image_path, output_dir="outputs"):
    """
    Phát hiện đặc trưng SURF trong ảnh và lưu kết quả.

    Args:
        image_path: Đường dẫn tới ảnh cần phát hiện
        output_dir: Thư mục lưu kết quả

    Returns:
        keypoints: Danh sách các điểm đặc trưng phát hiện được
        descriptors: Descriptors tương ứng
    """
    # Đọc ảnh
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Không thể đọc ảnh: {image_path}")

    # Chuyển sang grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Khởi tạo SURF detector
    try:
        surf = cv2.xfeatures2d.SURF_create(400)  # hessian threshold = 400
        print(f"✓ Đã khởi tạo SURF detector")
    except (AttributeError, cv2.error) as e:
        raise RuntimeError(
            f"SURF không khả dụng trong OpenCV hiện tại.\n"
            f"Vui lòng build OpenCV với OPENCV_ENABLE_NONFREE=ON.\n"
            f"Chi tiết: {str(e)}"
        )

    # Phát hiện keypoints và tính descriptors
    keypoints, descriptors = surf.detectAndCompute(gray, None)

    print(f"Đã phát hiện {len(keypoints)} keypoints")

    # Vẽ keypoints lên ảnh
    img_with_keypoints = cv2.drawKeypoints(
        img,
        keypoints,
        None,
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    )

    # Tạo thư mục output nếu chưa có
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Lưu ảnh kết quả
    img_name = Path(image_path).stem
    output_file = output_path / f"{img_name}_surf_detected.jpg"
    cv2.imwrite(str(output_file), img_with_keypoints)

    print(f"✓ Đã lưu kết quả: {output_file}")

    # Lưu thông tin keypoints vào file text
    info_file = output_path / f"{img_name}_surf_info.txt"
    with open(info_file, 'w', encoding='utf-8') as f:
        f.write(f"Ảnh: {image_path}\n")
        f.write(f"Số keypoints: {len(keypoints)}\n\n")
        f.write("Chi tiết keypoints (10 điểm đầu):\n")
        for i, kp in enumerate(keypoints[:10]):
            f.write(f"  {i+1}. Tọa độ: ({kp.pt[0]:.2f}, {kp.pt[1]:.2f}), "
                   f"Size: {kp.size:.2f}, Angle: {kp.angle:.2f}°, "
                   f"Response: {kp.response:.4f}\n")

    print(f"✓ Đã lưu thông tin: {info_file}")

    return keypoints, descriptors


def list_image_files(data_dir="data"):
    """
    Liệt kê tất cả các file ảnh trong thư mục.

    Args:
        data_dir: Đường dẫn thư mục chứa ảnh

    Returns:
        List các đường dẫn ảnh
    """
    data_path = Path(data_dir)
    if not data_path.exists():
        raise FileNotFoundError(f"Thư mục không tồn tại: {data_dir}")

    # Danh sách extension ảnh hỗ trợ
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.JPG', '.JPEG', '.PNG'}

    # Tìm tất cả file ảnh (tìm recursive)
    image_files = sorted([
        f for f in data_path.rglob('*')
        if f.is_file() and f.suffix in image_extensions
    ])

    return image_files


def parse_args():
    parser = argparse.ArgumentParser(
        description="Phát hiện đặc trưng SURF trên tất cả ảnh trong thư mục data"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="./data",
        help="Thư mục chứa ảnh (mặc định: data)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs",
        help="Thư mục lưu kết quả (mặc định: outputs)"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print("=== SURF Feature Detection (Batch Mode) ===")
    print(f"Thư mục dữ liệu: {args.data_dir}")
    print(f"Thư mục output: {args.output_dir}")
    print()

    try:
        # Liệt kê các file ảnh
        image_files = list_image_files(args.data_dir)

        if not image_files:
            print(f"⚠️ Không tìm thấy ảnh nào trong {args.data_dir}")
            return 1

        print(f"Tìm thấy {len(image_files)} ảnh:")
        for f in image_files:
            print(f"  - {f}")
        print()

        # Xử lý từng ảnh
        total_keypoints = 0
        processed_count = 0
        failed_count = 0

        for image_file in image_files:
            try:
                print(f"Xử lý: {image_file.name}")
                keypoints, descriptors = detect_surf_features(
                    str(image_file),
                    output_dir=args.output_dir
                )
                total_keypoints += len(keypoints)
                processed_count += 1
            except Exception as e:
                print(f"  ❌ Lỗi: {e}")
                failed_count += 1
            print()

        # Tóm tắt kết quả
        print("=== Hoàn tất ===")
        print(f"Tổng ảnh: {len(image_files)}")
        print(f"Thành công: {processed_count}")
        print(f"Thất bại: {failed_count}")
        print(f"Tổng keypoints: {total_keypoints}")

        if failed_count == 0:
            return 0
        else:
            return 1

    except Exception as e:
        print(f"❌ Lỗi: {e}")
        return 1


if __name__ == '__main__':
    exit(main())
