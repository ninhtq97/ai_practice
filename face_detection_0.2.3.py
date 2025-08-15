import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import scipy.io
from matplotlib.patches import Rectangle
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC


# ==============================================================================
# HÀM 1: TẢI DỮ LIỆU TRAINING TỪ FILE .MAT
# ==============================================================================
def load_training_data(positive_file, negative_file):
    """
    Tải dữ liệu training dương tính (khuôn mặt) và âm tính (không phải khuôn mặt).
    """
    print("Đang tải dữ liệu training...")
    try:
        pos_data = scipy.io.loadmat(positive_file)
        neg_data = scipy.io.loadmat(negative_file)
        pos_samples = pos_data['possamples']
        neg_samples = neg_data['negsamples']
        patch_size = (pos_samples.shape[0], pos_samples.shape[1])
        print(f"Kích thước patch được xác định từ dữ liệu: {patch_size} pixels.")
        num_pos_samples = pos_samples.shape[2]
        num_neg_samples = neg_samples.shape[2]
        pos_samples_flat = pos_samples.reshape(-1, num_pos_samples).T
        neg_samples_flat = neg_samples.reshape(-1, num_neg_samples).T
        X = np.vstack((pos_samples_flat, neg_samples_flat))
        y = np.array([1] * num_pos_samples + [-1] * num_neg_samples)
        print(f"Tải xong! Tổng số mẫu: {len(y)}.")
        return X, y, patch_size
    except Exception as e:
        print(f"Lỗi khi tải dữ liệu: {e}")
        return None, None, None


# ==============================================================================
# HÀM 2: HUẤN LUYỆN MÔ HÌNH SVM
# ==============================================================================
def train_svm_model(X_train_scaled, y_train):
    """
       Sử dụng GridSearchCV để tìm tham số C tốt nhất cho LinearSVC và huấn luyện mô hình cuối cùng.

       Args:
           X_train_scaled (np.array): Dữ liệu training đã được chuẩn hóa.
           y_train (np.array): Nhãn của dữ liệu training.

       Returns:
           LinearSVC: Mô hình SVM tốt nhất đã được huấn luyện trên toàn bộ tập train.
       """
    print("\n--- Tinh chỉnh siêu tham số (Hyperparameter Tuning) cho SVM ---")

    # Định nghĩa lưới các giá trị C cần thử.
    # Các giá trị này trải dài trên nhiều bậc độ lớn.
    param_grid = {'C': [0.001, 0.01, 0.1, 1, 10]}

    # Thiết lập GridSearchCV
    # cv=3: sử dụng 3-fold cross-validation.
    # n_jobs=-1: sử dụng tất cả các CPU core có sẵn để tăng tốc quá trình.
    # max_iter: tăng số vòng lặp tối đa để đảm bảo thuật toán hội tụ.
    grid_search = GridSearchCV(
        LinearSVC(random_state=42, tol=1e-5, max_iter=4000),
        param_grid,
        cv=3,
        scoring='accuracy',
        n_jobs=-1
    )

    # Bắt đầu tìm kiếm
    print("Đang thực hiện Grid Search để tìm C tốt nhất... (có thể mất một lúc)")
    grid_search.fit(X_train_scaled, y_train)

    # In ra kết quả
    print("\nKết quả Grid Search:")
    print(f"  - Tham số C tốt nhất được tìm thấy: {grid_search.best_params_['C']}")
    print(f"  - Độ chính xác cross-validation tốt nhất: {grid_search.best_score_:.4f}")

    print("Huấn luyện hoàn tất với tham số C tốt nhất.")

    # GridSearchCV tự động huấn luyện lại mô hình tốt nhất trên toàn bộ tập dữ liệu train
    # và trả về mô hình đó qua thuộc tính `best_estimator_`
    return grid_search.best_estimator_


# ==============================================================================
# ĐÁNH GIÁ HIỆU SUẤT CỦA BỘ PHÂN LOẠI
# ==============================================================================
def evaluate_classifier(model, X_test_scaled, y_test):
    """
    Đánh giá hiệu suất của bộ phân loại trên tập dữ liệu test đã được chuẩn hóa.

    Args:
        model (LinearSVC): Mô hình SVM đã huấn luyện.
        X_test_scaled (np.array): Dữ liệu test đã được chuẩn hóa.
        y_test (np.array): Nhãn của dữ liệu test.
    """
    print("\n--- Đánh giá hiệu suất của bộ phân loại (Classifier) ---")
    predictions = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, predictions)
    print(f"Độ chính xác (Accuracy) trên tập test: {accuracy:.4f}")
    print("Báo cáo phân loại (Classification Report):")
    # 1: Face, -1: Non-Face. Sắp xếp nhãn để báo cáo dễ đọc.
    print(classification_report(y_test, predictions, target_names=['Non-Face (-1)', 'Face (1)']))


# ==============================================================================
# HÀM 3: THỰC HIỆN SLIDING WINDOW
# ==============================================================================
def sliding_window_detector(image, model, scaler, patch_size, step_size=4, scale_factor=1.25, conf_thresh=0.5):
    """
    Quét ảnh và phân loại từng patch, sử dụng scaler đã được huấn luyện.

    Args:
        image (np.array): Ảnh đầu vào (grayscale).
        model (LinearSVC): Mô hình SVM đã huấn luyện.
        scaler (StandardScaler): Scaler đã được fit trên dữ liệu training.
        ... các tham số khác ...
    """
    detections = []
    h_patch, w_patch = patch_size
    current_image = image.copy()
    current_scale = 1.0

    while current_image.shape[0] >= h_patch and current_image.shape[1] >= w_patch:
        for y in range(0, current_image.shape[0] - h_patch + 1, step_size):
            for x in range(0, current_image.shape[1] - w_patch + 1, step_size):
                patch = current_image[y:y + h_patch, x:x + w_patch]

                # Làm phẳng patch
                patch_flat = patch.flatten().reshape(1, -1).astype(np.float64)

                # **QUAN TRỌNG**: Dùng scaler đã fit để chuẩn hóa patch mới
                patch_scaled = scaler.transform(patch_flat)

                # Dự đoán trên patch đã chuẩn hóa
                score = model.decision_function(patch_scaled)

                if score[0] > conf_thresh:
                    original_x = int(x * current_scale)
                    original_y = int(y * current_scale)
                    original_w = int(w_patch * current_scale)
                    original_h = int(h_patch * current_scale)
                    detections.append({'box': [original_x, original_y, original_w, original_h], 'score': score[0]})

        new_width = int(current_image.shape[1] / scale_factor)
        new_height = int(current_image.shape[0] / scale_factor)
        current_image = cv2.resize(current_image, (new_width, new_height), interpolation=cv2.INTER_AREA)
        current_scale *= scale_factor

    return detections


# ==============================================================================
# HÀM 4 & 5: NMS VÀ VẼ KẾT QUẢ
# ==============================================================================
def non_maximal_suppression(detections, iou_threshold):
    if not detections: return []
    boxes = np.array([d['box'] for d in detections])
    scores = np.array([d['score'] for d in detections])
    x1, y1 = boxes[:, 0], boxes[:, 1]
    x2, y2 = x1 + boxes[:, 2], y1 + boxes[:, 3]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]
    keep_indices = []
    while order.size > 0:
        i = order[0]
        keep_indices.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        intersection = w * h
        union = areas[i] + areas[order[1:]] - intersection
        iou = intersection / union
        inds_to_keep = np.where(iou <= iou_threshold)[0]
        order = order[inds_to_keep + 1]
    return [detections[i] for i in keep_indices]


def display_results(image, detections, title=""):
    fig, ax = plt.subplots(1, figsize=(12, 8))
    ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    for d in detections:
        box, score = d['box'], d['score']
        x, y, w, h = box
        rect = Rectangle((x, y), w, h, linewidth=2, edgecolor='lime', facecolor='none')
        ax.add_patch(rect)
        ax.text(x, y - 10, f'{score:.2f}', color='white', fontsize=10,
                bbox=dict(facecolor='lime', alpha=0.8, edgecolor='none', boxstyle='round,pad=0.2'))
    plt.axis('off')
    plt.title(title)
    plt.tight_layout()
    plt.show()


# ==============================================================================
# HÀM CHÍNH (MAIN)
# ==============================================================================
def main():
    """
    Hàm chính điều phối toàn bộ quy trình Machine Learning.
    """
    # --- THAM SỐ CẤU HÌNH ---
    DATA_BASED_PATH = 'data'
    POS_SAMPLES_FILE = os.path.join(DATA_BASED_PATH, 'possamples.mat')
    NEG_SAMPLES_FILE = os.path.join(DATA_BASED_PATH, 'negsamples.mat')
    IMAGE_FILES = ['img1.jpg', 'img2.jpg', 'img3.jpg', 'img4.jpg']
    CONF_THRESH_PRE_NMS = 1.0
    IOU_THRESH_NMS = 0.2

    # --- BƯỚC 1: TẢI DỮ LIỆU ---
    X, y, patch_size = load_training_data(POS_SAMPLES_FILE, NEG_SAMPLES_FILE)
    if X is None:
        print("Kết thúc chương trình do không tải được dữ liệu.")
        return

    # --- BƯỚC 2: CHIA DỮ LIỆU TRAIN-TEST ---
    print("\n--- Chia dữ liệu thành tập Train và Test (80/20) ---")
    # stratify=y để đảm bảo tỷ lệ face/non-face trong tập train và test là như nhau
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Kích thước tập Train: {X_train.shape[0]} mẫu")
    print(f"Kích thước tập Test: {X_test.shape[0]} mẫu")

    # --- BƯỚC 3: CHUẨN HÓA DỮ LIỆU (SCALING) ---
    print("\n--- Chuẩn hóa dữ liệu (Standardization) ---")
    # Quan trọng: Fit scaler CHỈ trên tập train và dùng nó để transform cho cả train và test
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train.astype(np.float64))
    X_test_scaled = scaler.transform(X_test.astype(np.float64))
    print("Chuẩn hóa hoàn tất.")

    # --- BƯỚC 4: HUẤN LUYỆN SVM TRÊN DỮ LIỆU ĐÃ CHUẨN HÓA ---
    svm_model = train_svm_model(X_train_scaled, y_train)

    # --- BƯỚC 5: ĐÁNH GIÁ HIỆU SUẤT CỦA BỘ PHÂN LOẠI ---
    evaluate_classifier(svm_model, X_test_scaled, y_test)

    # --- BƯỚC 6: ÁP DỤNG BỘ NHẬN DIỆN (DETECTOR) LÊN ẢNH THỰC TẾ ---
    print("\n--- Áp dụng bộ nhận diện lên các ảnh test ---")
    IMG_BASE_PATH = 'images'
    for image_file in IMAGE_FILES:
        img_file = os.path.join(IMG_BASE_PATH, image_file)
        if not os.path.exists(img_file):
            print(f"\nCảnh báo: Không tìm thấy file ảnh {image_file}. Bỏ qua.")
            continue

        print(f"\n{'=' * 20} Đang xử lý ảnh: {image_file} {'=' * 20}")
        image_color = cv2.imread(img_file)
        if image_color is None: continue
        image_gray = cv2.cvtColor(image_color, cv2.COLOR_BGR2GRAY)

        # **QUAN TRỌNG**: Truyền scaler đã fit vào hàm detector
        raw_detections = sliding_window_detector(
            image_gray, svm_model, scaler, patch_size,
            step_size=4, scale_factor=1.25, conf_thresh=CONF_THRESH_PRE_NMS
        )
        print(f"Tìm thấy {len(raw_detections)} phát hiện thô (trước NMS).")

        final_detections = non_maximal_suppression(raw_detections, IOU_THRESH_NMS)
        print(f"Số phát hiện cuối cùng sau NMS: {len(final_detections)}.")

        title = f'Kết quả trên {image_file} (conf={CONF_THRESH_PRE_NMS}, iou={IOU_THRESH_NMS})'
        display_results(image_color, final_detections, title)


if __name__ == '__main__':
    main()
