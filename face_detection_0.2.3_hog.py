import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat
from skimage.feature import hog
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# Kích thước của mỗi bản vá hình ảnh (patch) - giả sử từ tài liệu
PATCH_SIZE = (24, 24)
# Kích thước bước cho cửa sổ trượt
SLIDING_WINDOW_STEP = 8
# Tham số cho HOG, có thể điều chỉnh
HOG_ORIENTATIONS = 9
HOG_PIXELS_PER_CELL = (8, 8)
HOG_CELLS_PER_BLOCK = (3, 3)


# visualize=False để HOG chỉ trả về feature vector, không phải ảnh HOG

# --- Phần 1: Chuẩn bị dữ liệu huấn luyện ---

def extract_hog_features(image_patch, patch_size):
    """
    Trích xuất đặc trưng HOG từ một bản vá hình ảnh.
    Args:
        image_patch (np.array): Bản vá hình ảnh.
        patch_size (tuple): Kích thước (chiều rộng, chiều cao) của bản vá.
    Returns:
        np.array: Vector đặc trưng HOG.
    """
    # Đảm bảo bản vá có kích thước chính xác trước khi tính HOG
    if image_patch.shape != patch_size:
        # Thay đổi kích thước bản vá nếu nó không khớp với PATCH_SIZE
        image_patch = cv2.resize(image_patch, patch_size, interpolation=cv2.INTER_AREA)

    # Tính toán đặc trưng HOG
    # Đã loại bỏ 'multichannel=False' vì nó có thể gây lỗi TypeError với một số phiên bản skimage cũ hơn.
    # Vì ảnh đầu vào đã là ảnh xám (grayscale), tham số này không cần thiết.
    features = hog(image_patch, orientations=HOG_ORIENTATIONS,
                   pixels_per_cell=HOG_PIXELS_PER_CELL,
                   cells_per_block=HOG_CELLS_PER_BLOCK,
                   visualize=False)
    return features


def load_and_prepare_data(pos_mat_path, neg_mat_path, patch_size):
    """
    Tải dữ liệu từ các tệp .mat, trích xuất đặc trưng HOG, định dạng và chuẩn bị cho SVM.
    Args:
        pos_mat_path (str): Đường dẫn đến tệp .mat chứa các mẫu dương tính.
        neg_mat_path (str): Đường dẫn đến tệp .mat chứa các mẫu âm tính.
        patch_size (tuple): Kích thước (chiều rộng, chiều cao) của mỗi bản vá hình ảnh.
    Returns:
        tuple: (Xtrain, ytrain, Xval, yval, mean, std), dữ liệu đã chuẩn bị
               cùng với giá trị trung bình và độ lệch chuẩn để chuẩn hóa.
    """
    print("--- Bắt đầu Phần 1: Chuẩn bị dữ liệu huấn luyện ---")
    try:
        data_pos = loadmat(pos_mat_path)
        data_neg = loadmat(neg_mat_path)
    except FileNotFoundError:
        print(f"Lỗi: Không tìm thấy tệp {pos_mat_path} hoặc {neg_mat_path}. Vui lòng đảm bảo chúng ở đúng vị trí.")
        return None, None, None, None, None, None

    # Thử các khóa khác nhau để tìm dữ liệu mẫu
    X_pos_raw = data_pos.get('possamples')
    if X_pos_raw is None:
        print("Lỗi: Không tìm thấy khóa 'possamples' trong tệp dương tính. Vui lòng kiểm tra cấu trúc tệp .mat.")
        return None, None, None, None, None, None

    X_neg_raw = data_neg.get('negsamples')
    if X_neg_raw is None:
        print("Lỗi: Không tìm thấy khóa 'negsamples' trong tệp âm tính. Vui lòng kiểm tra cấu trúc tệp .mat.")
        return None, None, None, None, None, None

    print(f"Shape of X_pos_raw: {X_pos_raw.shape}")
    print(f"Shape of X_neg_raw: {X_neg_raw.shape}")

    hog_features_pos = []
    # Xử lý dữ liệu dương tính để đảm bảo định dạng (num_samples, H, W)
    if X_pos_raw.ndim == 3:
        # MATLAB thường lưu dưới dạng (height, width, num_samples)
        # Nếu dimension cuối cùng lớn hơn đáng kể so với các dimension khác, đó có thể là số lượng mẫu
        # Ưu tiên chuyển về (num_samples, height, width)
        if X_pos_raw.shape[2] > X_pos_raw.shape[0] and X_pos_raw.shape[2] > X_pos_raw.shape[1]:
            X_pos_processed = X_pos_raw.transpose(2, 0, 1)  # Chuyển đổi từ (H, W, N) sang (N, H, W)
        else:
            X_pos_processed = X_pos_raw  # Có thể đã ở dạng (N, H, W) hoặc một số định dạng 3D khác
    elif X_pos_raw.ndim == 2:  # (num_samples, H*W) - cần định hình lại
        # Đảm bảo định hình lại thành (num_samples, height, width)
        X_pos_processed = X_pos_raw.reshape(X_pos_raw.shape[0], patch_size[1], patch_size[0])
    else:
        print(f"Lỗi: Định dạng dữ liệu X_pos_raw không mong muốn ({X_pos_raw.ndim} chiều).")
        return None, None, None, None, None, None

    for i in range(X_pos_processed.shape[0]):
        hog_features_pos.append(extract_hog_features(X_pos_processed[i], patch_size))

    hog_features_neg = []
    # Xử lý dữ liệu âm tính
    if X_neg_raw.ndim == 3:
        if X_neg_raw.shape[2] > X_neg_raw.shape[0] and X_neg_raw.shape[2] > X_neg_raw.shape[1]:
            X_neg_processed = X_neg_raw.transpose(2, 0, 1)  # Chuyển đổi từ (H, W, N) sang (N, H, W)
        else:
            X_neg_processed = X_neg_raw
    elif X_neg_raw.ndim == 2:  # (num_samples, H*W)
        X_neg_processed = X_neg_raw.reshape(X_neg_raw.shape[0], patch_size[1], patch_size[0])
    else:
        print(f"Lỗi: Định dạng dữ liệu X_neg_raw không mong muốn ({X_neg_raw.ndim} chiều).")
        return None, None, None, None, None, None

    for i in range(X_neg_processed.shape[0]):
        hog_features_neg.append(extract_hog_features(X_neg_processed[i], patch_size))

    X_hog_pos = np.array(hog_features_pos)
    X_hog_neg = np.array(hog_features_neg)

    print(f"Số lượng mẫu dương tính (HOG): {X_hog_pos.shape[0]}")
    print(f"Số lượng mẫu âm tính (HOG): {X_hog_neg.shape[0]}")
    print(f"Kích thước mỗi vector HOG: {X_hog_pos.shape[1]}")

    # Gán nhãn: 1 cho khuôn mặt, -1 cho không phải khuôn mặt
    y_pos = np.ones(X_hog_pos.shape[0])
    y_neg = -np.ones(X_hog_neg.shape[0])

    # Kết hợp dữ liệu HOG
    X = np.vstack((X_hog_pos, X_hog_neg))
    y = np.hstack((y_pos, y_neg))

    # Chuẩn hóa trung bình-phương sai (Mean-variance normalization) cho các đặc trưng HOG
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    std[std == 0] = 1  # Tránh chia cho 0
    X_normalized = (X - mean) / std

    # Tách dữ liệu thành tập huấn luyện và tập xác thực (80% train, 20% val)
    Xtrain, Xval, ytrain, yval = train_test_split(X_normalized, y, test_size=0.2, random_state=42)

    print(f"Kích thước tập huấn luyện (HOG): Xtrain={Xtrain.shape}, ytrain={ytrain.shape}")
    print(f"Kích thước tập xác thực (HOG): Xval={Xval.shape}, yval={yval.shape}")
    print("--- Kết thúc Phần 1 ---")
    return Xtrain, ytrain, Xval, yval, mean, std


# --- Phần 2: Phân loại SVM ---
# (Phần này không thay đổi vì SVM làm việc với bất kỳ vector đặc trưng nào)

def train_and_evaluate_svm(Xtrain, ytrain, Xval, yval, patch_size):
    """
    Huấn luyện và đánh giá bộ phân loại SVM tuyến tính, chọn giá trị C tốt nhất.
    Args:
        Xtrain (np.array): Dữ liệu huấn luyện đã chuẩn hóa.
        ytrain (np.array): Nhãn huấn luyện.
        Xval (np.array): Dữ liệu xác thực đã chuẩn hóa.
        yval (np.array): Nhãn xác thực.
        patch_size (tuple): Kích thước (chiều rộng, chiều cao) của mỗi bản vá hình ảnh.
    Returns:
        tuple: (best_svm_model, W, b), mô hình SVM tốt nhất, siêu mặt phẳng W và độ lệch b.
    """
    print("\n--- Bắt đầu Phần 2: Phân loại SVM ---")

    # Xác định các giá trị C để thử nghiệm
    C_values = [0.001, 0.01, 0.1, 1, 10]
    best_accuracy = -1
    best_svm_model = None
    W = None
    b = None

    accuracy_history_train = []
    accuracy_history_val = []

    for c in C_values:
        print(f"Đang huấn luyện SVM với C = {c}...")
        clf = svm.SVC(kernel='linear', C=c, random_state=42)
        clf.fit(Xtrain, ytrain)

        # Tính toán độ chính xác trên tập huấn luyện và xác thực
        y_pred_train = clf.predict(Xtrain)
        acc_train = accuracy_score(ytrain, y_pred_train)
        accuracy_history_train.append(acc_train)

        y_pred_val = clf.predict(Xval)
        acc_val = accuracy_score(yval, y_pred_val)
        accuracy_history_val.append(acc_val)

        print(f"  Độ chính xác trên tập huấn luyện: {acc_train:.4f}")
        print(f"  Độ chính xác trên tập xác thực: {acc_val:.4f}")

        # Chọn mô hình tốt nhất dựa trên độ chính xác trên tập xác thực
        if acc_val > best_accuracy:
            best_accuracy = acc_val
            best_svm_model = clf
            W = clf.coef_[0]
            b = clf.intercept_[0]

    print(f"\nGiá trị C tốt nhất: {best_svm_model.C} với độ chính xác trên tập xác thực: {best_accuracy:.4f}")

    # Trực quan hóa W (lưu ý: W ở đây là các trọng số cho đặc trưng HOG, không còn là pixel ảnh trực tiếp)
    # Việc trực quan hóa W từ đặc trưng HOG phức tạp hơn một chút, nên chúng ta sẽ bỏ qua bước này hoặc cần một cách xử lý khác.
    # Để đơn giản, ở đây chúng ta sẽ chỉ in ra thông báo rằng W là từ HOG.
    print(f"W là một vector trọng số cho các đặc trưng HOG có kích thước: {W.shape}")
    # Nếu muốn trực quan hóa W, có thể thử ánh xạ ngược lại hoặc tìm cách biểu diễn khác.
    # Ví dụ: plt.imshow(W.reshape(kích_thước_phù_hợp_với_HOG_descriptor))
    # Tuy nhiên, điều này không trực quan như khi W là pixel ảnh.

    print("--- Kết thúc Phần 2 ---")
    return best_svm_model, W, b


# --- Phần 3: Phát hiện khuôn mặt ---

def sliding_window(image, step_size, window_size):
    """
    Tạo các cửa sổ trượt qua hình ảnh.
    Args:
        image (np.array): Hình ảnh đầu vào.
        step_size (int): Kích thước bước để di chuyển cửa sổ.
        window_size (tuple): Kích thước (chiều rộng, chiều cao) của cửa sổ.
    Yields:
        tuple: (x, y, window), tọa độ góc trên bên trái và cửa sổ hình ảnh.
    """
    for y in range(0, image.shape[0] - window_size[1] + 1, step_size):
        for x in range(0, image.shape[1] - window_size[0] + 1, step_size):
            yield (x, y, image[y:y + window_size[1], x:x + window_size[0]])


def non_maxima_suppression(boxes, scores, overlap_thresh):
    """
    Thực hiện Non-maxima suppression (NMS) để lọc các hộp giới hạn chồng chéo.
    Args:
        boxes (np.array): Mảng các hộp giới hạn [[x1, y1, x2, y2], ...].
        scores (np.array): Mảng các điểm tin cậy tương ứng.
        overlap_thresh (float): Ngưỡng chồng chéo.
    Returns:
        np.array: Các hộp giới hạn đã lọc.
    """
    if len(boxes) == 0:
        return []

    # Nếu hộp giới hạn không phải là kiểu float, hãy chuyển đổi chúng
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")

    pick = []
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    # Tính diện tích của các hộp giới hạn và sắp xếp các chỉ số theo điểm tin cậy
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(scores)  # Sắp xếp theo điểm tin cậy, từ thấp đến cao

    # Lặp lại trong khi vẫn còn các chỉ số trong danh sách
    while len(idxs) > 0:
        # Lấy chỉ số cuối cùng trong danh sách (hộp có điểm cao nhất) và thêm nó vào danh sách các lựa chọn
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        # Tìm tọa độ (x, y) lớn nhất và (x, y) nhỏ nhất cho vùng chồng chéo
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        # Tính toán chiều rộng và chiều cao của vùng chồng chéo
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        # Tính toán tỷ lệ chồng chéo (Intersection over Union)
        overlap = (w * h) / area[idxs[:last]]

        # Xóa tất cả các chỉ số có tỷ lệ chồng chéo lớn hơn ngưỡng
        idxs = np.delete(idxs, np.concatenate(([last], np.where(overlap > overlap_thresh)[0])))

    # Trả về các hộp giới hạn đã chọn
    return boxes[pick].astype("int")


def detect_faces(image_path, svm_model, mean, std, patch_size, output_folder, conf_thresh=0.5, nms_thresh=0.3):
    """
    Sử dụng mô hình SVM với đặc trưng HOG để phát hiện khuôn mặt trong hình ảnh.
    Args:
        image_path (str): Đường dẫn đến hình ảnh cần phát hiện.
        svm_model (sklearn.svm.SVC): Mô hình SVM đã huấn luyện.
        mean (np.array): Giá trị trung bình được sử dụng để chuẩn hóa đặc trưng HOG.
        std (np.array): Độ lệch chuẩn được sử dụng để chuẩn hóa đặc trưng HOG.
        patch_size (tuple): Kích thước (chiều rộng, chiều cao) của bản vá hình ảnh.
        output_folder (str): Thư mục để lưu ảnh kết quả.
        conf_thresh (float): Ngưỡng tin cậy để chọn bản vá.
        nms_thresh (float): Ngưỡng NMS để lọc hộp giới hạn.
    Returns:
        np.array: Hình ảnh với các hộp giới hạn khuôn mặt được vẽ.
    """
    print(f"\n--- Bắt đầu Phần 3: Phát hiện khuôn mặt trong ảnh: {image_path} (Với HOG) ---")
    if not os.path.exists(image_path):
        print(f"Lỗi: Không tìm thấy ảnh {image_path}. Vui lòng kiểm tra đường dẫn.")
        return None

    # Tải và chuyển đổi ảnh sang ảnh xám
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print(f"Lỗi: Không thể tải ảnh {image_path}.")
        return None

    original_image_display = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)  # Để hiển thị màu

    (winW, winH) = patch_size
    all_detections = []  # (x1, y1, x2, y2, score)

    print(f"Đang quét ảnh với cửa sổ trượt kích thước {winW}x{winH} và bước {SLIDING_WINDOW_STEP}...")

    # Dùng cửa sổ trượt để trích xuất các bản vá, trích xuất HOG và phân loại
    for (x, y, window) in sliding_window(image, step_size=SLIDING_WINDOW_STEP, window_size=(winW, winH)):
        # Đảm bảo kích thước cửa sổ đúng
        if window.shape[0] != winH or window.shape[1] != winW:
            continue

        # Trích xuất đặc trưng HOG từ bản vá
        hog_features = extract_hog_features(window, patch_size)

        # Chuẩn hóa đặc trưng HOG (sử dụng mean và std từ tập huấn luyện)
        hog_features_normalized = (hog_features - mean) / std

        # Lấy điểm tin cậy từ mô hình SVM (khoảng cách đến siêu mặt phẳng)
        score = svm_model.decision_function(hog_features_normalized.reshape(1, -1))[0]

        # Nếu điểm tin cậy vượt quá ngưỡng, lưu lại phát hiện
        if score > conf_thresh:
            all_detections.append((x, y, x + winW, y + winH, score))

    print(f"Tìm thấy {len(all_detections)} phát hiện HOG ban đầu.")

    if not all_detections:
        print("Không tìm thấy khuôn mặt nào với ngưỡng tin cậy đã cho.")
        plt.imshow(original_image_display)
        plt.title(
            f"Phát hiện khuôn mặt trong {os.path.basename(image_path)} (Không tìm thấy, Conf={conf_thresh}, NMS={nms_thresh})")
        plt.axis('off')
        plt.show()

        # Lưu ảnh gốc vào thư mục output ngay cả khi không có phát hiện
        output_filename = os.path.join(output_folder,
                                       f"{os.path.basename(image_path).split('.')[0]}_no_faces_detected.jpg")
        cv2.imwrite(output_filename, cv2.cvtColor(original_image_display, cv2.COLOR_RGB2BGR))
        print(f"Đã lưu ảnh '{os.path.basename(output_filename)}' vào thư mục '{output_folder}'.")

        print("--- Kết thúc Phần 3 ---")
        return original_image_display

    # Chuyển đổi sang mảng numpy để NMS xử lý
    boxes = np.array([(d[0], d[1], d[2], d[3]) for d in all_detections])
    scores = np.array([d[4] for d in all_detections])

    # Áp dụng NMS để lọc các hộp chồng chéo
    selected_boxes = non_maxima_suppression(boxes, scores, nms_thresh)

    print(f"Số lượng khuôn mặt sau NMS: {len(selected_boxes)}")

    # Hiển thị kết quả trên ảnh gốc
    output_image = original_image_display.copy()
    for (startX, startY, endX, endY) in selected_boxes:
        cv2.rectangle(output_image, (startX, startY), (endX, endY), (0, 255, 0), 1)  # Vẽ hộp màu xanh lá

    plt.figure(figsize=(10, 8))
    plt.imshow(output_image)
    plt.title(f"Phát hiện khuôn mặt trong {os.path.basename(image_path)} (Conf={conf_thresh}, NMS={nms_thresh})")
    plt.axis('off')
    plt.show()

    # Lưu ảnh kết quả vào thư mục output
    output_filename = os.path.join(output_folder, f"{os.path.basename(image_path).split('.')[0]}_detected.jpg")
    cv2.imwrite(output_filename, cv2.cvtColor(output_image, cv2.COLOR_RGB2BGR))
    print(f"Đã lưu ảnh '{os.path.basename(output_filename)}' vào thư mục '{output_folder}'.")

    print("--- Kết thúc Phần 3 ---")
    return output_image


# --- Hàm chính để chạy chương trình ---
def main():
    # CẬP NHẬT: Đảm bảo các đường dẫn này trỏ đến đúng vị trí tệp của bạn.
    # Ví dụ: Nếu possamples.mat nằm trong thư mục gốc, hãy sử dụng 'possamples.mat'
    # Nếu nó nằm trong thư mục 'data', hãy sử dụng 'data/possamples.mat'
    DATA_BASED_PATH = 'data'
    pos_mat_file = os.path.join(DATA_BASED_PATH, 'possamples.mat')
    neg_mat_file = os.path.join(DATA_BASED_PATH, 'negsamples.mat')

    # Thiết lập thư mục đầu ra
    output_folder = 'outputs'
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"Đã tạo thư mục đầu ra: {output_folder}")

    # 1. Chuẩn bị dữ liệu
    Xtrain, ytrain, Xval, yval, mean, std = load_and_prepare_data(pos_mat_file, neg_mat_file, PATCH_SIZE)

    if Xtrain is None:
        print("Không thể tiếp tục do lỗi tải hoặc chuẩn bị dữ liệu.")
        return

    # 2. Huấn luyện và đánh giá SVM
    # Lưu ý: W ở đây là các trọng số cho đặc trưng HOG, không còn là pixel ảnh trực tiếp.
    # Việc trực quan hóa W sẽ cần một cách tiếp cận khác hoặc bỏ qua bước hiển thị ảnh W.
    best_svm_model, W, b = train_and_evaluate_svm(Xtrain, ytrain, Xval, yval, PATCH_SIZE)

    if best_svm_model is None:
        print("Không thể tiếp tục do lỗi huấn luyện SVM.")
        return

    # 3. Phát hiện khuôn mặt trong ảnh test
    # CẬP NHẬT: Đảm bảo các đường dẫn này trỏ đến đúng vị trí tệp của bạn.
    IMG_BASE_PATH = 'images/test'
    test_images = ['img1.jpg', 'img2.jpg', 'img3.jpg', 'img4.jpg', 'img5.jpg']

    # CẬP NHẬT RẤT QUAN TRỌNG: Điều chỉnh các ngưỡng này để cải thiện phát hiện.
    # Bắt đầu bằng cách TĂNG confidence_threshold (ví dụ: lên 1.0, 2.0, 3.0, 4.0, 5.0)
    # cho đến khi chỉ những khuôn mặt rõ ràng được phát hiện.
    confidence_threshold = 1  # Ngưỡng tin cậy cho bản vá (decision_function score)
    nms_threshold = 0.1  # Ngưỡng chồng chéo cho NMS

    print("\n--- Bắt đầu thử nghiệm phát hiện trên các ảnh test (Với HOG) ---")
    for img_file in test_images:
        img_file = os.path.join(IMG_BASE_PATH, img_file)
        if not os.path.exists(img_file):
            print(f"\nCảnh báo: Không tìm thấy file ảnh {img_file}. Bỏ qua.")
            continue
        detect_faces(img_file, best_svm_model, mean, std, PATCH_SIZE,
                     output_folder, conf_thresh=confidence_threshold, nms_thresh=nms_threshold)
    print("--- Kết thúc thử nghiệm phát hiện ---")


if __name__ == "__main__":
    main()
