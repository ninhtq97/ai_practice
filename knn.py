from math import ceil

import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier


# --- Hàm tự động gợi ý n_splits ---
def suggest_n_splits(X, y, k_max_desired, max_splits=10):
    """
    Gợi ý n_splits lớn nhất sao cho mỗi lớp có đủ mẫu để chia,
    và số mẫu train tối thiểu ≥ k_max_desired
    """
    n_samples = len(X)
    _, class_counts = np.unique(y, return_counts=True)
    max_possible_splits = min(min(class_counts), n_samples, max_splits)

    for n_splits in range(max_possible_splits, 1, -1):
        max_test_size = ceil(n_samples / n_splits)
        min_train_size = n_samples - max_test_size
        if min_train_size >= k_max_desired:
            return n_splits
    return 2


# --- Hàm tính k tối đa thật sự có thể dùng (dù đã chọn n_splits hợp lý) ---
def compute_k_max(n_samples, n_splits):
    max_test_size = ceil(n_samples / n_splits)
    return n_samples - max_test_size


# --- Pipeline đầy đủ ---
def knn_cv_pipeline(X, y, k_max_desired=10):
    n_samples = len(X)

    # 1. Gợi ý n_splits phù hợp
    n_splits = suggest_n_splits(X, y, k_max_desired)
    print(f"🔁 Dùng n_splits = {n_splits}")

    # 2. Tính lại k_max thật sự an toàn theo n_splits đó
    k_max = compute_k_max(n_samples, n_splits)
    print(f"✅ Dải K dùng được: từ 1 đến {k_max}")

    # 3. Chạy cross-validation cho từng K
    k_values = list(range(1, k_max + 1))
    cv_scores = []
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    for k in k_values:
        knn = KNeighborsClassifier(n_neighbors=k)
        scores = cross_val_score(knn, X, y, cv=cv, scoring='accuracy')
        cv_scores.append(scores.mean())

    # 4. Vẽ biểu đồ
    plt.plot(k_values, cv_scores, marker='o')
    plt.xlabel('K (số hàng xóm)')
    plt.ylabel('Độ chính xác (CV)')
    plt.title('Chọn K tốt nhất bằng cross-validation')
    plt.grid(True)
    plt.show()

    # 5. In ra K tốt nhất
    best_k = k_values[np.argmax(cv_scores)]
    print(f"🏆 K tốt nhất là: {best_k}")

    return best_k, cv_scores


X = np.array([
    [160, 50], [170, 70], [165, 55], [180, 80],
    [158, 48], [175, 75], [162, 52], [167, 66], [163, 58]
])
y = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0])

knn_cv_pipeline(X, y, k_max_desired=5)
