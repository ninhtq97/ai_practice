import cv2
import numpy as np
import sys
import argparse
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="SURF-based object tracking in video")
    parser.add_argument("--hessian", type=int, default=200, help="SURF Hessian threshold (lower = more keypoints, default 200)")
    parser.add_argument("--min-matches", type=int, default=6, help="Minimum matches to consider object found (default 6)")
    parser.add_argument("--ratio", type=float, default=0.75, help="Lowe's ratio test threshold (default 0.75)")
    parser.add_argument("--use-affine", action="store_true", help="Use Affine transform instead of Homography (better for small objects)")
    parser.add_argument("--smooth", type=float, default=0.3, help="Bbox smoothing factor 0-0.9 (default 0.3)")
    parser.add_argument("--template-update", type=float, default=0.0, help="Template update rate 0-0.3 (0=disabled, helps adapt to changes)")
    parser.add_argument("--geometric-check", action="store_true", help="Apply geometric constraints to filter bad matches")
    parser.add_argument("--median-filter", action="store_true", help="Use median filtering on bbox for stability")
    parser.add_argument("--clahe", action="store_true", help="Apply CLAHE to handle illumination changes")
    parser.add_argument("--debug", action="store_true", help="Show debug info (keypoints, matches)")
    parser.add_argument("--save", type=str, default=None, help="Save output video to this path")
    parser.add_argument("--redetect-interval", type=int, default=0, help="Re-detect ROI features every N frames (0=disabled)")
    return parser.parse_args()


def clamp_bbox(bbox, frame_shape):
    """Đảm bảo bbox nằm trong khung hình."""
    x, y, w, h = bbox
    x = max(0, x)
    y = max(0, y)
    w = min(w, frame_shape[1] - x)
    h = min(h, frame_shape[0] - y)
    return (x, y, w, h) if w > 10 and h > 10 else None


def apply_bbox_filtering(new_bbox, bbox_history, roi_bbox_smooth, args):
    """Áp dụng median filter và smoothing cho bbox."""
    # Median filtering
    if args.median_filter:
        bbox_history.append(new_bbox)
        if len(bbox_history) > 5:
            bbox_history.pop(0)
        if len(bbox_history) >= 3:
            x_med = int(np.median([b[0] for b in bbox_history]))
            y_med = int(np.median([b[1] for b in bbox_history]))
            w_med = int(np.median([b[2] for b in bbox_history]))
            h_med = int(np.median([b[3] for b in bbox_history]))
            new_bbox = (x_med, y_med, w_med, h_med)

    # Smoothing
    if roi_bbox_smooth is not None and args.smooth > 0:
        alpha = max(0.0, min(0.9, args.smooth))
        x_new = int((1-alpha) * new_bbox[0] + alpha * roi_bbox_smooth[0])
        y_new = int((1-alpha) * new_bbox[1] + alpha * roi_bbox_smooth[1])
        w_new = int((1-alpha) * new_bbox[2] + alpha * roi_bbox_smooth[2])
        h_new = int((1-alpha) * new_bbox[3] + alpha * roi_bbox_smooth[3])
        new_bbox = (x_new, y_new, w_new, h_new)

    return new_bbox


def update_template(frame_gray, roi_bbox, surf, kp_roi, des_roi, args, frame_count, inlier_ratio):
    """Cập nhật template nếu điều kiện thỏa mãn."""
    if args.template_update > 0 and inlier_ratio > 0.5 and frame_count % 10 == 0:
        x, y, w, h = roi_bbox
        if w > 10 and h > 10:
            new_roi_gray = frame_gray[y:y+h, x:x+w]
            new_kp, new_des = surf.detectAndCompute(new_roi_gray, None)
            if new_des is not None and len(new_kp) >= len(kp_roi) * 0.7:
                return new_roi_gray, new_kp, new_des
    return None, None, None


def draw_tracking_status(display_frame, inlier_count, total_matches, inlier_ratio, color):
    """Vẽ thông tin tracking."""
    status = "Tracking" if inlier_ratio >= 0.3 else "Low quality"
    cv2.putText(display_frame, f"{status}: {inlier_count}/{total_matches} ({inlier_ratio:.1%})",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)


def main():
    args = parse_args()

    # 1. Khởi tạo Camera (webcam)
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print(f"Lỗi: Không thể mở camera")
        sys.exit()

    print("HƯỚNG DẪN (SURF Object Tracking):")
    print("- Nhấn phím 's' để chọn vùng đối tượng cần theo dõi (ROI).")
    print("- Kéo chuột để chọn vùng, sau đó nhấn SPACE hoặc ENTER để xác nhận.")
    print("- Nhấn 'r' để tái phát hiện đặc trưng trong ROI hiện tại.")
    print("- Nhấn 'q' để thoát chương trình.")

    # Biến lưu trữ ảnh mẫu (template) và keypoints/descriptors của nó
    roi_gray = None
    roi_bbox = None  # Lưu bbox gốc (x, y, w, h)
    roi_bbox_smooth = None  # Bbox sau smoothing
    bbox_history = []  # Lịch sử bbox cho median filter
    kp_roi = None
    des_roi = None
    is_tracking = False
    frame_count = 0

    # Video writer
    writer = None
    if args.save:
        out_path = Path(args.save)
        out_path.parent.mkdir(parents=True, exist_ok=True)

    # CLAHE cho thay đổi ánh sáng
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)) if args.clahe else None

    # Khởi tạo thuật toán SURF
    try:
        surf = cv2.xfeatures2d.SURF_create(args.hessian)
        print(f"✓ SURF initialized (hessian={args.hessian})")
    except AttributeError:
        print("\n[LỖI QUAN TRỌNG] SURF không khả dụng trong phiên bản OpenCV này.")
        print("SURF là thuật toán có bản quyền. Hãy cài đặt 'opencv-contrib-python' hoặc build OpenCV với OPENCV_ENABLE_NONFREE=ON.")
        sys.exit()

    # Cấu hình FLANN Matcher (Dùng để so khớp đặc trưng nhanh)
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    MIN_MATCH_COUNT = args.min_matches

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Không nhận được tín hiệu hình ảnh hoặc kết thúc video.")
            break

        frame_count += 1

        # Khởi tạo writer nếu cần
        if writer is None and args.save:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
            h, w = frame.shape[:2]
            writer = cv2.VideoWriter(args.save, fourcc, fps, (w, h))

        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Apply CLAHE nếu được bật
        if clahe is not None:
            frame_gray = clahe.apply(frame_gray)

        display_frame = frame.copy()

        # Tái phát hiện đặc trưng định kỳ nếu được cấu hình
        if (is_tracking and des_roi is not None and
            args.redetect_interval > 0 and
            frame_count % args.redetect_interval == 0 and
            roi_bbox is not None):
            x, y, w, h = roi_bbox
            roi_gray = frame_gray[y:y+h, x:x+w]
            kp_roi, des_roi = surf.detectAndCompute(roi_gray, None)
            if des_roi is not None:
                print(f"Frame {frame_count}: Re-detected {len(kp_roi)} keypoints in ROI")

        # Nếu đang ở chế độ tracking
        if is_tracking and des_roi is not None:
            # 2. Tìm đặc trưng SURF trên khung hình hiện tại
            kp_frame, des_frame = surf.detectAndCompute(frame_gray, None)

            if des_frame is not None and len(kp_frame) > 0:
                # 3. So khớp đặc trưng giữa ROI mẫu và khung hình hiện tại
                matches = flann.knnMatch(des_roi, des_frame, k=2)

                # 4. Lọc các điểm khớp tốt (Good Matches) theo tỷ lệ Lowe's ratio test
                good_matches = []
                for match_pair in matches:
                    if len(match_pair) == 2:
                        m, n = match_pair
                        if m.distance < args.ratio * n.distance:
                            good_matches.append(m)

                # 4.5. Geometric verification - lọc matches dựa trên khoảng cách và góc
                if args.geometric_check and len(good_matches) > 10:
                    filtered_matches = []
                    distances = [m.distance for m in good_matches]
                    median_dist = np.median(distances)
                    std_dist = np.std(distances)

                    for m in good_matches:
                        # Chỉ giữ matches có distance gần median (loại outliers)
                        if abs(m.distance - median_dist) < 2.0 * std_dist:
                            filtered_matches.append(m)

                    if len(filtered_matches) >= args.min_matches:
                        good_matches = filtered_matches

                # 5. Nếu đủ số lượng điểm khớp tốt, tiến hành vẽ khung bao quanh
                if len(good_matches) >= MIN_MATCH_COUNT:
                    # Lấy tọa độ các điểm khớp từ mẫu và từ khung hình
                    src_pts = np.float32([kp_roi[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                    dst_pts = np.float32([kp_frame[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

                    # Tính ma trận Homography hoặc Affine
                    if args.use_affine:
                        M, inliers = cv2.estimateAffinePartial2D(
                            src_pts, dst_pts,
                            method=cv2.RANSAC,
                            ransacReprojThreshold=2.0,  # Nghiêm ngặt hơn
                            maxIters=2000,
                            confidence=0.995
                        )
                        if M is not None:
                            # Chuyển từ 2x3 sang 3x3 để dùng perspectiveTransform
                            M = np.vstack([M, [0, 0, 1]])
                    else:
                        M, inliers = cv2.findHomography(
                            src_pts, dst_pts,
                            cv2.RANSAC,
                            ransacThreshold=3.0,
                            maxIters=2000,
                            confidence=0.995
                        )

                    if M is not None:
                        # Kiểm tra chất lượng transform (inlier ratio)
                        inlier_count = int(inliers.sum()) if inliers is not None and hasattr(inliers, 'sum') else len(good_matches)
                        inlier_ratio = inlier_count / len(good_matches) if len(good_matches) > 0 else 0

                        h, w = roi_gray.shape
                        pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)

                        # Biến đổi phối cảnh để lấy khung bao mới
                        dst = cv2.perspectiveTransform(pts, M)

                        # Vẽ khung bao quanh đối tượng (màu thay đổi theo chất lượng)
                        color = (0, 255, 0) if inlier_ratio >= 0.3 else (0, 165, 255)  # Xanh lá nếu tốt, cam nếu yếu
                        cv2.polylines(display_frame, [np.int32(dst)], True, color, 3, cv2.LINE_AA)

                        # Cập nhật roi_bbox dựa trên dst
                        dst_pts = dst.reshape(-1, 2)
                        x_new = int(dst_pts[:, 0].min())
                        y_new = int(dst_pts[:, 1].min())
                        w_new = int(dst_pts[:, 0].max() - x_new)
                        h_new = int(dst_pts[:, 1].max() - y_new)

                        # Clamp và validate bbox
                        new_bbox = clamp_bbox((x_new, y_new, w_new, h_new), frame.shape)

                        if new_bbox and inlier_ratio >= 0.3:
                            # Áp dụng filtering chỉ khi chất lượng tốt
                            new_bbox = apply_bbox_filtering(new_bbox, bbox_history, roi_bbox_smooth, args)
                            roi_bbox = new_bbox
                            roi_bbox_smooth = roi_bbox

                            # Template update
                            updated = update_template(frame_gray, roi_bbox, surf, kp_roi, des_roi,
                                                     args, frame_count, inlier_ratio)
                            if updated[0] is not None:
                                roi_gray, kp_roi, des_roi = updated

                        # Vẽ status
                        draw_tracking_status(display_frame, inlier_count, len(good_matches), inlier_ratio, color)

                        # Debug: vẽ các điểm khớp (chỉ inliers)
                        if args.debug:
                            for i, m in enumerate(good_matches):
                                if inliers is not None and i < len(inliers) and inliers[i] == 1:
                                    pt = dst_pts[i].reshape(2)
                                    cv2.circle(display_frame, (int(pt[0]), int(pt[1])), 3, (0, 255, 0), -1)
                    else:
                        cv2.putText(display_frame, "Mat dau (Homography failed)", (10, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                else:
                    cv2.putText(display_frame, f"Mat dau (Not enough matches: {len(good_matches)}/{MIN_MATCH_COUNT})", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                # Vẽ các điểm đặc trưng khớp (tùy chọn - để debug)
                # display_frame = cv2.drawMatches(roi_gray, kp_roi, frame_gray, kp_frame, good_matches, None, flags=2)

        else:
            cv2.putText(display_frame, "Nhan 's' de chon doi tuong", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.imshow('SURF Object Tracking', display_frame)

        # Ghi video nếu được bật
        if writer is not None:
            writer.write(display_frame)

        key = cv2.waitKey(1) & 0xFF

        # Xử lý phím tắt
        if key == ord('q'):
            break
        elif key == ord('s'):
            # Chọn vùng ROI
            bbox = cv2.selectROI('SURF Object Tracking', frame, fromCenter=False, showCrosshair=True)
            try:
                cv2.destroyWindow('ROI selector')
            except Exception:
                pass

            if bbox[2] > 0 and bbox[3] > 0:  # Đảm bảo vùng chọn hợp lệ
                x, y, w, h = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
                roi_bbox = (x, y, w, h)
                roi_gray = frame_gray[y:y+h, x:x+w]

                # Tính toán đặc trưng cho mẫu (Template) ngay khi chọn xong
                kp_roi, des_roi = surf.detectAndCompute(roi_gray, None)

                if des_roi is not None and len(kp_roi) >= 4:
                    is_tracking = True
                    frame_count = 0
                    roi_bbox_smooth = roi_bbox
                    bbox_history = []
                    print(f"✓ Đã chọn đối tượng với {len(kp_roi)} keypoints. Bắt đầu tracking...")
                    print(f"  Hessian: {args.hessian}, Min-matches: {args.min_matches}, Ratio: {args.ratio}")
                    print(f"  Mode: {'Affine' if args.use_affine else 'Homography'}, Smoothing: {args.smooth}")
                    if args.geometric_check:
                        print(f"  ✓ Geometric verification enabled")
                    if args.median_filter:
                        print(f"  ✓ Median filtering enabled")
                    if args.template_update > 0:
                        print(f"  ✓ Template update: {args.template_update}")
                else:
                    print(f"✗ Vùng chọn chỉ có {len(kp_roi) if kp_roi else 0} keypoints (cần >=4).")
                    print(f"  Tip: Thử giảm --hessian (hiện tại: {args.hessian})")
        elif key == ord('r'):
            # Tái phát hiện đặc trưng trong ROI hiện tại
            if is_tracking and roi_bbox is not None:
                validated_bbox = clamp_bbox(roi_bbox, frame.shape)
                if validated_bbox:
                    x, y, w, h = validated_bbox
                    roi_gray = frame_gray[y:y+h, x:x+w]
                    kp_roi, des_roi = surf.detectAndCompute(roi_gray, None)
                    if des_roi is not None:
                        print(f"Tái phát hiện: {len(kp_roi)} keypoints trong ROI hiện tại")
    cap.release()
    if writer is not None:
        writer.release()
        print(f"✓ Đã lưu video: {args.save}")
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()