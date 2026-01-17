"""
YOLO + SURF Tracking (Fast)
- YOLO detection mỗi N frame (default 5)
- SURF tracking liên tục
- Tối ưu: mịn, nhanh, không crash
"""
import cv2
import numpy as np
import sys
import argparse
import time

try:
    from ultralytics import YOLO
except ImportError:
    print("❌ YOLOv8 chưa cài. Chạy: pip install ultralytics")
    sys.exit()


def parse_args():
    parser = argparse.ArgumentParser(description="YOLO + SURF Fast Tracking")
    parser.add_argument("--camera", type=int, default=0, help="Camera index")
    parser.add_argument("--video", type=str, default=None, help="Video path")
    parser.add_argument("--yolo-model", type=str, default="yolo11n.pt", help="YOLO model")
    parser.add_argument("--yolo-skip", type=int, default=5, help="Run YOLO every N frames")
    parser.add_argument("--classes", type=str, default="car,person,truck,bus", help="Classes")
    parser.add_argument("--conf", type=float, default=0.4, help="YOLO confidence")
    parser.add_argument("--hessian", type=int, default=120, help="SURF Hessian (lower=more features)")
    parser.add_argument("--resize-width", type=int, default=640, help="Resize width (0=no resize)")
    parser.add_argument("--nms", type=float, default=0.25, help="NMS threshold (lower=stricter filtering)")
    parser.add_argument("--max-age", type=int, default=30, help="Frames to keep dead tracker")
    parser.add_argument("--roi", type=str, default="0,0.4,1,1", help="ROI as 'x1,y1,x2,y2' in 0-1 (normalized) or pixels")
    parser.add_argument("--target-fps", type=float, default=15.0, help="Target playback FPS (0=no limit)")
    parser.add_argument("--save", type=str, default=None, help="Save output")
    return parser.parse_args()


class Tracker:
    def __init__(self, tid, roi_gray, surf):
        self.tid = tid
        self.roi_gray = roi_gray
        self.kp, self.des = surf.detectAndCompute(roi_gray, None)
        self.surf = surf
        self.bbox = None
        self.bbox_history = []  # Last 3 bbox for smoothing
        self.age = 0
        self.flann = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=30))

    def is_valid(self):
        return self.des is not None and len(self.kp) >= 4

    def update_template(self, roi_gray):
        """Fast template update."""
        if roi_gray.shape[0] > 10 and roi_gray.shape[1] > 10:
            self.roi_gray = roi_gray
            self.kp, self.des = self.surf.detectAndCompute(roi_gray, None)

    def predict(self, frame_gray):
        """Fast SURF prediction."""
        if not self.is_valid():
            return None

        kp_f, des_f = self.surf.detectAndCompute(frame_gray, None)
        if des_f is None or len(kp_f) < 4:
            return None

        try:
            matches = self.flann.knnMatch(self.des, des_f, k=2)
            good = [pair[0] for pair in matches if len(pair) == 2 and pair[0].distance < 0.7 * pair[1].distance]

            if len(good) < 4:
                return None

            src = np.float32([self.kp[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
            dst = np.float32([kp_f[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

            M, _ = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC,
                                               ransacReprojThreshold=1.0, confidence=0.95)
            if M is None:
                return None

            M = np.vstack([M, [0, 0, 1]])
            h, w = self.roi_gray.shape
            pts = np.float32([[0, 0], [0, h-1], [w-1, h-1], [w-1, 0]]).reshape(-1, 1, 2)
            dst_pts = cv2.perspectiveTransform(pts, M).reshape(-1, 2)

            x = int(dst_pts[:, 0].min())
            y = int(dst_pts[:, 1].min())
            bw = int(dst_pts[:, 0].max() - x)
            bh = int(dst_pts[:, 1].max() - y)

            if bw > 15 and bh > 15:
                return (max(0, x), max(0, y), bw, bh)
        except:
            pass

        return None

    def smooth_and_update(self, new_bbox):
        """Smooth bbox with history."""
        self.bbox_history.append(new_bbox)
        if len(self.bbox_history) > 3:
            self.bbox_history.pop(0)

        # Simple median filter
        if len(self.bbox_history) >= 3:
            xs = sorted([b[0] for b in self.bbox_history])
            ys = sorted([b[1] for b in self.bbox_history])
            ws = sorted([b[2] for b in self.bbox_history])
            hs = sorted([b[3] for b in self.bbox_history])
            smoothed = (xs[1], ys[1], ws[1], hs[1])
        else:
            smoothed = new_bbox

        self.bbox = smoothed
        return smoothed


def iou(b1, b2):
    x1, y1, w1, h1 = b1
    x2, y2, w2, h2 = b2
    xi1, yi1 = max(x1, x2), max(y1, y2)
    xi2, yi2 = min(x1+w1, x2+w2), min(y1+h1, y2+h2)
    intersection = max(0, xi2-xi1) * max(0, yi2-yi1)
    union = w1*h1 + w2*h2 - intersection
    return intersection / union if union > 0 else 0


def nms(dets, thresh=0.4):
    if len(dets) == 0:
        return []
    dets = sorted(dets, key=lambda x: x['conf'], reverse=True)
    keep = []
    while dets:
        best = dets.pop(0)
        keep.append(best)
        dets = [d for d in dets if iou(best['bbox'], d['bbox']) < thresh]
    return keep


def parse_roi(roi_str, frame_h, frame_w):
    """Parse ROI string to pixel coordinates.
    Format: 'x1,y1,x2,y2' (normalized 0-1) or 'x1,y1,x2,y2' (pixels if > 1)
    """
    if roi_str is None:
        return None
    try:
        parts = [float(x) for x in roi_str.split(',')]
        x1, y1, x2, y2 = parts
        # If all values <= 1, assume normalized; else assume pixels
        if x1 <= 1 and y1 <= 1 and x2 <= 1 and y2 <= 1:
            x1, y1, x2, y2 = int(x1*frame_w), int(y1*frame_h), int(x2*frame_w), int(y2*frame_h)
        else:
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        return (x1, y1, x2, y2)
    except:
        return None


def is_in_roi(bbox, roi):
    """Check if bbox center is within ROI."""
    if roi is None:
        return True
    x, y, w, h = bbox
    cx, cy = x + w//2, y + h//2
    x1, y1, x2, y2 = roi
    return x1 <= cx <= x2 and y1 <= cy <= y2


def main():
    args = parse_args()

    print(f"Loading {args.yolo_model}...")
    yolo = YOLO(args.yolo_model)
    class_names = set(args.classes.split(','))

    cap = cv2.VideoCapture(args.video if args.video else args.camera)
    if not cap.isOpened():
        print("❌ Cannot open video")
        return 1

    try:
        surf = cv2.xfeatures2d.SURF_create(args.hessian)
    except:
        print("❌ SURF not available")
        return 1

    trackers = {}
    next_id = 1
    frame_idx = 0
    writer = None
    roi = None

    fps_time = time.time()
    fps = 0
    frame_time = 1.0 / args.target_fps if args.target_fps > 0 else 0

    print("=" * 50)
    print(f"YOLO+SURF Fast | Model: {args.yolo_model}")
    print(f"Classes: {args.classes}")
    print(f"SURF: hessian={args.hessian}")
    print(f"YOLO interval: every {args.yolo_skip} frames")
    print("=" * 50)

    while True:
        frame_start_time = time.time()  # Start timing this frame

        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1

        # Resize
        if args.resize_width and frame.shape[1] != args.resize_width:
            scale = args.resize_width / frame.shape[1]
            frame = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)

        # Init writer
        if writer is None and args.save:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            fps_vid = cap.get(cv2.CAP_PROP_FPS) or 25.0
            h, w = frame.shape[:2]
            writer = cv2.VideoWriter(args.save, fourcc, fps_vid, (w, h))

        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        disp = frame.copy()

        # Parse ROI on first frame
        if frame_idx == 1:
            roi = parse_roi(args.roi, frame.shape[0], frame.shape[1])
            if roi:
                x1, y1, x2, y2 = roi
                cv2.rectangle(disp, (x1, y1), (x2, y2), (255, 255, 0), 1)  # Draw ROI region

        # YOLO detection every N frames
        doing_yolo = (frame_idx % args.yolo_skip == 1)
        dets = []

        if doing_yolo:
            results = yolo(frame, conf=args.conf, verbose=False)
            for r in results:
                if r.boxes is None:
                    continue
                for box in r.boxes:
                    cls_name = yolo.names[int(box.cls[0])]
                    if cls_name not in class_names:
                        continue
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    dets.append({
                        'bbox': (x1, y1, x2-x1, y2-y1),
                        'class': cls_name,
                        'conf': float(box.conf[0])
                    })

            # NMS + ROI filtering
            dets_before = len(dets)
            dets = nms(dets, args.nms)
            dets = [d for d in dets if is_in_roi(d['bbox'], roi)]

            if frame_idx % 30 == 1:
                print(f"F{frame_idx}: YOLO {dets_before} -> NMS+ROI {len(dets)}")

        # Update trackers: SURF predict or YOLO refresh
        if not doing_yolo and len(trackers) > 0:
            # SURF tracking on inter-YOLO frames
            updated = {}
            for tid, t in trackers.items():
                pred = t.predict(frame_gray)
                if pred:
                    t.smooth_and_update(pred)
                    t.age += 1
                    updated[tid] = t
                elif t.age < args.max_age:
                    t.age += 1
                    updated[tid] = t
            trackers = updated

        elif doing_yolo:
            # Create new trackers from YOLO detections
            trackers = {}
            for det in dets:
                x, y, w, h = det['bbox']
                if w > 15 and h > 15:
                    roi_img = frame_gray[max(0,y):min(frame_gray.shape[0],y+h),
                                        max(0,x):min(frame_gray.shape[1],x+w)]
                    t = Tracker(next_id, roi_img, surf)
                    if t.is_valid():
                        t.smooth_and_update(det['bbox'])
                        trackers[next_id] = t
                        next_id += 1

        # Draw all active trackers
        for tid, t in trackers.items():
            if t.bbox:
                x, y, w, h = t.bbox
                cv2.rectangle(disp, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(disp, f"T{tid}", (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # Draw ROI visualization
        if roi:
            x1, y1, x2, y2 = roi
            cv2.rectangle(disp, (x1, y1), (x2, y2), (0, 255, 255), 3)
            corner_len = 30
            for cx, cy in [(x1, y1), (x2, y1), (x1, y2), (x2, y2)]:
                cv2.line(disp, (cx-corner_len, cy), (cx+corner_len, cy), (0, 255, 255), 2)
                cv2.line(disp, (cx, cy-corner_len), (cx, cy+corner_len), (0, 255, 255), 2)
            cv2.putText(disp, "ROI", (x1+5, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        # Info
        if frame_idx % 60 == 0:
            fps = 60 / (time.time() - fps_time + 0.001)
            fps_time = time.time()

        info = f"Frm {frame_idx} | Track: {len(trackers)} | FPS {fps:.1f}"
        cv2.putText(disp, info, (5, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

        cv2.imshow("YOLO+SURF", disp)
        if writer:
            writer.write(disp)

        # Control frame rate to maintain constant speed
        if args.target_fps > 0:
            elapsed = time.time() - frame_start_time
            delay = max(1, int((frame_time - elapsed) * 1000))  # Convert to ms, min 1ms
            key = cv2.waitKey(delay) & 0xFF
        else:
            key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    sys.exit(main())
