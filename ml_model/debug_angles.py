import sys, os
# Ensure project root is on sys.path so 'ml_model' package is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ml_model.yolo_fightingpose_detection import ZonePoseDetector
import cv2

print('Starting headless debug capture (will run 30 frames)')
detector = ZonePoseDetector()
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print('ERROR: Could not open camera')
    exit(1)

frame_count = 0
try:
    while frame_count < 30:
        ret, frame = cap.read()
        if not ret:
            print('ERROR: failed to read frame')
            break
        frame_count += 1
        try:
            annotated, pose, angles = detector.process_frame(frame)
        except Exception as e:
            print('process_frame error:', e)
            break

        # Get raw keypoints via model inference directly for debugging
        kps = None
        try:
            res = None
            model_obj = getattr(detector, 'model', None)
            if model_obj is not None and callable(model_obj):
                try:
                    # try calling with verbose if supported
                    out = model_obj(frame, verbose=False)
                except TypeError:
                    # some model wrappers don't accept verbose kwarg
                    out = model_obj(frame)
                # normalize output to a single result if it's a list/tuple
                if isinstance(out, (list, tuple)):
                    res = out[0] if len(out) > 0 else None
                else:
                    res = out
                if res is not None:
                    kp = getattr(res, 'keypoints', None)
                    if kp is not None:
                        try:
                            kps = kp.data[0].cpu().numpy()
                        except Exception:
                            kps = kp.data[0].numpy()
            else:
                # model not available or not callable; skip raw keypoint extraction
                res = None
        except Exception as e:
            kps = None

        print(f'Frame {frame_count}: pose={pose}, angles={angles}')
        if kps is not None:
            idxs = [('L_sh',5),('R_sh',6),('L_el',7),('R_el',8),('L_wr',9),('R_wr',10),('L_hip',11),('R_hip',12)]
            for name,i in idxs:
                if i < len(kps):
                    x,y,c = kps[i]
                    print(f'  {name}: x={x:.1f}, y={y:.1f}, conf={c:.2f}')
        else:
            print('  No keypoints found in raw result')

    print('Headless debug finished')
finally:
    cap.release()
    cv2.destroyAllWindows()
