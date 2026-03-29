import numpy as np
import cv2
from PIL import Image
import io
import base64

# Graceful import for MediaPipe
MEDIAPIPE_AVAILABLE = False
mp_holistic = None
mp_drawing = None

try:
    import mediapipe.solutions.holistic as tmp_holistic
    import mediapipe.solutions.drawing_utils as tmp_drawing
    mp_holistic = tmp_holistic
    mp_drawing = tmp_drawing
    MEDIAPIPE_AVAILABLE = True
except (ImportError, AttributeError):
    try:
        from mediapipe.python.solutions import holistic as tmp_holistic
        from mediapipe.python.solutions import drawing_utils as tmp_drawing
        mp_holistic = tmp_holistic
        mp_drawing = tmp_drawing
        MEDIAPIPE_AVAILABLE = True
    except (ImportError, AttributeError):
        MEDIAPIPE_AVAILABLE = False


def analyze_anatomy(image: Image.Image) -> dict:
    """Detect anatomical structures and flag impossibilities using MediaPipe Holistic."""
    img_cv = cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2BGR)
    h, w, _ = img_cv.shape
    
    anomalies = []
    flags = {
        "arms_count": 0,
        "legs_count": 0,
        "hand_count": 0,
        "total_fingers": 0
    }
    
    if not MEDIAPIPE_AVAILABLE:
        return {
            "is_suspicious": False,
            "anomalies": ["MediaPipe library not available. Anatomical check skipped."],
            "severity": "low",
            "flags": flags,
            "overlay_b64": None,
            "explanation": "MediaPipe module could not be initialized."
        }

    with mp_holistic.Holistic(
        static_image_mode=True, 
        model_complexity=2, 
        enable_segmentation=False, 
        refine_face_landmarks=False
    ) as holistic:
        results = holistic.process(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))
        
        # 1. Pose Analysis (Limbs and Torso Spatial Checks)
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            
            # Count visible limbs
            if landmarks[13].visibility > 0.5 or landmarks[15].visibility > 0.5: flags["arms_count"] += 1
            if landmarks[14].visibility > 0.5 or landmarks[16].visibility > 0.5: flags["arms_count"] += 1
            if landmarks[25].visibility > 0.5 or landmarks[27].visibility > 0.5: flags["legs_count"] += 1
            if landmarks[26].visibility > 0.5 or landmarks[28].visibility > 0.5: flags["legs_count"] += 1

            # Torso Bounds (Shoulders to Hips)
            sh_l, sh_r = landmarks[11], landmarks[12]
            hip_l, hip_r = landmarks[23], landmarks[24]
            
            torso_y_min = min(sh_l.y, sh_r.y)
            torso_y_max = max(hip_l.y, hip_r.y)
            torso_x_min = min(sh_r.x, hip_r.x)
            torso_x_max = max(sh_l.x, hip_l.x)

            # Check for impossible limb positioning (wrist intersecting deep torso without occlusion logic)
            # This flags "ghost hands" often spawned by AI inside the chest
            for wrist_idx in [15, 16]:
                if landmarks[wrist_idx].visibility > 0.7:
                    wx, wy = landmarks[wrist_idx].x, landmarks[wrist_idx].y
                    # If wrist is tightly bound inside the center of torso
                    if torso_x_min + 0.1 < wx < torso_x_max - 0.1 and torso_y_min + 0.1 < wy < torso_y_max - 0.1:
                        # Ensure elbow is also visible to confirm it's not just folded arms
                        elbow_idx = 13 if wrist_idx == 15 else 14
                        if landmarks[elbow_idx].visibility < 0.2:
                            anomalies.append("Floating appendage detected intersecting torso region.")

        # 2. Hand Analysis
        for hand_landmarks, hand_name in [(results.left_hand_landmarks, "Left"), (results.right_hand_landmarks, "Right")]:
            if hand_landmarks:
                flags["hand_count"] += 1
                fingers = 0
                
                # Finger counting logic (Tip Y < PIP Y)
                for tip, pip in [(8, 6), (12, 10), (16, 14), (20, 18)]:
                    if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[pip].y:
                        fingers += 1
                # Thumb (X compare, simplified)
                if abs(hand_landmarks.landmark[4].x - hand_landmarks.landmark[0].x) > abs(hand_landmarks.landmark[3].x - hand_landmarks.landmark[0].x):
                     fingers += 1
                     
                flags["total_fingers"] += fingers
                
                if fingers > 5:
                    anomalies.append(f"Polydactyly detected structural mutation: {hand_name} hand has abnormal finger count.")
                if fingers < 4:
                    anomalies.append(f"Missing digits detected: {hand_name} hand appears fused or incomplete.")

    # 3. Final Verdict
    # High-confidence signals that force override
    is_suspicious = len(anomalies) > 0
    severity = "low"
    if is_suspicious:
        severity = "high"

    # Draw Overlay
    overlay_img = img_cv.copy()
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(overlay_img, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
    if results.left_hand_landmarks:
        mp_drawing.draw_landmarks(overlay_img, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    if results.right_hand_landmarks:
        mp_drawing.draw_landmarks(overlay_img, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

    _, buffer = cv2.imencode('.png', overlay_img)
    overlay_b64 = base64.b64encode(buffer).decode()

    expl = ". ".join(anomalies) if anomalies else "No primary anatomical structural defects found."

    return {
        "is_suspicious": is_suspicious,
        "anomalies": anomalies,
        "severity": severity,
        "flags": flags,
        "overlay_b64": overlay_b64,
        "explanation": "Suspicious — Anatomical Anomaly Detected. " + expl if is_suspicious else expl
    }
