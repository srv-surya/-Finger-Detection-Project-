import cv2
import mediapipe as mp
import math

# === CONSTANTS ===
# Landmark indices for fingertips and their lower joints (PIP/MCP)
FINGER_TIPS = [4, 8, 12, 16, 20]     # Thumb, Index, Middle, Ring, Pinky tips
FINGER_PIPS = [3, 6, 10, 14, 18]     # Joints below each finger tip
THRESHOLD_Y = 0.02  # margin for vertical detection stability
THUMB_EXTEND_RATIO = 0.4  # Lower = more sensitive thumb detection

def distance_2d(lm1, lm2):
    """Helper to compute 2D Euclidean distance between two landmarks"""
    return math.hypot(lm1.x - lm2.x, lm1.y - lm2.y)

def count_fingers(hand_landmarks):
    """
    Count how many fingers are raised for a single detected hand.
    Uses distance for thumb detection instead of only X-axis.
    """
    lm = hand_landmarks.landmark
    fingers = []

    # === 1) Thumb detection using distance ===
    thumb_tip = lm[FINGER_TIPS[0]]
    index_mcp = lm[5]  # base joint of the index finger
    wrist = lm[0]

    dist_thumb_index = distance_2d(thumb_tip, index_mcp)
    dist_wrist_index = distance_2d(wrist, index_mcp)

    # More sensitive detection → count thumb even for smaller separations
    fingers.append(1 if dist_thumb_index > dist_wrist_index * THUMB_EXTEND_RATIO else 0)

    # === 2) Other four fingers detection ===
    # Tip higher than pip joint → finger is open
    for tip, pip in zip(FINGER_TIPS[1:], FINGER_PIPS[1:]):
        tip_y = lm[tip].y
        pip_y = lm[pip].y
        # smaller y = higher on screen
        fingers.append(1 if (pip_y - tip_y) > THRESHOLD_Y else 0)

    return sum(fingers)

def main():
    # === Initialize Mediapipe modules ===
    mp_hands = mp.solutions.hands
    mp_draw  = mp.solutions.drawing_utils

    # Hands() parameters:
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7
    )

    # === Open webcam ===
    cap = cv2.VideoCapture(0)
    print("✅ Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Flip horizontally → like a mirror (more natural interaction)
        frame = cv2.flip(frame, 1)

        # Convert BGR → RGB (Mediapipe needs RGB)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame → detect hands & landmarks
        result = hands.process(rgb_frame)

        # If any hands detected
        if result.multi_hand_landmarks:
            for landmarks in result.multi_hand_landmarks:
                # Draw landmarks & connections on the frame
                mp_draw.draw_landmarks(frame, landmarks, mp_hands.HAND_CONNECTIONS)

                # Count how many fingers are up
                finger_count = count_fingers(landmarks)

                # Show finger count text on screen
                cv2.putText(frame, f"Fingers: {finger_count}",
                            (30, 50),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1, (0, 255, 0), 2)

        # Show the webcam feed
        cv2.imshow("Hand Gesture Recognition", frame)

        # Exit on 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # === Cleanup ===
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
