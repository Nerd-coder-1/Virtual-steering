import cv2
import mediapipe as mp
import math
from pynput.keyboard import Controller

keyboard = Controller()

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    max_num_hands=2,
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

w_down = s_down = a_down = d_down = False

SMOOTH_FACTOR = 0.4
MAX_ANGLE = 60   
MIN_ANGLE = 5    

def smooth_angle(prev, new, factor=SMOOTH_FACTOR):
    return prev + factor * (new - prev)

def press_once(key):
    keyboard.press(key)

def release_once(key):
    try:
        keyboard.release(key)
    except:
        pass

def get_angle(p1, p2):
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    return math.degrees(math.atan2(dy, dx))

cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

smoothed_angle = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    display_frame = frame.copy()

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    h, w, _ = frame.shape
    wrist_positions = []

    if results.multi_hand_landmarks:
        for hand in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(display_frame, hand, mp_hands.HAND_CONNECTIONS)
            cx = int(hand.landmark[mp_hands.HandLandmark.WRIST].x * w)
            cy = int(hand.landmark[mp_hands.HandLandmark.WRIST].y * h)
            wrist_positions.append((cx, cy))

  
    if len(wrist_positions) == 2:
        p1, p2 = wrist_positions

        cv2.line(display_frame, p1, p2, (0, 255, 255), 3)

        raw_angle = get_angle(p1, p2)
        smoothed_angle = smooth_angle(smoothed_angle, raw_angle)

        cv2.putText(display_frame, f"Angle: {int(smoothed_angle)} deg",
                    (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)

        if not w_down:
            press_once('w'); w_down = True
        if s_down:
            release_once('s'); s_down = False

        if smoothed_angle > MIN_ANGLE:
            if not d_down:
                press_once('d'); d_down = True
            if a_down:
                release_once('a'); a_down = False
        elif smoothed_angle < -MIN_ANGLE:
            if not a_down:
                press_once('a'); a_down = True
            if d_down:
                release_once('d'); d_down = False
        else:
            if a_down: release_once('a'); a_down = False
            if d_down: release_once('d'); d_down = False

    elif len(wrist_positions) == 1:
        if not s_down:
            press_once('s'); s_down = True
        if w_down:
            release_once('w'); w_down = False
        if a_down:
            release_once('a'); a_down = False
        if d_down:
            release_once('d'); d_down = False

        cv2.putText(display_frame, "BRAKE", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)

    else:
        if w_down: release_once('w'); w_down = False
        if s_down: release_once('s'); s_down = False
        if a_down: release_once('a'); a_down = False
        if d_down: release_once('d'); d_down = False

        cv2.putText(display_frame, "NO HANDS", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)

    small_display = cv2.resize(display_frame, (400, 300))
    cv2.namedWindow("Gesture Control")
    cv2.setWindowProperty("Gesture Control", cv2.WND_PROP_TOPMOST, 1)
    cv2.imshow("Gesture Control", small_display)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
