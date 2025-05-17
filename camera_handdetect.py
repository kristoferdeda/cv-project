import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
from collections import deque

model = tf.keras.models.load_model("model.h5")
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

def preprocess(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (100, 100))
    gray = gray / 255.0
    return gray.reshape(1, 100, 100, 1)

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)

prediction_buffer = deque(maxlen=5)

cap = cv2.VideoCapture(0)
print("Camera started — press 'q' to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(img_rgb)

    pred_digit = "Detecting..."

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            x_vals = [int(lm.x * w) for lm in hand_landmarks.landmark]
            y_vals = [int(lm.y * h) for lm in hand_landmarks.landmark]
            x_min, x_max = min(x_vals), max(x_vals)
            y_min, y_max = min(y_vals), max(y_vals)

            cx, cy = (x_min + x_max) // 2, (y_min + y_max) // 2
            box_size = max(x_max - x_min, y_max - y_min) + 40 
            half = box_size // 2
            x1 = max(cx - half, 0)
            x2 = min(cx + half, w)
            y1 = max(cy - half, 0)
            y2 = min(cy + half, h)

            hand_img = frame[y1:y2, x1:x2]

            try:
                processed = preprocess(hand_img)
                prediction = model.predict(processed, verbose=0)[0]
                confidence = np.max(prediction)
                digit = int(np.argmax(prediction))

                if confidence > 0.7:
                    prediction_buffer.append(digit)
                    most_common = max(set(prediction_buffer), key=prediction_buffer.count)
                    pred_digit = f"Digit: {most_common} ({confidence:.2f})"
                else:
                    pred_digit = "Low Confidence"

            except Exception as e:
                pred_digit = "Prediction Error"
                print("Prediction failed:", e)

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, pred_digit, (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    cv2.imshow("Digit Recognizer", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
