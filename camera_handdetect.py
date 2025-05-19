import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
from collections import deque

# Load the pre-trained model
model = tf.keras.models.load_model("model.h5")
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# Preprocess the hand image for prediction
def preprocess(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_gray = cv2.resize(img_gray, (100, 100))
    img_gray = img_gray / 255.0  # normalize pixel values
    return img_gray.reshape(1, 100, 100, 1)

# Initialize MediaPipe hands
mpHands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands_detector = mpHands.Hands(max_num_hands=1, min_detection_confidence=0.7)

# keep track of our last few predictions
history = deque(maxlen=5)

# start the webcam
cap = cv2.VideoCapture(0)
print("Camera on, click q to make it stop")

while True: # keeps going till q is pressed or we stop it, effective to capture frames from the webcam
    success, frame = cap.read()  # read a frame from the webcam
    if not success:
        break  # just exit if cannot read or camera is being weird

    frame = cv2.flip(frame, 1)  # mirror effect pretty much
    height, width, _ = frame.shape  # need dimensions for the bounding box
    # Convert the frame to RGB format for MediaPipe
    rgbF = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  
    result = hands_detector.process(rgbF)  # detect hands using mediapipe

    screenText = "Detecting..."  # default will say this

    if result.multi_hand_landmarks:  # if any hands are detected
        for hand in result.multi_hand_landmarks:  # iterate through the hands we have detected
            mp_draw.draw_landmarks(frame, hand, mpHands.HAND_CONNECTIONS)  # draw hand landmarks on the hand using mediapipe's detector that is very useful
            # bounding box
            xList = []
            for point in hand.landmark:
                xList.append(int(point.x * width)) # convert to pixel coordinates
            yList = []
            for point in hand.landmark:
                yList.append(int(point.y * height)) # convert to pixel coordinates
            # get the min and max x and y coordinates for the bounding box
            xMin, x_max = min(xList), max(xList)
            yMin, y_max = min(yList), max(yList)

            # expand slightly it cropped better we saw
            center_x, center_y = (xMin + x_max) // 2, (yMin + y_max) // 2 # center of the hand
            boxSize = max(x_max - xMin, y_max - yMin) + 40  # some padding
            half = boxSize // 2

            # need to make sure the box is within the frame, so these are the boundaries where we can crop

            x1 = max(center_x - half, 0)
            x2 = min(center_x + half, width)
            y1 = max(center_y - half, 0)
            y2 = min(center_y + half, height)

            cropHand = frame[y1:y2, x1:x2]  # crop the hand from the frame

            try:
                inpImg = preprocess(cropHand)  # process crop
                prediction = model.predict(inpImg, verbose=0)[0]  # predict number using model
                top = np.max(prediction)  # make note of highest confidence score
                predicted_digit = int(np.argmax(prediction))  # get predicted digit

                if top > 0.7:  # Check if the confidence is above the threshold
                    history.append(predicted_digit)  # add to history
                    # get most common prediction, from which we can then vote on which one it likely is
                    most_common_digit = max(set(history), key=history.count) 
                    screenText = f"Digit: {most_common_digit} ({top:.2f})"  # prediction text
                else:
                    screenText = "Low Confidence"  # low confidence if not above that threshold

            except Exception as err:
                print("failed:", err)  # error
                screenText = "Prediction error, sorry"  

            # Draw the bounding box and display the prediction text
            # green rectangle around hand
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            # green rectangle around hand
            cv2.putText(frame, screenText, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    # show image with our predictions
    cv2.imshow("Digit Recognizer", frame)

   # exit if 1 is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# deallocate
cap.release()
cv2.destroyAllWindows()
