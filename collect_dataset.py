import os
import cv2
import time

def setup_folders(root="Dataset"):
    if not os.path.exists(root):
        os.makedirs(root)
    for i in range(10):
        class_path = os.path.join(root, str(i))
        os.makedirs(class_path, exist_ok=True)

def get_next_index(folder):
    files = [f for f in os.listdir(folder) if f.endswith(".jpg")]
    numbers = [int(f.split(".")[0]) for f in files if f.split(".")[0].isdigit()]
    return max(numbers) + 1 if numbers else 1

def get_image_count(folder):
    return len([f for f in os.listdir(folder) if f.endswith(".jpg")])

dataset_path = "./Dataset"
roi_start, roi_end = (100, 100), (300, 300)
img_size = (100, 100)
font = cv2.FONT_HERSHEY_SIMPLEX

setup_folders(dataset_path)
cap = cv2.VideoCapture(0)
print("Press 0–9 to capture, 'q' to quit.")

last_save_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    x1, y1 = roi_start
    x2, y2 = roi_end

    cv2.rectangle(frame, roi_start, roi_end, (0, 255, 0), 2)
    cv2.putText(frame, "Press 0–9 to capture", (x1, y1 - 10), font, 0.7, (255, 0, 0), 2)

    for i in range(10):
        count = get_image_count(os.path.join(dataset_path, str(i)))
        cv2.putText(frame, f"{i}: {count}", (10, 30 + i * 25), font, 0.6, (0, 255, 255), 2)

    roi = frame[y1:y2, x1:x2]
    roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    roi_resized = cv2.resize(roi_gray, img_size)

    cv2.imshow("Live Feed", frame)
    cv2.imshow("ROI", roi_resized)

    key = cv2.waitKey(1) & 0xFF

    if ord('0') <= key <= ord('9'):
        digit = chr(key)
        save_dir = os.path.join(dataset_path, digit)
        index = get_next_index(save_dir)
        filename = f"{index:04d}.jpg"
        path = os.path.join(save_dir, filename)

        success = cv2.imwrite(path, roi_resized)
        if success:
            print(f"Saved: {path}")
            time.sleep(0.2) 
        else:
            print(f"Failed to save: {path}")

    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
