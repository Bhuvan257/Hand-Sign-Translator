import os
import pickle

import mediapipe as mp
import cv2

# Suppress TensorFlow logs if they are not needed
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow logs

mp_hands = mp.solutions.hands

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Initialize MediaPipe Hands with static images mode
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

DATA_DIR = './data'

data = []
labels = []

if not os.path.exists(DATA_DIR):
    print(f"The specified data directory '{DATA_DIR}' does not exist.")
else:
    for dir_ in os.listdir(DATA_DIR):
        dir_path = os.path.join(DATA_DIR, dir_)
        if not os.path.isdir(dir_path):
            continue

        for img_path in os.listdir(dir_path):
            img_full_path = os.path.join(dir_path, img_path)
            img = cv2.imread(img_full_path)
            if img is None:
                print(f"Could not read image '{img_full_path}'. Skipping.")
                continue

            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            print(f"Processing: {img_full_path}")

            results = hands.process(img_rgb)

            # Skip images with no hands detected or multiple hands
            if not results.multi_hand_landmarks:
                # print(f"No hand detected in '{img_full_path}', skipping.")
                continue

            if len(results.multi_hand_landmarks) != 1:
                # print(f"Multiple hands detected in '{img_full_path}', skipping.")
                continue

            data_aux = []
            x_ = []
            y_ = []

            for hand_landmarks in results.multi_hand_landmarks:
                # Collect x and y coordinates of landmarks
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    x_.append(x)
                    y_.append(y)

                # Normalize coordinates relative to the minimum x and y
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    data_aux.append(x - min(x_))
                    data_aux.append(y - min(y_))

            # Check feature vector length
            if len(data_aux) != 42:  # 21 landmarks × 2 coords
                print(f"Warning: feature vector length {len(data_aux)} != 42 for image '{img_full_path}'. Skipping.")
                continue

            data.append(data_aux)
            labels.append(dir_)

hands.close()  # Free resources

with open('data.pickle', 'wb') as f:
    pickle.dump({'data': data, 'labels': labels}, f)

print("Data has been successfully saved to 'data.pickle'.")
