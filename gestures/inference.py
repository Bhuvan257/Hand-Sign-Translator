import pickle
import cv2
import mediapipe as mp
import numpy as np
import time

# Load trained model
model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

# Load the hand sign reference image
sign_reference = cv2.imread("D:\CV project\data\Screenshot 2025-05-25 154516.png")

# Resize reference image to fit screen (optional)
sign_reference = cv2.resize(sign_reference, (640, 360))

# Start webcam
cap = cv2.VideoCapture(1)

# MediaPipe setup
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# Label map
labels_dict = {i: chr(65 + i) for i in range(26)}  

last_prediction_time = time.time()
predicted_sentence = ""
predicted_character = ""

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    H, W, _ = frame.shape
    data_aux = []
    x_, y_ = [], []

    predicted_character = ""

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )

            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                x_.append(x)
                y_.append(y)

            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                data_aux.append(x - min(x_))
                data_aux.append(y - min(y_))

        if len(data_aux) == 42:  
            current_time = time.time()
            if current_time - last_prediction_time > 4: 
                prediction = model.predict([np.asarray(data_aux)])
                predicted_character = labels_dict[int(prediction[0])]
                predicted_sentence += predicted_character
                last_prediction_time = current_time

    # Show prediction in webcam window
    cv2.putText(frame, f"Detecting", (10, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
    cv2.imshow('Hand Sign Detection', frame)

    # Show sentence in separate output window
    output_window = np.ones((200, 640, 3), dtype=np.uint8) * 255
    cv2.putText(output_window, f"Typed: {predicted_sentence}", (10, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 2)
    cv2.imshow("Typed Output", output_window)

    cv2.imshow("Sign Reference", sign_reference)


    # Keyboard input handling
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('b'):  # Backspace
        predicted_sentence = predicted_sentence[:-1]
    elif key == ord('c'):  # Clear sentence
        predicted_sentence = ""
    elif key == ord(' '):  # Spacebar
        predicted_sentence += " "

# Cleanup
cap.release()
cv2.destroyAllWindows()
