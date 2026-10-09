import cv2
import csv
import os
import mediapipe as mp

# Target label you want to collect data for
LABEL = 'SPACE' 
OUTPUT_CSV = 'no_kaggle_landmarks.csv'

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)

# Create CSV with matching columns if it doesn't exist
if not os.path.exists(OUTPUT_CSV):
    headers = []
    for i in range(21):
        headers.extend([f'x{i}', f'y{i}', f'z{i}'])
    headers.append('label')
    
    with open(OUTPUT_CSV, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(headers)

cap = cv2.VideoCapture(0)
count = 0

print(f"Collecting data for label: '{LABEL}'. Press 's' to save single frame, 'r' to hold-record, 'q' to quit.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    current_landmarks = None

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # Flatten raw landmarks (x0, y0, z0, x1, y1, z1...)
            current_landmarks = []
            for lm in hand_landmarks.landmark:
                current_landmarks.extend([lm.x, lm.y, lm.z])

    cv2.putText(frame, f"Label: {LABEL} | Recorded: {count}", (10, 40), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Data Collector", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif (key == ord('s') or key == ord('r')) and current_landmarks:
        row = current_landmarks + [LABEL]
        with open(OUTPUT_CSV, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(row)
        count += 1

cap.release()
cv2.destroyAllWindows()