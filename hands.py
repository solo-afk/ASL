import cv2 as cv
import numpy as np
import mediapipe as mp
import pickle
from collections import Counter, deque

print("Loading ml model...")
with open('asl_model.p', 'rb') as f:
    model_data = pickle.load(f)
elModelo = model_data['model']
print("El Modelo loaded successfully!")

cap = cv.VideoCapture(0)

if not cap.isOpened():
    print("Cannot open camera")
    exit()

#creates a handmarker object
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7
)

#drawing function
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

prediction_buffer = deque(maxlen=45)
currentLetter = ""

sentence = deque()
added = False

while True:
    ret, frame = cap.read()
    if not ret:
        print("Can't recieve frame (stream end?). Exiting...")
        break
    h, w, c = frame.shape
    
    frame = cv.flip(frame,1)
    #need to convert to RGB
    rgbframe = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    result = hands.process(rgbframe)

    predicted_letter = ""
   
    if result.multi_hand_landmarks:
        
        hand_landmarks = result.multi_hand_landmarks[0]

        #get raw coordinates for 21 joints
        raw_coords = []
        for lm in hand_landmarks.landmark:
            raw_coords.append(lm.x)
            raw_coords.append(lm.y)
            raw_coords.append(lm.z)
        
        #make everything relative
        wrist_x = raw_coords[0]
        wrist_y = raw_coords[1]
        wrist_z = raw_coords[2]
        relative_coords = []

        for i in range(21):
            relative_coords.append(raw_coords[i*3] - wrist_x)     
            relative_coords.append(raw_coords[i*3 + 1] - wrist_y) 
            relative_coords.append(raw_coords[i*3 + 2] - wrist_z) 
   
        #refer to the model
        prediction = elModelo.predict([relative_coords])
        predicted_letter = prediction[0]

        mp_drawing.draw_landmarks(
            frame,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style()
        )
    
    if predicted_letter:
        prediction_buffer.append(predicted_letter)
        counts = Counter(prediction_buffer)
        mostCommon, count = counts.most_common(1)[0]

        if count >= 30:
            if not added:
                currentLetter = mostCommon
                sentence.append(currentLetter)
                added = True
                prediction_buffer.clear()
                added = False
        
        if currentLetter == "delete":
            sentence.pop()

    if predicted_letter == "":
        added = False
        prediction_buffer.clear()
    cv.putText(
            frame,
            f"*: {(''.join(sentence))}",
            (30,200),
            cv.FONT_HERSHEY_SIMPLEX,
            4,
            (255, 255, 255),
            3,
            cv.LINE_AA
        )

    cv.imshow('changed', frame)
    if cv.waitKey(1) == ord('q'):
        break
        
cap.release()
cv.destroyAllWindows()
