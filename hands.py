import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import cv2 as cv
import numpy as np
import mediapipe as mp
import pickle
from collections import Counter, deque

def check_j(buf):
    if len(buf) < 30:
        return False

    # --- 1. Extract Start, Current (End), and Pinky Trajectory ---
    (start_ix, start_iy), (start_px, start_py) = buf[0]
    (end_ix, end_iy), (end_px, end_py) = buf[-1]

    # Extract all pinky (x, y) coordinates across the buffer history
    pinky_pts = [frame[1] for frame in buf]

    # --- 2. Find the lowest point drawn by the pinky (maximum Y) ---
    lowest_pinky_pt = max(pinky_pts, key=lambda pt: pt[1])
    lowest_px, lowest_py = lowest_pinky_pt

    # --- 3. Geometric Threshold Checks ---
    # Downward stroke: Lowest point must be significantly lower than start position
    vertical_drop = lowest_py - start_py
    
    # Inward hook: Direction depends on index finger relative to pinky
    inward_direction = -1 if end_ix < end_px else 1
    inward_hook = (end_px - lowest_px) * inward_direction

    # Check thresholds (adjust px values if too sensitive or hard to trigger)
    if vertical_drop > 30 and inward_hook > 15:
        return True

def check_z(buf):
    # Need enough frames to draw 3 distinct strokes (~20 frames minimum)
    if len(buf) < 30:
        return False

    # Get index finger points (x, y) across history
    index_pts = [frame[0] for frame in buf]

    start_x, start_y = index_pts[0]
    end_x, end_y = index_pts[-1]

    # 1. Find Top-Right Corner (Max X)
    top_right_pt = max(index_pts, key=lambda pt: pt[0])
    top_right_x, top_right_y = top_right_pt

    # Find the index position in buf where Top-Right occurred
    top_right_idx = index_pts.index(top_right_pt)

    # 2. Find Bottom-Left Corner (Min X AFTER Top-Right)
    remaining_pts = index_pts[top_right_idx:]
    if not remaining_pts:
        return False
        
    bottom_left_pt = min(remaining_pts, key=lambda pt: pt[0])
    bottom_left_x, bottom_left_y = bottom_left_pt

    # --- GEOMETRIC CHECKS FOR Z ---
    # Stroke 1: Moved Right to Top-Right corner
    stroke1_right = top_right_x - start_x > 25

    # Stroke 2: Moved Diagonal Left-Down to Bottom-Left corner
    stroke2_left = top_right_x - bottom_left_x > 25
    stroke2_down = bottom_left_y - top_right_y > 20

    # Stroke 3: Moved Right from Bottom-Left corner to End
    stroke3_right = end_x - bottom_left_x > 20

    # Progressive downward motion overall
    overall_down = end_y - start_y > 30

    return stroke1_right and stroke2_left and stroke2_down and stroke3_right and overall_down
    
def dynamic_movement(buf):
    if len(buf) < 30:
        return None

    if check_j(buf):
        return 'J'
    
    if check_z(buf):
        return 'Z'
    
    return None



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
movement_buffer = deque(maxlen=45)
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
        #just draws green lines
        mp_drawing.draw_landmarks(
            frame,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style()
        )
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
   
        #Get pinky and index for J/Z
        index_tip = hand_landmarks.landmark[8]
        pinky_tip = hand_landmarks.landmark[20]

        index_x = int(index_tip.x * w)
        index_y = int(index_tip.y * h)
        pinky_x = int(pinky_tip.x * w)
        pinky_y = int(pinky_tip.y * h)

        movement_buffer.append(((index_x, index_y), (pinky_x, pinky_y)))
        
        dynamix = dynamic_movement(movement_buffer)
        if dynamix != None:
            sentence.append(dynamix)
            movement_buffer.clear()
            prediction_buffer.clear()
            print("buffers clear")
        
        else:
            #refer to the model
            prediction = elModelo.predict([relative_coords])
            predicted_letter = prediction[0]

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
                    movement_buffer.clear()
                    print("buffers cleared")
                    added = False
        
    
    key = cv.waitKey(1)
    if key == 127:
        if sentence:
            sentence.pop()
        
    if key == ord(' '):
        sentence.append(' ')
    
    if predicted_letter == "":
        added = False
        prediction_buffer.clear()
        movement_buffer.clear()
    
    cv.putText(
            frame,
            f"*: {(''.join(sentence))}",
            (30,200),
            cv.FONT_HERSHEY_COMPLEX,
            4,
            (255, 255, 255),
            3,
            cv.LINE_AA
        )

    cv.imshow('changed', frame)
    if key == ord('q'):
        break
        
cap.release()
cv.destroyAllWindows()
