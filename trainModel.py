import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import numpy as np
import pickle

# 1. Load the data
print("Loading dataset")
file = pd.read_csv('no_kaggle_landmarks.csv') 
print(file['label'].value_counts())


#feature engineering: converying absolute coordinates to relative
print("Engineering relative coordinates")

#seperate features and labels first
x_raw = file.drop(columns=['label']).values #converts to a numpy array
y = file['label']

x_relative = []

for row in x_raw:
    #firstt 3 columns are the xyz of the writst
    wrist_x = row[0]
    wrist_y = row[1]
    wrist_z = row[2]

    row_relative = []

    for i in range(21):
        #getting relative coordinate (wrist - currentjoint)
        row_relative.append(row[i*3] - wrist_x)
        row_relative.append(row[i*3 + 1] - wrist_y)
        row_relative.append(row[i*3 + 2] - wrist_z)
    
    x_relative.append(row_relative)

#convert processed list back to array
x = np.array(x_relative)

# 3. Split into 80% Training data and 20% Testing data
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

print(f"Training with {len(X_train)} samples.")
print(f"Testing with {len(X_test)} samples.")

# 4. Initialize and train the Machine Learning Model
print("\nTraining the model... hang tight...")
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)
print("Training complete!")

# 5. Check how smart the model is
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print(f"\nModel Accuracy: {accuracy * 100:.2f}%")

#save trained model
print(f"\nSaving {accuracy * 100:.2f}% model to disk...")

model_data = { 'model' : model}
with open('asl_model.p', 'wb') as f:
    pickle.dump(model_data, f)

print("Success! 'asl_model.p' is ready for use")
