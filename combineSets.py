import pandas as pd

#loads the sets
original = pd.read_csv('asl_landmarks_final.csv')
custom = pd.read_csv('my_custom_landmarks.csv')

#merge, ignore makes new lines start from 0; avoid duplicates
combined = pd.concat([original,custom], ignore_index=True)

#ignores J and Z
static = combined[~combined['label'].isin(['J', 'Z'])].copy()

static.to_csv('asl_landmarks_combined.csv', index=False)


print(f"Testing static letters. Total samples: {len(static)}")

