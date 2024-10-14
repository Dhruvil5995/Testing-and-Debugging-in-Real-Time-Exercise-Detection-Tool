import cv2
import numpy as np
import os
import mediapipe as mp
from tensorflow import keras
import tensorflow
from keras.layers import Dense, Dropout , Activation
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import TensorBoard


exercise_list =np.array(['rest', 'Shoulder','Tricep', 'Bicep'])
#folder = '30_Data'
#frame = 30
#video = 30

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils  # Drawing utilities

def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # COLOR CONVERSION BGR 2 RGB
    image.flags.writeable = False  # Image is no longer writeable
    skeleton_model = model.process(image)  # Make prediction
    image.flags.writeable = True  # Image is now writeable
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # COLOR COVERSION RGB 2 BGR
    return image, skeleton_model


def draw_styled_landmarks(image, skeleton_model):
    # Draw pose connections
    mp_drawing.draw_landmarks(image, skeleton_model.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                              mp_drawing.DrawingSpec(color=(80, 22, 10), thickness=2, circle_radius=4),
                              mp_drawing.DrawingSpec(color=(80, 44, 121), thickness=2, circle_radius=2))


def extract_keypoints(skeleton_model):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in skeleton_model.pose_landmarks.landmark]).flatten() if skeleton_model.pose_landmarks else np.zeros(132)
    return np.concatenate([pose])

label_map = {label:num for num, label in enumerate(exercise_list)}
print(label_map)


sequences, labels = [], []
for exe in exercise_list:
    for sequence in np.array(os.listdir(os.path.join(os.path.join('30_Data'), exe))).astype(int):
        window = []
        for frame_num in range(30):
            res = np.load(os.path.join(os.path.join('30_Data'), exe, str(sequence), "{}.npy".format(frame_num)))
            window.append(res)
        sequences.append(window)
        labels.append(label_map[exe])

np.array(sequences).shape
print(labels)
X = np.array(sequences)

y = to_categorical(labels).astype(int)
print(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

#train LSTM  neural network
"""
log_dir = os.path.join('Logs')
tb_callback = TensorBoard(log_dir=log_dir)

model = Sequential()
model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(30,132)))
model.add(LSTM(128, return_sequences=True, activation='relu'))
model.add(LSTM(64, return_sequences=False, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(exercise_list.shape[0], activation='softmax'))


model.compile(optimizer=tensorflow.keras.optimizers.Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['categorical_accuracy'])

model.fit(X_train, y_train, epochs=500)

#model_in_json = model.to_json()
#with open('model60new.json','w') as json_file:
    #json_file.write(model_in_json)



#model.save('30videos.h5')
"""


































































































