import cv2
import numpy as np
import mediapipe as mp

from keras.models import load_model

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

exercise_list =np.array(['rest', 'Shoulder','Tricep', 'Bicep'])
folder = '30_Data'
frame = 30
video = 30



#model_file=open('model60new.json','r')
#json_model= model_file.read()
#model60 = model_from_json(json_model)
#model60.summary()
model = load_model("E:\\python_projects_CV\\LSTM_numpy\\30videos.h5", compile=True)

def load_trained_model():
    # Load the model from a file
    model = load_model("E:\\python_projects_CV\\LSTM_numpy\\30videos.h5", compile=True)
    return model

def run_camera(model):
    sequence = []
    sentence = []
    predictions = []
    threshold = 0.5

    cap = cv2.VideoCapture(0)
    # Set mediapipe model
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as Pose:
        while cap.isOpened():
            # Read feed
            ret, frame = cap.read()

            # Make detections
            image, skeleton_model = mediapipe_detection(frame, Pose)
            print(skeleton_model)

            # Draw landmarks
            draw_styled_landmarks(image, skeleton_model)

            # Prediction logic
            keypoints = extract_keypoints(skeleton_model)
            sequence.append(keypoints)
            sequence = sequence[-30:]

            if len(sequence) == 30:
                res = model.predict(np.expand_dims(sequence, axis=0))[0]
                print(res)
                display = exercise_list[np.argmax(res)]
                print(display)
                predictions.append(np.argmax(res))

                cv2.putText(image, display, (3, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

            # Show to screen
            cv2.imshow('OpenCV Feed', image)

            # Break gracefully
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    model = load_trained_model()
    run_camera(model)