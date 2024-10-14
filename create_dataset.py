import cv2
import numpy as np
import os
import mediapipe as mp


mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils  # Drawing utilities


def mediapipe_detection(image, model):
    image = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_BGR2RGB)  # Ensure the image is in uint8 format
    image.flags.writeable = False  # Image is no longer writeable
    skeleton_model = model.process(image)  # Make prediction
    image.flags.writeable = True  # Image is now writeable
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # COLOR CONVERSION RGB 2 BGR
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
folder = 'test_Data'
frame = 20
video = 20


def create_dataset(mp_pose, exercise, videos, frames, folder_name):
    cap = cv2.VideoCapture(0)
    # Set mediapipe model
    with mp_pose:
        # NEW LOOP
        # Loop through exercise

        for exe in exercise:

            c = 0

            # Loop through sequences  videos
            for sequence in range(videos):

                # Loop through video length , sequence length
                for frame_num in range(frames):

                    # Read feed
                    ret, frame = cap.read()

                    # Make detections
                    image, skeleton_model = mediapipe_detection(frame, mp_pose)

                    # Draw landmarks
                    draw_styled_landmarks(image, skeleton_model)

                    #Apply wait logic for data collection
                    print(frame_num, 'number of frame')
                    if frame_num == 0 and c == 0:
                        cv2.putText(image, 'STARTING COLLECTION for {} exercises'.format(exe), (120, 200),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 4, cv2.LINE_AA)
                        #print('before', c)


                    elif frame_num == 0 and c != 0:
                         cv2.putText(image,
                                    'Soon Collecting frames for {} Video Number {} will start'.format(exe, sequence),
                                    (15, 12),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                         #print('after', c)
                    else:
                        cv2.putText(image, 'Collecting frames for {} Video Number {}'.format(exe, sequence), (15, 12),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                    # NEW Export keypoints
                    keypoints = extract_keypoints(skeleton_model)
                    try:
                        os.makedirs(os.path.join(os.path.join(str(folder_name)), exe, str(sequence)))
                    except:
                        print('already exist', os.path.join(os.path.join(str(folder_name)), exe, str(sequence)))
                    # print('0 done')
                    np_path = os.path.join(str(folder_name), exe, str(sequence), str(frame_num))
                    # print('1 done')
                    np.save(np_path, keypoints)

                    imS = cv2.resize(image, (1280, 800))  # Resize image
                    cv2.imshow("output", imS)
                    if (frame_num == 0 and c == 0) or (frame_num == 0 and c != 0):
                        cv2.waitKey(2000)

                    # Break gracefully
                    if cv2.waitKey(10) & 0xFF == ord('q'):
                        break
                    c += 1

        cap.release()
        cv2.destroyAllWindows()

#create_dataset(mp_pose,exercise_list,video,frame,folder)

#mediapipe_detection(frame,mp_pose.Pose)
