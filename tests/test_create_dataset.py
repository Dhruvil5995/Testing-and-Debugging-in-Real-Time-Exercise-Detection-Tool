import numpy as np
from unittest.mock import MagicMock, patch
from create_dataset import extract_keypoints, mediapipe_detection,  create_dataset
from mediapipe.framework.formats import landmark_pb2
import cv2


def test_extract_keypoints_output():
    # Create a mock skeleton model with mock pose landmarks
    mock_skeleton_model = MagicMock()

    # Mock the pose_landmarks object, with 33 landmarks (as MediaPipe usually outputs)
    mock_landmarks = [MagicMock(x=0.1 * i, y=0.2 * i, z=0.3 * i, visibility=0.9) for i in range(33)]
    mock_skeleton_model.pose_landmarks.landmark = mock_landmarks

    # Call the extract_keypoints function with the mock model
    keypoints = extract_keypoints(mock_skeleton_model)

    # Assert the correct output length (33 landmarks * 4 elements each = 132)
    assert len(keypoints) == 132
    assert isinstance(keypoints, np.ndarray)
    assert np.all(keypoints >= 0)


def test_mediapipe_detection():
    # Create a dummy image (a blank image with the same size as a frame you would normally process)
    test_image = np.zeros((480, 640, 3), dtype=np.uint8)  # A black image to simulate a video frame

    # Mock the MediaPipe pose model and its process method
    mock_pose_model = MagicMock()

    # Mock the output of the process method (e.g., mock a skeleton model)
    mock_pose_output = MagicMock()
    mock_pose_model.process.return_value = mock_pose_output

    # Call the function with the mock model and dummy image
    processed_image, skeleton_model = mediapipe_detection(test_image, mock_pose_model)

    # Assert the processed image is returned and has the same shape as the input
    assert processed_image.shape == test_image.shape

    # Assert the skeleton_model is the one returned by the mock
    assert skeleton_model == mock_pose_output


def test_create_dataset():
    # Mock os.makedirs to avoid creating directories
    with patch('os.makedirs') as mock_makedirs, \
         patch('numpy.save') as mock_save, \
         patch('cv2.VideoCapture') as mock_video_capture, \
         patch('create_dataset.mediapipe_detection') as mock_mediapipe_detection, \
         patch('create_dataset.extract_keypoints') as mock_extract_keypoints:

        # Mock the video capture's behavior
        mock_cap = MagicMock()
        mock_video_capture.return_value = mock_cap
        mock_cap.read.return_value = (True, np.zeros((480, 640, 3), dtype=np.uint8))

        # Create a valid mock for pose landmarks
        mock_landmarks = landmark_pb2.NormalizedLandmarkList()

        # Add 33 valid landmarks to the mock_landmarks (as MediaPipe typically returns 33 landmarks)
        for i in range(33):
            landmark = landmark_pb2.NormalizedLandmark(
                x=0.1 * i, y=0.2 * i, z=0.3 * i, visibility=0.9
            )
            mock_landmarks.landmark.append(landmark)

        # Mock the output of mediapipe_detection to return a valid image and pose landmarks
        mock_mediapipe_detection.return_value = (np.zeros((480, 640, 3), dtype=np.uint8), MagicMock(pose_landmarks=mock_landmarks))

        # Mock extract_keypoints to return dummy keypoints
        mock_extract_keypoints.return_value = np.zeros(132)

        # Call the create_dataset function
        create_dataset(MagicMock(), ['Bicep'], 2, 5, 'test_folder')

        # Assert that os.makedirs was called to create directories
        assert mock_makedirs.call_count > 0

        # Assert that np.save was called to save the keypoints
        assert mock_save.call_count == 10  # 2 videos * 5 frames each = 10 saves
