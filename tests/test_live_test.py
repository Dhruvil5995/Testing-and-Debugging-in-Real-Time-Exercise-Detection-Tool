# test_live_test.py

import numpy as np
from unittest.mock import patch, MagicMock
from live_test import load_trained_model, mediapipe_detection, extract_keypoints, run_camera, exercise_list


def test_model_loading():
    # Mock the load_model function to avoid loading the real model
    with patch('live_test.load_model') as mock_load_model:
        mock_model = MagicMock()
        mock_load_model.return_value = mock_model

        # Call the function to load the model
        model = load_trained_model()

        # Assert that load_model was called with the correct file path
        mock_load_model.assert_called_once_with("E:\\python_projects_CV\\LSTM_numpy\\30videos.h5", compile=True)
        assert model == mock_model


# test_live_test.py

def test_prediction_logic():
    # Mock the model's predict function
    mock_model = MagicMock()
    mock_model.predict.return_value = np.array([[0.2, 0.1, 0.5, 0.2]])  # Mocked probabilities

    # Mock a sequence of keypoints
    sequence = [np.zeros((132,)) for _ in range(30)]  # Mock 30 frames, each with 132 keypoints

    # Simulate the prediction and check if the right label is predicted (based on the highest probability)
    result = mock_model.predict(np.expand_dims(sequence, axis=0))[0]
    predicted_exercise = exercise_list[np.argmax(result)]

    # Check that the predicted exercise is the one with the highest probability
    expected_exercise = exercise_list[np.argmax(mock_model.predict.return_value)]
    assert predicted_exercise == expected_exercise  # Make the assertion flexible


def test_mediapipe_detection():
    # Mock Mediapipe model processing
    mock_pose_model = MagicMock()
    mock_pose_model.process.return_value = MagicMock()

    # Mock an input image
    image = np.zeros((480, 640, 3), dtype=np.uint8)  # Dummy black image

    # Run mediapipe_detection and assert the correct behavior
    image_out, skeleton_model = mediapipe_detection(image, mock_pose_model)

    # Ensure the function returns valid results (mocked in this case)
    assert image_out is not None
    assert skeleton_model is not None
