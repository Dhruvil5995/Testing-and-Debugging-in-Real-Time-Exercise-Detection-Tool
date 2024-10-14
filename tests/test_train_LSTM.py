# test_train_LSTM.py

import numpy as np
import os
from unittest.mock import patch
from train_LSTM import exercise_list, label_map


def test_data_loading():
    # Mock the np.load function to return dummy numpy arrays for testing
    with patch('numpy.load') as mock_npy_load:
        mock_npy_load.return_value = np.zeros((132,))  # Mock 132 keypoints

        # Mock os.listdir to simulate existing directories with sequence numbers
        with patch('os.listdir') as mock_listdir:
            mock_listdir.return_value = [str(i) for i in range(2)]  # Mock 2 sequences

            # Prepare sequences and labels list for testing
            sequences, labels = [], []
            for exe in exercise_list:
                for sequence in np.array(os.listdir(os.path.join(os.path.join('30_Data'), exe))).astype(int):
                    window = []
                    for frame_num in range(30):
                        res = np.load(
                            os.path.join(os.path.join('30_Data'), exe, str(sequence), "{}.npy".format(frame_num)))
                        window.append(res)
                    sequences.append(window)
                    labels.append(label_map[exe])

            # Assertions to ensure data is loaded correctly
            assert len(sequences) == 2 * len(exercise_list)  # 2 sequences for each exercise
            assert len(labels) == 2 * len(exercise_list)
            assert len(sequences[0]) == 30  # 30 frames per sequence


# test_train_LSTM.py

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from train_LSTM import exercise_list

def test_lstm_model_creation():
    # Create the LSTM model
    model = Sequential()
    model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(30, 132)))
    model.add(LSTM(128, return_sequences=True, activation='relu'))
    model.add(LSTM(64, return_sequences=False, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(len(exercise_list), activation='softmax'))

    # Assert model has the correct number of layers
    assert len(model.layers) == 6

    # Assert output shape of the first LSTM layer
    assert model.layers[0].output_shape == (None, 30, 64)  # First LSTM layer shape

    # Assert output shape of the final Dense layer
    assert model.layers[-1].output_shape == (None, len(exercise_list))  # Output layer for classification
