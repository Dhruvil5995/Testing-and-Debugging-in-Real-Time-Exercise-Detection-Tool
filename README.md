
# Project Overview
This project focuses on real-time exercise detection using Mediapipe for pose estimation and an LSTM model for classifying exercises based on key point data. Throughout the development process, I applied rigorous testing and debugging techniques to ensure the system's accuracy and reliability, especially when handling real-time data from a webcam.

The project has been structured to make testing and debugging an integral part of the development process. This helped me ensure that each part of the system works as expected before integrating it into a fully functional real-time application.


## Technologies I Used
* Python for scripting and overall application logic. 
* Mediapipe for real-time pose detection.
* OpenCV to handle video feeds from the webcam.
* TensorFlow/Keras is used to build and train the LSTM model.
* pytest and unit test for writing and running tests.
* Mocking with unit test.mock to simulate external dependencies like the webcam feed and model loading.
* Logging and Debugging tools to trace and fix issues during real-time execution.

### Skills I Gained

* I strengthened my knowledge of software testing, particularly through unit testing and using pytest for test automation.
* I became proficient in mocking and patching external dependencies, which helped me isolate and test individual components of the system.
* Debugging became a lot easier through the use of logging and step-through debugging with Python’s pdb module.


## Screenshots of Test Results

![Capture](https://github.com/user-attachments/assets/4d7827a0-06f5-42d7-9bfe-f1880ea5bace)


![Capture1](https://github.com/user-attachments/assets/f18779d6-cd19-4988-af2f-4d1bacd52cff)


![Capture2](https://github.com/user-attachments/assets/133cf00d-6e2c-4f76-baf0-2ba76ad40e4c)

## Testing and Debugging
Testing and debugging were a crucial part of this project. I wrote unit tests for every core function and made sure the system behaved as expected, even in a real-time setting. Below are some of the things I focused on during testing:

1. **Unit Testing with pytest**
I applied unit testing to ensure each part of the system functions correctly in isolation. For example, I tested the model loading process, keypoint extraction, and the prediction logic of the LSTM model. This gave me confidence that any issues would be caught early before integrating the components.

- Here’s an example test for model loading:

  ´´´sh
      def test_model_loading():
            with patch('live_test.load_model') as mock_load_model:
                mock_model = MagicMock()
                mock_load_model.return_value = mock_model
                model = load_model('dummy_model.h5')
                assert model == mock_model

2. **Mocking and Patching**
I used mocking to simulate external dependencies like webcam input and model loading, which allowed me to isolate individual components for testing. This helped me avoid relying on hardware (like a physical webcam) during the testing phase.

3. **Debugging**
I debugged the system using logging to track key events (e.g., pose detection, prediction accuracy) and pdb for stepping through the code. This allowed me to quickly identify and fix any issues, especially during the real-time execution.



















