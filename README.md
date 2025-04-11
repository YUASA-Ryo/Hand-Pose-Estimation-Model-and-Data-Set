# Hand Pose Estimation Model and Dataset for Text Input Systems
This repository provides a dataset and pre-trained model for classifying hand poses based on the coordinates of the fingertips of the left hand, obtained from a hand tracker in an XR environment.

## Dataset

This dataset is intended for research on hand pose estimation. It consists of six CSV files containing 3D coordinates of each fingertip of the left hand, captured while performing different hand poses. These poses are categorized into six classes: Pose 1 to Pose 5, and Other.

 The hand data is standardized by using the position and orientation of the wrist base as the origin, based on the WristRoot joint obtained from [OVRSkeleton Documentation](https://developers.meta.com/horizon/reference/unity/v69/class_o_v_r_skeleton/).

The hand shape represented by each class, along with the specific hand poses and the number of data samples included, are as follows:

### Pose 1: All fingers are bent.

The dataset consists of 5,000 samples of the hand pose shown in Figure 1-1, and 1,000 samples of the pose shown in Figure 1-2.

Figure 1-1

![Image](https://github.com/user-attachments/assets/56f22372-4763-4b1f-8771-6e221506cab1)

Figure 1-2

![Image](https://github.com/user-attachments/assets/8b123756-dd41-492d-adb0-5d6eb6c32a15)

### Pose 2: Only the thumb is extended.

The dataset consists of 5,000 samples of the hand pose shown in Figure 2-1.

Figure 2-1

![Image](https://github.com/user-attachments/assets/4bacd6df-3586-48d1-b8f5-8eb86f38df01)

### Pose 3: The thumb and index finger are extended.

The dataset consists of 5,000 samples of the hand pose shown in Figure 3-1.

Figure 3-1

![Image](https://github.com/user-attachments/assets/bc4b4368-e1d7-42e3-9ef5-0d79421b85d1)

### Pose 4: The thumb through the middle finger are extended.

The dataset consists of 5,000 samples of the hand pose shown in Figure 4-1.

Figure 4-1

![Image](https://github.com/user-attachments/assets/1e924ca3-b629-4eb6-884c-c7545e2b6ae2)

### Pose 5: The thumb through the ring finger, or all fingers, are extended.

The dataset consists of 5,000 samples of the hand pose shown in Figure 5-1, and 1,000 samples of the pose shown in Figure 5-2.

Figure 5-1

![Image](https://github.com/user-attachments/assets/0715a37f-0df3-4926-861d-fb3b4e05d898)

Figure 5-2

![Image](https://github.com/user-attachments/assets/23bccc16-9b36-4a1f-9ba7-9a888edf4f1d)

### Other: Hand poses that do not belong to Pose 1 through Pose 5.

The dataset includes 1,000 samples for each of the poses shown in Figure 6-1.
It consists of 14 hand poses, including poses inspired by Japanese fingerspelling and naturally relaxed hand shapes.

Figure 6-1

![Image](https://github.com/user-attachments/assets/1095a313-ef3e-4ba0-8b76-116b6c1df731)

## Pre-trained Models

The ONNX-format pre-trained model provided in this repository performs 6-class classification based on the 3D coordinates of each fingertip of the left hand. Since the model was built using standardized data, preprocessing with the mean and standard deviation is required when performing inference.

A Python program for classifying test data using the trained model can be found [here](classification_test.py).
