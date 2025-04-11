# Hand Pose Estimation Model and Dataset for Text Input Systems
This repository provides a dataset and pre-trained model for classifying hand poses based on the coordinates of the fingertips of the left hand, obtained from a hand tracker in an XR environment.

## Dataset

This dataset is intended for research on hand pose estimation. It consists of six CSV files containing 3D coordinates of each fingertip of the left hand, captured while performing different hand poses. These poses are categorized into six classes: Pose 1 to Pose 5, and Other.

 The hand data is standardized by using the position and orientation of the wrist base as the origin, based on the WristRoot joint obtained from [OVRSkeleton Documentation](https://developers.meta.com/horizon/reference/unity/v69/class_o_v_r_skeleton/).

The hand shape represented by each class, along with the specific hand poses and the number of data samples included, are as follows:

### Pose 1: All fingers are bent.

The dataset consists of 5,000 samples of the hand pose shown in Figure 1-1, and 1,000 samples of the pose shown in Figure 1-2.

Figure 1-1

![Pose1-1](https://github.com/user-attachments/assets/8a370059-32ef-40c9-893c-f6b20a622bae)


Figure 1-2

![Pose1-2](https://github.com/user-attachments/assets/f86749a9-1f4f-4cad-98a5-b754784cca1f)


### Pose 2: Only the thumb is extended.

The dataset consists of 5,000 samples of the hand pose shown in Figure 2-1.

Figure 2-1

![Pose2](https://github.com/user-attachments/assets/09c944e9-1eb0-4002-a105-623026dffea2)


### Pose 3: The thumb and index finger are extended.

The dataset consists of 5,000 samples of the hand pose shown in Figure 3-1.

Figure 3-1

![Pose3](https://github.com/user-attachments/assets/8a18e88a-38d2-4971-b241-77d9f4f35eb7)


### Pose 4: The thumb through the middle finger are extended.

The dataset consists of 5,000 samples of the hand pose shown in Figure 4-1.

Figure 4-1

![Pose4](https://github.com/user-attachments/assets/b694fdd4-b887-4a57-b34b-abd8c8057ac7)


### Pose 5: The thumb through the ring finger, or all fingers, are extended.

The dataset consists of 5,000 samples of the hand pose shown in Figure 5-1, and 1,000 samples of the pose shown in Figure 5-2.

Figure 5-1

![Pose5-1](https://github.com/user-attachments/assets/2f1d05c8-d5db-4f7f-ac89-7bda848f4d1b)


Figure 5-2

![Pose5-2](https://github.com/user-attachments/assets/42b2b877-2331-4981-9f06-774791e88e7e)


### Other: Hand poses that do not belong to Pose 1 through Pose 5.

The dataset includes 1,000 samples for each of the poses shown in Figure 6-1.
It consists of 14 hand poses, including poses inspired by Japanese fingerspelling and naturally relaxed hand shapes.

Figure 6-1

![otherPose](https://github.com/user-attachments/assets/3249d0c8-3fb0-4efe-a128-1dac63e2850d)


## Pre-trained Models

The ONNX-format pre-trained model provided in this repository performs 6-class classification based on the 3D coordinates of each fingertip of the left hand. Since the model was built using standardized data, preprocessing with the mean and standard deviation is required when performing inference.

A Python program for classifying test data using the trained model can be found [here](classification_test.py).
