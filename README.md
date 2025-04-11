# Hand Pose Estimation Model and Dataset for Text Input Systems
This repository provides a dataset and pre-trained model for classifying hand poses based on the coordinates of the fingertips of the left hand, obtained from a hand tracker in an XR environment.

## Dataset

This dataset is intended for research on hand pose estimation. It consists of six CSV files containing 3D coordinates of each fingertip of the left hand, captured while performing different hand poses. These poses are categorized into six classes: Pose 1 to Pose 5, and Other.

 The hand data is standardized by using the position and orientation of the wrist base as the origin, based on the WristRoot joint obtained from [OVRSkeleton Documentation](https://developers.meta.com/horizon/reference/unity/v69/class_o_v_r_skeleton/).

The hand shape represented by each class, along with the specific hand poses and the number of data samples included, are as follows:

### Pose 1: All fingers are bent.

The dataset consists of 5,000 samples of the hand pose shown in Figure 1-1, and 1,000 samples of the pose shown in Figure 1-2.

### Pose 2: Only the thumb is extended.

The dataset consists of 5,000 samples of the hand pose shown in Figure 2-1.

### Pose 3: The thumb and index finger are extended.

The dataset consists of 5,000 samples of the hand pose shown in Figure 3-1.

### Pose 4: The thumb through the middle finger are extended.

The dataset consists of 5,000 samples of the hand pose shown in Figure 4-1.

### Pose 5: The thumb through the ring finger, or all fingers, are extended.

The dataset consists of 5,000 samples of the hand pose shown in Figure 5-1, and 1,000 samples of the pose shown in Figure 5-2.

### Other: Hand poses that do not belong to Pose 1 through Pose 5.

The dataset includes 1,000 samples for each of the poses shown in Figure 6-1.

## Pre-trained Models


