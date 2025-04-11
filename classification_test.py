import numpy as np
import pandas as pd
import onnx
import onnxruntime as ort
import os
import random

filepath = os.path.join(os.path.dirname(os.path.abspath(__file__)), "MLData") + "\\"
label_names = ["Pose1_data", "Pose2_data", "Pose3_data", "Pose4_data", "Pose5_data", "Other_data"]

mean = np.array([-52.92750971,  20.67229755, -27.46303961, -67.47224812,  19.05142937,
                 -11.79876146, -60.15714922,  21.38782248,  -1.28671012, -48.38585485,
                 19.5798139,    5.58100285, -43.40979635,  18.41535248,  14.2151645])

std = np.array([5.78268672, 12.91420177, 19.31955857, 23.83113147, 11.2134667,   4.90108966,
                27.12181405, 12.35284633,  4.4784133,  23.97315656, 11.19723229,  5.98329016,
                19.5786793,   8.78913735, 10.50512291])

test_num = 100

def load_data():
    data_list = []
    labels_list = []

    for label, name in enumerate(label_names):
        df = pd.read_csv(filepath + name + ".csv")
        scaled_data = (df.values - mean) / std
        data_list.append(scaled_data)
        labels_list.extend([label] * len(df))

    all_data = np.concatenate(data_list, axis=0)
    labels = np.array(labels_list)

    combined = list(zip(all_data, labels))
    random.shuffle(combined)
    all_data, labels = zip(*combined)
    
    return np.array(all_data), np.array(labels)

def load_model(model_path):
    session = ort.InferenceSession(model_path)
    return session

def predict(model, data):
    input_name = model.get_inputs()[0].name
    outputs = model.run(None, {input_name: data.astype(np.float32)})
    return np.argmax(outputs[0], axis=1)

def main():
    X, y = load_data()

    X_test, y_test = X[:test_num], y[:test_num]

    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "trained_model.onnx")
    model = load_model(model_path)

    predictions = predict(model, X_test)

    for pred, true_label in zip(predictions, y_test):
        print(f"Prediction: {pred}, True label: {true_label}")

if __name__ == "__main__":
    main()