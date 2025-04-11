import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, confusion_matrix, ConfusionMatrixDisplay
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping


filepath = os.path.join(os.path.dirname(os.path.abspath(__file__)), "MLData") + "\\"
label_names = ["Pose1_data", "Pose2_data", "Pose3_data", "Pose4_data", "Pose5_data", "Other_data"]
input_Dim = 15
num_Classes = len(label_names)

def load_and_prepare_data():
    data_list = []
    labels_list = []

    for label, name in enumerate(label_names):
        df = pd.read_csv(filepath + name + ".csv")
        data_list.append(df)
        labels_list.extend([label] * len(df))

    all_data = pd.concat(data_list, ignore_index=True)
    labels = np.array(labels_list)

    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(all_data)
    print("mean:", scaler.mean_)
    print("sd:", scaler.scale_)

    return scaled_data, labels, scaler.mean_, scaler.scale_

def build_model(input_dim, num_classes):
    model = Sequential([
        Dense(32, activation='relu', input_dim=input_dim, name='dense_input'),
        Dense(16, activation='relu', name='dense_hidden'),
        Dense(num_classes, activation='softmax', name='dense_output')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def plot_training_history(history):
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Loss Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Accuracy Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.show()

def evaluate_model(model, X_test, y_test):
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f'Test Loss: {loss:.4f}, Test Accuracy: {accuracy:.4f}')

    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)

    cm = confusion_matrix(y_test, y_pred_classes)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap=plt.cm.Blues)
    for text in disp.text_.ravel():
        text.set_fontsize(10)
    plt.title("Confusion Matrix")
    plt.show()

    f1 = f1_score(y_test, y_pred_classes, average='weighted')
    print(f'F1 Score: {f1:.4f}')

def main():
    X, y, mean, std = load_and_prepare_data()

    X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=1/9, random_state=42)

    model = build_model(input_Dim, num_Classes)

    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=2,
        min_delta=0.0015,
        verbose=1,
        restore_best_weights=True
    )

    history = model.fit(
        X_train, y_train,
        epochs=100,
        batch_size=32,
        validation_data=(X_val, y_val),
        callbacks=[early_stopping]
    )

    plot_training_history(history)
    evaluate_model(model, X_test, y_test)

if __name__ == "__main__":
    main()