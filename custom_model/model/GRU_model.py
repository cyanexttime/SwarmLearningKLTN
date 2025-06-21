import logging
import os
import pickle
import pandas as pd
import numpy as np
import tensorflow as tf

import matplotlib
matplotlib.use('Agg')

from sklearn.utils import resample
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc, mean_absolute_error, mean_squared_error
import seaborn as sns
import matplotlib.pyplot as plt

from keras.models import Sequential
from sklearn.model_selection import train_test_split
from keras.layers import Dense, GRU, Dropout
from sklearn.preprocessing import MinMaxScaler
from swarmlearning.tf import SwarmCallback

def GRU_model(input_size):
    model = Sequential()
    model.add(GRU(32, input_shape=(input_size, 1), return_sequences=False))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.build()
    print(model.summary())
    return model

def compile_train(model, X_train, y_train, X_val, y_val, maxEpochs, swarm_callback=None, deep=True):
    history = None
    if deep:
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        callbacks = []
        if swarm_callback is not None:
            callbacks.append(swarm_callback)

        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=maxEpochs,
            batch_size=256,
            verbose=1,
            callbacks=callbacks
        )
        print(model.metrics_names)
    else:
        model.fit(X_train, y_train)
    print('Model Compiled and Trained')
    return model, history

def testes(model, X_test, y_test, y_pred=None, deep=True, threshold=0.8):
    if deep:
        score = model.evaluate(X_test, y_test, verbose=1)
        print('Model evaluation score:', score)

    if y_pred is None:
        y_score = model.predict(X_test)
        y_score = y_score[:, 0] if y_score.ndim > 1 else y_score
        y_pred = (y_score >= threshold).astype(int)
    else:
        y_score = model.predict(X_test)
        y_score = y_score[:, 0] if y_score.ndim > 1 else y_score

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    avrg = (acc + prec + rec + f1) / 4
    mae = mean_absolute_error(y_test, y_score)
    mse = mean_squared_error(y_test, y_score)
    rmse = np.sqrt(mse)

    print(f"\nAccuracy: {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall: {rec:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"Average: {avrg:.4f}")
    print(f"MAE: {mae:.4f}, MSE: {mse:.4f}, RMSE: {rmse:.4f}")

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Normal", "Attack"], yticklabels=["Normal", "Attack"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.savefig("confusion_matrix.png")
    plt.close()

    fpr, tpr, _ = roc_curve(y_test, y_score)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(6, 4))
    plt.plot(fpr, tpr, label=f"ROC curve (AUC = {roc_auc:.2f})", color='darkorange')
    plt.plot([0, 1], [0, 1], linestyle='--', color='navy')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.grid(True)
    plt.savefig("roc_curve.png")
    plt.close()

    return acc, prec, rec, f1, avrg, mae, mse, rmse

def test_normal_atk(y_test, y_pred):
    df = pd.DataFrame({'y_test': y_test, 'y_pred': y_pred})
    normal = len(df[df['y_test'] == 0])
    atk = len(df) - normal
    wrong = df[df['y_test'] != df['y_pred']]
    normal_detect_rate = (normal - len(wrong[wrong['y_test'] == 0])) / normal
    atk_detect_rate = (atk - len(wrong[wrong['y_test'] == 1])) / atk
    return normal_detect_rate, atk_detect_rate

def train_test(samples):
    data = samples.copy()
    class_0 = data[data.iloc[:, -1] == 0]
    class_1 = data[data.iloc[:, -1] == 1]
    train_0, val_0 = train_test_split(class_0, test_size=0.25, random_state=42)
    train_1, val_1 = train_test_split(class_1, test_size=0.25, random_state=42)
    train_data = pd.concat([train_0, train_1]).sample(frac=1, random_state=42).reset_index(drop=True)
    val_data = pd.concat([val_0, val_1]).sample(frac=1, random_state=42).reset_index(drop=True)
    X_train, y_train = train_data.iloc[:, :-1], train_data.iloc[:, -1]
    X_val, y_val = val_data.iloc[:, :-1], val_data.iloc[:, -1]
    return X_train, X_val, y_train, y_val

def normalize_data(X_train, X_test, X_val, scaler_path='scaler.pkl'):
    scaler = MinMaxScaler(feature_range=(-1, 1)).fit(X_train)
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    return scaler.transform(X_train), scaler.transform(X_test), scaler.transform(X_val)

def format_3d(df):
    X = np.array(df)
    return np.reshape(X, (X.shape[0], X.shape[1], 1))

def load_and_prepare_data(train_path, test_path, scaler_path):
    samples = pd.read_csv(train_path)
    X_train, X_val, y_train, y_val = train_test(samples)
    X = pd.concat([X_train, y_train], axis=1)
    normal = X[X[' Label'] == 0]
    ddos = X[X[' Label'] != 0]
    normal_upsampled = resample(normal, replace=True, n_samples=len(ddos), random_state=27)
    upsampled = pd.concat([normal_upsampled, ddos])
    X_train, y_train = upsampled.iloc[:, :-1], upsampled.iloc[:, -1]
    test_data = pd.read_csv(test_path)
    X_test, y_test = test_data.iloc[:, :-1], test_data.iloc[:, -1]
    X_train, X_test, X_val = normalize_data(X_train, X_test, X_val, scaler_path)
    return X_train, X_test, y_train, y_test, X_val, y_val

def train_and_evaluate(X_train, y_train, X_test, y_test, X_val, y_val, maxEpochs, minPeers, save_path):
    scratch_dir = os.getenv('SCRATCH_DIR', '/platform/scratch')
    swarm_callback = SwarmCallback(
        syncFrequency=1024,
        minPeers=minPeers,
        useAdaptiveSync=False,
        adsValData=(format_3d(X_val), y_val),
        adsValBatchSize=512,
        mergeMethod='mean',
        node_weightage=1,
        logDir=os.path.join(scratch_dir, 'swarm_logs')
    )
    swarm_callback.logger.setLevel(logging.DEBUG)
    model = GRU_model(X_train.shape[1])
    model, history = compile_train(model, format_3d(X_train), y_train, format_3d(X_val), y_val, maxEpochs, swarm_callback)

    # Plot training history
    if history is not None:
        hist = history.history
        plt.figure(figsize=(12, 5))

        if 'accuracy' in hist:
            plt.subplot(1, 2, 1)
            plt.plot(hist['accuracy'], label='Train Accuracy')
            if 'val_accuracy' in hist:
                plt.plot(hist['val_accuracy'], label='Validation Accuracy')
            plt.title('Accuracy over Epochs')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy')
            plt.legend()

        if 'loss' in hist:
            plt.subplot(1, 2, 2)
            plt.plot(hist['loss'], label='Train Loss')
            if 'val_loss' in hist:
                plt.plot(hist['val_loss'], label='Validation Loss')
            plt.title('Loss over Epochs')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()

        plt.tight_layout()
        plot_path = os.path.join(scratch_dir, 'training_history.png')
        plt.savefig(plot_path)
        print(f"Training history plot saved to: {plot_path}")
        plt.close()

    y_pred = model.predict(format_3d(X_test)).round()
    norm, atk = test_normal_atk(y_test, y_pred)
    acc, prec, rec, f1, avrg, mae, mse, rmse = testes(model, format_3d(X_test), y_test, y_pred, True)
    model.save(save_path)
    print(f"Model saved to {save_path}")

    results = pd.DataFrame([{
        'Method': 'GRU',
        'Accuracy': acc,
        'Precision': prec,
        'Recall': rec,
        'F1_Score': f1,
        'Average': avrg,
        'Normal_Detect_Rate': norm,
        'Atk_Detect_Rate': atk,
        'MAE': mae,
        'MSE': mse,
        'RMSE': rmse,
    }])
    return results

defaultMaxEpoch = 5
defaultMinPeers = 2
trainFileName = 'h1_lite_true_proc_added.csv'
testFileName = 'test_lite_true_proc.csv'

def main():
    dataDir = os.getenv('DATA_DIR', '/platform/data')
    scratchDir = os.getenv('SCRATCH_DIR', '/platform/scratch')
    os.makedirs(scratchDir, exist_ok=True)
    save_path = os.path.join(scratchDir, 'gru_model.h5')
    maxEpoch = int(os.getenv('MAX_EPOCHS', str(defaultMaxEpoch)))
    minPeers = int(os.getenv('MIN_PEERS', str(defaultMinPeers)))
    train_file = os.path.join(dataDir, trainFileName)
    test_file = os.path.join(dataDir, testFileName)

    print("Loading and preparing data...")
    X_train, X_test, y_train, y_test, X_val, y_val = load_and_prepare_data(train_file, test_file, scaler_path=os.path.join(scratchDir, 'scaler.pkl'))
    print("Data loaded and prepared.")
    print("Training and evaluating GRU model...")
    results = train_and_evaluate(X_train, y_train, X_test, y_test, X_val, y_val, maxEpoch, minPeers, save_path)
    results.to_csv(os.path.join(scratchDir, 'gru_ddos_results.csv'), index=False)
    print("Results saved to 'gru_ddos_results.csv'")
    print(results)

if __name__ == '__main__':
    main()
