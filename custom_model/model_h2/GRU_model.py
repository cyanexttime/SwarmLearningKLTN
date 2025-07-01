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
# Import required libraries
from keras.models import Sequential
# Import `train_test_split` from `sklearn.model_selection`
from sklearn.model_selection import train_test_split
import numpy as np

from keras.layers import Dense,GRU,Embedding,Dropout,Flatten,Conv1D,MaxPooling1D,LSTM
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from swarmlearning.tf import SwarmCallback

from tensorflow.keras.callbacks import History
from tensorflow.keras.callbacks import Callback

#input_size
# -> CIC-DDoS2019 82
# -> CIC-IDS2018 78



class SyncLossLogger(Callback):
    def __init__(self, sync_freq):
        super().__init__()
        self.sync_freq = sync_freq
        self.sync_losses = []
        self.sync_batches = []
        self.batch_count = 0

    def on_train_batch_end(self, batch, logs=None):
        self.batch_count += 1
        if self.batch_count % self.sync_freq == 0:
            self.sync_losses.append(logs.get('loss'))
            self.sync_batches.append(self.batch_count)


# def GRU_model(input_size):
   
#     # Initialize the constructor
#     model = Sequential()
    
#     model.add(GRU(32, input_shape=(input_size,1), return_sequences=False)) #
#     model.add(Dropout(0.5))    
#     model.add(Dense(10, activation='relu'))
#     model.add(Dense(1, activation='sigmoid'))
    
#     model.build()
#     print(model.summary())
    
#     return model

def CNN_model(input_size):
    model = Sequential()
    model.add(Conv1D(filters=64, kernel_size=8, activation='relu', padding='same', input_shape=(input_size, 1)))
    model.add(MaxPooling1D(2))
    model.add(Conv1D(filters=32, kernel_size=5, activation='relu', padding='same'))
    model.add(MaxPooling1D(2))
    model.add(Conv1D(filters=16, kernel_size=3, activation='relu', padding='same'))
    model.add(MaxPooling1D(2))

    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(10, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    model.build()
    print(model.summary())
    return model

# def GRU_model(input_shape):

#     model = Sequential()
#     model.add(GRU(32, input_shape=input_shape, return_sequences=False))
#     model.add(Dropout(0.5))
#     model.add(Dense(10, activation='relu'))
#     model.add(Dense(1, activation='sigmoid'))

#     model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
#     print(model.summary())
#     return model

# def reshape_to_sequence(X, y, sequence_length=10):
#     X_seq, y_seq = [], []
#     for i in range(len(X) - sequence_length + 1):
#         X_seq.append(X[i:i + sequence_length])
#         y_seq.append(y[i + sequence_length - 1])
#     return np.array(X_seq), np.array(y_seq)

# compile and train learning model
def compile_train(model, X_train, y_train, X_val, y_val, maxEpochs, swarm_callback=None, deep=True, plot_save_path=None):
    # Callbacks
    history_callback = History()
    sync_logger = SyncLossLogger(sync_freq=1024)
    if deep:
        model.compile(
            loss='binary_crossentropy',
            optimizer='adam',
            metrics=['accuracy']
        )
        
        callbacks = [history_callback]
        if swarm_callback is not None:
            callbacks.append(swarm_callback)

        model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val) if X_val is not None and y_val is not None else None,
            epochs=maxEpochs,
            batch_size=512,
            verbose=1,
            callbacks=callbacks
        )

        # Plotting training and validation metrics
        hist = history_callback.history
        print("History keys:", hist.keys())

        acc_key = 'accuracy'
        val_acc_key = 'val_accuracy'
        loss_key = 'loss'
        val_loss_key = 'val_loss'

        plt.figure(figsize=(12, 5))

        # Accuracy plot
        if acc_key in hist:
            plt.subplot(1, 2, 1)
            plt.plot(hist[acc_key], label='Training Accuracy')
            if val_acc_key in hist:
                plt.plot(hist[val_acc_key], label='Validation Accuracy', linestyle='--')
            plt.title('Accuracy Over Epochs')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy')
            plt.legend()

            # Tick customization (manual)
            epochs = len(hist[acc_key])
            plt.xticks(np.arange(0, epochs + 1, step=1))  # x ticks every epoch
            plt.yticks(np.arange(0.9, 1.01, step=0.05))   # y ticks from 0.9 to 1.0 with step 0.05
            plt.ylim(0.9, 1.0)                            # y-axis limits between 0.9 and 1.0
        else:
            print("Accuracy key not found. Skipping accuracy plot.")

        # Loss plot
        if loss_key in hist:
            plt.subplot(1, 2, 2)
            plt.plot(hist[loss_key], label='Training Loss', color='orange')
            if val_loss_key in hist:
                plt.plot(hist[val_loss_key], label='Validation Loss', color='red', linestyle='--')
            plt.title('Loss Over Epochs')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()

            # Tick customization (manual)
            epochs = len(hist[loss_key])
            plt.xticks(np.arange(0, epochs + 1, step=1))   # x ticks every epoch

            max_loss = max(max(hist[loss_key]), max(hist.get(val_loss_key, [0])))
            plt.yticks(np.arange(0.0, 0.4, step=0.05))  # y ticks based on max loss
        else:
            print("Loss key not found. Skipping loss plot.")
            print(model.metrics_names)
        
        if plot_save_path is not None:
            os.makedirs(plot_save_path, exist_ok=True)
            plot_path = os.path.join(plot_save_path, 'training_plots.png')
            plt.savefig(plot_path)
            print(f'Training plots saved to {plot_path}')
        else:
            plt.show()
        
        if len(sync_logger.sync_losses) > 0:
            plt.figure(figsize=(6, 4))
            plt.plot(sync_logger.sync_batches, sync_logger.sync_losses, marker='o', linestyle='-', color='blue', label='Local Loss at Sync')
            plt.title('Local Loss at Each Swarm Sync Point')
            plt.xlabel('Batch Number')
            plt.ylabel('Loss')
            plt.grid(True)
            plt.legend()
            sync_loss_path = os.path.join(plot_save_path, 'sync_loss_plot.png') if plot_save_path else 'sync_loss_plot.png'
            plt.savefig(sync_loss_path)
            print(f'Sync loss plot saved to {sync_loss_path}')
            plt.close()

    else:
        model.fit(X_train, y_train)  # For non-deep models (e.g., SVM)

    print('Model Compiled and Trained')
    return model

def testes(model, X_test, y_test, y_pred=None, deep=True, threshold=0.8):
    # Evaluate deep learning model if applicable
    if deep:
        score = model.evaluate(X_test, y_test, verbose=1)
        print('Model evaluation score:', score)

    # Predict probabilities and apply threshold if y_pred not provided
    if y_pred is None:
        if hasattr(model, "predict_proba"):
            y_score = model.predict_proba(X_test)[:, 1]
        else:
            y_score = model.predict(X_test)
            if y_score.ndim > 1 and y_score.shape[1] > 1:
                y_score = y_score[:, 1]  # Assume binary classification
            else:
                y_score = y_score.ravel()

        y_pred = (y_score >= threshold).astype(int)
    else:
        # If y_pred is provided, derive y_score if needed for ROC
        if hasattr(model, "predict_proba"):
            y_score = model.predict_proba(X_test)[:, 1]
        else:
            y_score = model.predict(X_test)
            if y_score.ndim > 1 and y_score.shape[1] > 1:
                y_score = y_score[:, 1]
            else:
                y_score = y_score.ravel()

    # Compute evaluation metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    print(f"\nAccuracy: {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall: {rec:.4f}")
    print(f"F1 Score: {f1:.4f}")

    # # Regression-like metrics (based on predicted probabilities)
    # mae = mean_absolute_error(y_test, y_score)
    # mse = mean_squared_error(y_test, y_score)
    # rmse = np.sqrt(mse)

    # print(f"\n--- Regression-style Metrics on Probabilities ---")
    # print(f"MAE:  {mae:.4f}")
    # print(f"MSE:  {mse:.4f}")
    # print(f"RMSE: {rmse:.4f}")

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Normal", "Attack"], yticklabels=["Normal", "Attack"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.savefig("confusion_matrix.png")
    plt.close()

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_score)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(6, 4))
    plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic")
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.savefig("roc_curve.png")
    plt.close()

    plt.close('all')
    return acc, prec, rec, f1

def test_normal_atk(y_test,y_pred):
    df = pd.DataFrame()
    df['y_test'] = y_test
    df['y_pred'] = y_pred
    
    normal = len(df.query('y_test == 0'))
    atk = len(y_test)-normal
    
    wrong = df.query('y_test != y_pred')
    
    normal_detect_rate = (normal - wrong.groupby('y_test').count().iloc[0][0]) / normal
    atk_detect_rate = (atk - wrong.groupby('y_test').count().iloc[1][0]) / atk
    
    #print(normal_detect_rate,atk_detect_rate)
    
    return normal_detect_rate, atk_detect_rate
    


def train_test(samples):
    # # Separate features and labels
    # X = samples.iloc[:, :-1]
    # y = samples.iloc[:, -1]

    # Combine for easy manipulation
    data = samples.copy()
    
    # Separate by class
    class_0 = data[data.iloc[:, -1] == 0]
    class_1 = data[data.iloc[:, -1] == 1]
    
    # Split each class into train and val (25% to validation)
    train_0, val_0 = train_test_split(class_0, test_size=0.25, random_state=42)
    train_1, val_1 = train_test_split(class_1, test_size=0.25, random_state=42)
    
    # Combine training and validation sets
    train_data = pd.concat([train_0, train_1], ignore_index=True)
    val_data = pd.concat([val_0, val_1], ignore_index=True)
    
    # Shuffle both sets (optional but recommended)
    train_data = train_data.sample(frac=1, random_state=42).reset_index(drop=True)
    val_data = val_data.sample(frac=1, random_state=42).reset_index(drop=True)

    # Split into features and labels
    X_train = train_data.iloc[:, :-1]
    y_train = train_data.iloc[:, -1]
    X_val = val_data.iloc[:, :-1]
    y_val = val_data.iloc[:, -1]
    
    return X_train, X_val, y_train, y_val


# normalize input data

def normalize_data(X_train,X_test,X_val, scaler_path='scaler.pkl'):
    # Import `StandardScaler` from `sklearn.preprocessing`
    
    scaler = MinMaxScaler(feature_range=(-1, 1)).fit(X_train)

    # Save the scaler to file
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)

    # Transform the datasets
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    X_val = scaler.transform(X_val)
    
    return X_train, X_test, X_val


def format_3d(df):
    
    X = np.array(df)
    return np.reshape(X, (X.shape[0], X.shape[1], 1))

def format_2d(df):
    
    X = np.array(df)
    return np.reshape(X, (X.shape[0], X.shape[1]))

# def load_and_prepare_data(train_path, test_path, scaler_path):
#     # Load full dataset
#     samples = pd.read_csv(train_path, sep=',')
#     X_train, X_val, y_train, y_val = train_test(samples)

#     # ==== UPSAMPLE TRAINING DATA ====
#     X = pd.concat([X_train, y_train], axis=1)
#     normal = X[X[' Label'] == 0]
#     ddos = X[X[' Label'] != 0]

#     normal_upsampled = resample(
#         normal,
#         replace=True,
#         n_samples=len(ddos),
#         random_state=27
#     )

#     upsampled = pd.concat([normal_upsampled, ddos])
#     X_train = upsampled.iloc[:, :-1]
#     y_train = upsampled.iloc[:, -1]

#     print("Counts after upsampling (train):")
#     print(y_train.value_counts())

#     # ==== UPSAMPLE VALIDATION DATA ====
#     val = pd.concat([X_val, y_val], axis=1)
#     val_normal = val[val[' Label'] == 0]
#     val_ddos = val[val[' Label'] != 0]

#     val_normal_upsampled = resample(
#         val_normal,
#         replace=True,
#         n_samples=len(val_ddos),
#         random_state=27
#     )

#     val_upsampled = pd.concat([val_normal_upsampled, val_ddos])
#     X_val = val_upsampled.iloc[:, :-1]
#     y_val = val_upsampled.iloc[:, -1]

#     print("Counts after upsampling (validation):")
#     print(y_val.value_counts())

#     # ==== TEST DATA ====
#     test_data = pd.read_csv(test_path, sep=',')
#     X_test = test_data.iloc[:, :-1]
#     y_test = test_data.iloc[:, -1]

#     print("Test set shape:", X_test.shape, y_test.shape)
#     print("Counts in final test set:")
#     print(y_test.value_counts())

#     print("Validation set shape:", X_val.shape, y_val.shape)
#     print("Counts in final validation set:")
#     print(y_val.value_counts())

#     # ==== NORMALIZATION ====
#     X_train, X_test, X_val = normalize_data(X_train, X_test, X_val, scaler_path)

#     return X_train, X_test, y_train, y_test, X_val, y_val

def load_and_prepare_data(train_path, test_path, scaler_path):
    # Load full dataset
    samples = pd.read_csv(train_path, sep=',')
    X_train, X_val, y_train, y_val = train_test(samples)

    # ==== NO UPSAMPLING ====
    print("Counts in training set:")
    print(y_train.value_counts())

    print("Counts in validation set:")
    print(y_val.value_counts())

    # ==== TEST DATA ====
    test_data = pd.read_csv(test_path, sep=',')
    X_test = test_data.iloc[:, :-1]
    y_test = test_data.iloc[:, -1]

    print("Test set shape:", X_test.shape, y_test.shape)
    print("Counts in final test set:")
    print(y_test.value_counts())

    print("Validation set shape:", X_val.shape, y_val.shape)
    print("Counts in final validation set:")
    print(y_val.value_counts())

    # ==== NORMALIZATION ====
    X_train, X_test, X_val = normalize_data(X_train, X_test, X_val, scaler_path)

    return X_train, X_test, y_train, y_test, X_val, y_val




def train_and_evaluate(X_train, y_train, X_test, y_test, X_val, y_val, maxEpochs, minPeers, save_path, plot_save_path=None):
    swarm_callback = SwarmCallback(
        syncFrequency=333,
        minPeers=minPeers,
        useAdaptiveSync=False,
        adsValData=(format_3d(X_val), y_val),
        adsValBatchSize=512,
        mergeMethod='geomedian',
        node_weightage=1,
        logDir=os.path.join(os.getenv('SCRATCH_DIR', '/platform/scratch'), 'swarm_logs')
    )

    swarm_callback.logger.setLevel(logging.DEBUG)

    model = CNN_model(X_train.shape[1])
    model = compile_train(model, format_3d(X_train), y_train, format_3d(X_val), y_val, maxEpochs, swarm_callback, deep=True, plot_save_path=plot_save_path)

    y_pred = model.predict(format_3d(X_test)).round()
    norm, atk = test_normal_atk(y_test, y_pred)
    acc, prec, rec, f1 = testes(model, format_3d(X_test), y_test, y_pred, True)

    model.save(save_path)
    print(f"Model saved to {save_path}")

    results = pd.DataFrame([{
        'Method': 'CNN',
        'Accuracy': acc,
        'Precision': prec,
        'Recall': rec,
        'F1_Score': f1,
        'Normal_Detect_Rate': norm,
        'Atk_Detect_Rate': atk,
    }])
    return results


# def train_and_evaluate(X_train, y_train, X_test, y_test, X_val, y_val, maxEpochs, minPeers, save_path, plot_save_path=None):

#     swarm_callback = SwarmCallback(
#         syncFrequency=1255,
#         minPeers=minPeers,
#         useAdaptiveSync=False,
#         adsValData=(X_val, y_val),
#         adsValBatchSize=512,
#         mergeMethod='coordmedian',
#         node_weightage=1,
#         logDir=os.path.join(os.getenv('SCRATCH_DIR', '/platform/scratch'), 'swarm_logs')
#     )
#     swarm_callback.logger.setLevel(logging.DEBUG)

#     # Reshape input to sequences for GRU
#     X_train_seq, y_train_seq = reshape_to_sequence(X_train, y_train, sequence_length=10)
#     X_val_seq, y_val_seq = reshape_to_sequence(X_val, y_val, sequence_length=10)
#     X_test_seq, y_test_seq = reshape_to_sequence(X_test, y_test, sequence_length=10)

#     # Build and train model
#     input_shape = (X_train_seq.shape[1], X_train_seq.shape[2])
#     model = GRU_model(input_shape)
#     model = compile_train(model, X_train_seq, y_train_seq, X_val_seq, y_val_seq, maxEpochs, swarm_callback, deep=True, plot_save_path=plot_save_path)


#     # Evaluate model
#     y_pred = model.predict(X_test_seq).round()
#     acc = accuracy_score(y_test_seq, y_pred)
#     prec = precision_score(y_test_seq, y_pred)
#     rec = recall_score(y_test_seq, y_pred)
#     f1 = f1_score(y_test_seq, y_pred)

#     print("Evaluation results:")
#     print(f"Accuracy: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}, F1: {f1:.4f}")

#     model.save(save_path)
#     print(f"Model saved to {save_path}")

#     results = pd.DataFrame([{
#         'Method': 'GRU (10 flows)',
#         'Accuracy': acc,
#         'Precision': prec,
#         'Recall': rec,
#         'F1_Score': f1
#     }])
#     return results


defaultMaxEpoch = 10
defaultMinPeers = 2

trainFileName = 'h2_lite_true_proc_added.csv'
testFileName = 'test_lite_true_proc.csv'


def main():

    dataDir = os.getenv('DATA_DIR', '/platform/data')
    scratchDir = os.getenv('SCRATCH_DIR', '/platform/scratch')
    os.makedirs(scratchDir, exist_ok=True)

    # Save the trained model
    save_path = os.path.join(scratchDir, 'gru_model.h5')
    plot_save_path = os.path.join(scratchDir, 'plots')

    maxEpoch = int(os.getenv('MAX_EPOCHS', str(defaultMaxEpoch)))
    minPeers = int(os.getenv('MIN_PEERS', str(defaultMinPeers)))

    train_file = os.path.join(dataDir, trainFileName)
    test_file = os.path.join(dataDir, testFileName)

    # train_path = './01-12/train_set_1_proc.csv'
    # test_path = './01-12/test_set_proc.csv'

    print("Loading and preparing data...")
    X_train, X_test, y_train, y_test, X_val, y_val = load_and_prepare_data(train_file, test_file, scaler_path=os.path.join(scratchDir, 'scaler.pkl'))
    print("Data loaded and prepared.")

    print("Training and evaluating GRU model...")
    results = train_and_evaluate(X_train, y_train, X_test, y_test, X_val, y_val, maxEpoch, minPeers, save_path, plot_save_path)

    results.to_csv('gru_ddos_results.csv', index=False)
    print("Results saved to 'gru_ddos_results.csv'")
    print(results)

if __name__ == '__main__':
    main()