import logging
import os
import pickle
import pandas as pd
import numpy as np
import tensorflow as tf

import matplotlib
matplotlib.use('Agg')

from sklearn.utils import resample
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
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


#input_size
# -> CIC-DDoS2019 82
# -> CIC-IDS2018 78

def GRU_model(input_size):
   
    # Initialize the constructor
    model = Sequential()
    
    model.add(GRU(32, input_shape=(input_size,1), return_sequences=False)) #
    model.add(Dropout(0.5))    
    model.add(Dense(10, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    
    model.build()
    print(model.summary())
    
    return model

# compile and train learning model
def compile_train(model,X_train,y_train, X_val, y_val, maxEpochs, swarm_callback=None, deep=True):
    
    if(deep==True):
        model.compile(loss='binary_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])
        
        callbacks = []
        if swarm_callback is not None:
            callbacks.append(swarm_callback)

        model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val) if X_val is not None and y_val is not None else None,
            epochs=maxEpochs,
            batch_size=256,
            verbose=1,
            callbacks=callbacks
        )
        #model.fit(X_train, y_train,epochs=3)

        # # summarize history for accuracy
        # plt.plot(history.history['acc'])
        # plt.title('model accuracy')
        # plt.ylabel('accuracy')
        # plt.xlabel('epoch')
        # plt.legend(['train'], loc='upper left')
        # plt.savefig("model_accuracy.png")
        # # summarize history for loss
        # plt.plot(history.history['loss'])
        # plt.title('model loss')
        # plt.ylabel('loss')
        # plt.xlabel('epoch')
        # plt.legend(['train'], loc='upper left')
        # plt.savefig("model_loss.png")
        # plt.close()
        print(model.metrics_names)
    
    else:
        model.fit(X_train, y_train) #SVM, LR, GD
    
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
    avrg = (acc + prec + rec + f1) / 4

    print(f"\nAccuracy: {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall: {rec:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"Average (acc, prec, rec, f1): {avrg:.4f}")

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
    return acc, prec, rec, f1, avrg

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

def load_and_prepare_data(train_path, test_path, scaler_path):
    # Load full dataset
    samples = pd.read_csv(train_path, sep=',')
    X_train, X_val, y_train, y_val = train_test(samples)

    # Concatenate X and y to perform upsampling
    X = pd.concat([X_train, y_train], axis=1)

    # Split by label
    normal = X[X[' Label'] == 0]
    ddos = X[X[' Label'] != 0]

    # Upsample minority class
    normal_upsampled = resample(
        normal,
        replace=True,
        n_samples=len(ddos),
        random_state=27
    )

    upsampled = pd.concat([normal_upsampled, ddos])
    X_train = upsampled.iloc[:, :-1]
    y_train = upsampled.iloc[:, -1]

    print("Counts after upsampling:")
    print(y_train.value_counts())

    # Load separate test day dataset
    test_data = pd.read_csv(test_path, sep=',')
    X_test = test_data.iloc[:, :-1]
    y_test = test_data.iloc[:, -1]

    print("Test set shape:", X_test.shape, y_test.shape)
    print("Counts in final test set:")
    print(y_test.value_counts())

    print("Validation set shape:", X_val.shape, y_val.shape)
    print("Counts in final validation set:")
    print(y_val.value_counts())

    # Normalize the features
    X_train, X_test, X_val = normalize_data(X_train, X_test, X_val, scaler_path)

    return X_train, X_test, y_train, y_test, X_val, y_val


def train_and_evaluate(X_train, y_train, X_test, y_test, X_val, y_val, maxEpochs, minPeers, save_path):
    
    swarm_callback = SwarmCallback(
        syncFrequency=256,  # Number of training samples after which peers sync their model weights
        minPeers=minPeers,  # Minimum number of active peers required to perform synchronization
        useAdaptiveSync=False,  # Disable adaptive sync; use fixed sync frequency instead
        adsValData=(format_3d(X_val), y_val),  # Validation data for Adaptive Sync and model merging decision
        adsValBatchSize=128,  # Batch size used during validation at sync points
        mergeMethod='mean',  # Aggregation method to merge model weights from different peers
        node_weightage=1,  # Equal weightage given to this node's model when averaging
        logDir=os.path.join(
        os.getenv('SCRATCH_DIR', '/platform/scratch'),'swarm_logs')  # Directory path to store Swarm logs
    )

    # Set logging level for better visibility
    swarm_callback.logger.setLevel(logging.DEBUG)

    model = GRU_model(X_train.shape[1])
    model = compile_train(model, format_3d(X_train), y_train, format_3d(X_val), y_val, maxEpochs, swarm_callback, deep=True)

    y_pred = model.predict(format_3d(X_test)).round()
    norm, atk = test_normal_atk(y_test, y_pred)
    acc, prec, rec, f1, avrg = testes(model, format_3d(X_test), y_test, y_pred, True)

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
        'Atk_Detect_Rate': atk
    }])

    return results


defaultMaxEpoch = 10
defaultMinPeers = 2

trainFileName = 'h2_lite_true_proc.csv'
testFileName = 'test_lite_true_proc.csv'


def main():

    dataDir = os.getenv('DATA_DIR', '/platform/data')
    scratchDir = os.getenv('SCRATCH_DIR', '/platform/scratch')
    os.makedirs(scratchDir, exist_ok=True)

    # Save the trained model
    save_path = os.path.join(scratchDir, 'gru_model.h5')

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
    results = train_and_evaluate(X_train, y_train, X_test, y_test, X_val, y_val, maxEpoch, minPeers, save_path)

    results.to_csv('gru_ddos_results.csv', index=False)
    print("Results saved to 'gru_ddos_results.csv'")
    print(results)

if __name__ == '__main__':
    main()