import logging
import pandas as pd
import numpy as np
from sklearn.utils import resample
import tensorflow as tf
from sklearn import preprocessing
import matplotlib
matplotlib.use('Agg') # Use a non-interactive backend, good for scripts/Docker
import matplotlib.pyplot as plt
# Import required libraries
from keras.models import Sequential

from keras.layers import Dense,GRU,Embedding,Dropout,Flatten,Conv1D,MaxPooling1D,LSTM
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
# Import `train_test_split` from `sklearn.model_selection`
from sklearn.model_selection import train_test_split
import numpy as np
from swarmlearning.tf import SwarmCallback
import joblib
import matplotlib.pyplot as plt
import os
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, cohen_kappa_score, accuracy_score
# Import `StandardScaler` from `sklearn.preprocessing`
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from keras.callbacks import EarlyStopping
#input_size
# -> CIC-DDoS2019 82
# -> CIC-IDS2018 78

defaultMaxEpoch = 5
defaultMinPeers = 2

trainFileName = 'training_h2.csv'
testFileName = 'test.csv'
valFileName = 'vals.csv'

batch_size = 512

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



# def train_test(samples):
    
    
#     # Specify the data 
#     X=samples.iloc[:,0:(samples.shape[1]-1)]
    
#     # Specify the target labels and flatten the array
#     #y= np.ravel(amostras.type)
#     y= samples.iloc[:,-1]
    
#     # Split the data up in train and test sets
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    
#     return X_train, X_test, y_train, y_test


# normalize input data

def normalize_data(X_train,X_test, X_vals):
    
    
    # Define the scaler 
    #scaler = StandardScaler().fit(X_train)
    scaler = MinMaxScaler(feature_range=(-1, 1)).fit(X_train)
    
    # Scale the train set
    X_train = scaler.transform(X_train)
    
    # Scale the test set
    X_test = scaler.transform(X_test)
    
    # Scale the validation set
    X_vals = scaler.transform(X_vals)
    
    return X_train, X_test, X_vals

# Reshape data input

def format_3d(df):
    
    X = np.array(df)
    return np.reshape(X, (X.shape[0], X.shape[1], 1))

def format_2d(df):
    
    return np.array(df).flatten()

def compile_train(model, X_train, y_train, X_val, y_val, maxEpoch, minPeers, deep=True, model_name=None, graph_path=None):
    # Get model name if not provided
    if model_name is None:
        model_name = model.__class__.__name__
    
    if deep:
        model.compile(loss='binary_crossentropy',
                    optimizer='adam',
                    metrics=['accuracy'])
        
        # Prepare validation data specifically for adaptive data sharing
        # Ensure proper formatting for GRU model
        X_val_3d = format_3d(X_val)
        y_val_2d = format_2d(y_val)
        
        # Create validation dataset tuple for SwarmCallback
        Valdata = (X_val_3d, y_val_2d)
        
        # Monitor validation metrics during training
        print(f"Validation data shape for SwarmCallback: X={X_val_3d.shape}, y={y_val_2d.shape}")

        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=3,
            restore_best_weights=True
        )
        
        # Swarm learning callback with proper validation data
        swarm_callback = SwarmCallback(
            syncFrequency=1024,              # Sync after every 10 batches
            minPeers=minPeers,             # Minimum number of peers to sync
            useAdaptiveSync=False,          # Disable adaptive sync
            adsValData=Valdata,           # Properly formatted validation data
            adsValBatchSize=512,    # Use the global batch_size variable
            mergeMethod='coordmedian',            # Method for model merging
            # nodeWeightage=18,            # Weight for model averaging
            # Add logging to see what's happening during training
            logDir=os.path.join(os.getenv('SCRATCH_DIR', '/platform/scratch'), 'swarm_logs')
        )
        
        # Set logging level for better visibility
        swarm_callback.logger.setLevel(logging.DEBUG)
        
        # Format training data properly for GRU model
        X_train_3d = format_3d(X_train)
        y_train_2d = format_2d(y_train)
        
        # Train the model with SwarmCallback
        history = model.fit(
            X_train_3d, 
            y_train_2d, 
            epochs=maxEpoch, 
            batch_size=batch_size, 
            verbose=1,
            callbacks=[swarm_callback, early_stopping]
        )
        
        # # summarize history for accuracy
        # plt.figure(figsize=(10, 4))
        # plt.subplot(1, 2, 1)
        # plt.plot(history.history['accuracy'])  # Updated from 'acc' to 'accuracy'
        # plt.title('Model Accuracy')
        # plt.ylabel('Accuracy')
        # plt.xlabel('Epoch')
        # plt.legend(['train'], loc='upper left')
        
        # # summarize history for loss
        # plt.subplot(1, 2, 2)
        # plt.plot(history.history['loss'])
        # plt.title('Model Loss')
        # plt.ylabel('Loss')
        # plt.xlabel('Epoch')
        # plt.legend(['train'], loc='upper left')
        # plt.tight_layout()
        # plt.show()

        # plt.savefig(graph_path)

        # print(model.metrics_names)

        print("History keys:", history.history.keys())

        
        return model
    else:
        # For non-deep learning models
        model.fit(X_train, y_train)
        
        # Save the model using joblib
        try:
            model_path = os.path.join('models', f"{model_name}.joblib")
            joblib.dump(model, model_path)
            print(f"Model saved to {model_path}")
        except Exception as e:
            print(f"Warning: Could not save model. Error: {e}")
            print("Make sure 'joblib' is installed: pip install joblib")
    
    print('Model Compiled and Trained')
    return model

# Testing performance outcomes of the methods

def testes(model,X_test,y_test,y_pred, deep=True):
    if(deep==True): 
        score = model.evaluate(X_test, y_test,verbose=1)

        print(score)
    # Alguns testes adicionais
    #y_test = formatar2d(y_test)
    #y_pred = formatar2d(y_pred)
    # Import the modules from `sklearn.metrics`
    # Accuracy 
    acc = accuracy_score(y_test, y_pred)
    print('\nAccuracy')
    print(acc)
    # Precision 
    prec = precision_score(y_test, y_pred)#,average='macro')
    print('\nPrecision')
    print(prec)
    # Recall
    rec = recall_score(y_test, y_pred) #,average='macro')
    print('\nRecall')
    print(rec)
    # F1 score
    f1 = f1_score(y_test,y_pred) #,average='macro')
    print('\nF1 Score')
    print(f1)
    #average
    avrg = (acc+prec+rec+f1)/4
    print('\nAverage (acc, prec, rec, f1)')
    print(avrg)
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
    
def train_samples(input_file):    
    # UPSAMPLE OF NORMAL FLOWS
    samples = pd.read_csv(input_file, sep=',')

    # Split features and target
    X = samples.drop(' Label', axis=1)  # Features
    y = samples[' Label']               # Target variable

    # Recombine features and target
    combined_data = pd.concat([X, y], axis=1)

    # Separate benign and attack samples
    is_benign = combined_data[' Label'] == 0
    normal = combined_data[is_benign]
    ddos = combined_data[~is_benign]

    # Upsample benign to match number of attack samples
    normal_upsampled = resample(normal,
                                replace=True,
                                n_samples=len(ddos),
                                random_state=27)

    # Combine upsampled benign and attack samples
    upsampled = pd.concat([normal_upsampled, ddos])

    # Separate features and labels for training
    X_train = upsampled.iloc[:, :-1]
    y_train = upsampled.iloc[:, -1]

    input_size = (X_train.shape[1], 1)
    print('input_size:', input_size)
    print('X_train shape:', X_train.shape)

    # ✅ Print class distribution
    print("\nValue counts for 'Label' after upsampling:")
    print(upsampled[' Label'].value_counts())

    return X_train, y_train

def val_test_samples(input_file):
    tests_val = pd.read_csv(input_file, sep=',')

    # X_test = np.concatenate((X_test,(tests.iloc[:,0:(tests.shape[1]-1)]).to_numpy())) # testar 33% + dia de testes
    # y_test = np.concatenate((y_test,tests.iloc[:,-1]))

    X_test_val = tests_val.iloc[:,0:(tests_val.shape[1]-1)]                        
    y_test_val = tests_val.iloc[:,-1]

    print((y_test_val.shape))
    print((X_test_val.shape))

    # X_train, X_test = normalize_data(X_train,X_test)
    return X_test_val, y_test_val

def evaluate_model(model, X_test, y_test):
    # Combine features and labels into one DataFrame for balancing
    df = pd.concat([X_test.reset_index(drop=True), pd.Series(y_test, name='Label')], axis=1)

    # Separate benign (0) and attack (1 or others)
    normal = df[df['Label'] == 0]
    attack = df[df['Label'] != 0]

    # Upsample benign to match attack count
    normal_upsampled = resample(normal,
                                replace=True,
                                n_samples=len(attack),
                                random_state=27)

    # Combine back to balanced dataframe
    balanced_df = pd.concat([normal_upsampled, attack])

    # Shuffle balanced dataset to avoid ordering bias
    balanced_df = balanced_df.sample(frac=1, random_state=27).reset_index(drop=True)

    # Separate balanced features and labels
    y_balanced = balanced_df['Label']
    X_balanced = balanced_df.drop(columns=['Label'])

    # Predict using the model (assumes your format_3d function reshapes X properly)
    y_pred_prob = model.predict(format_3d(X_balanced))
    y_pred = y_pred_prob.round()

    # Prepare results dataframe
    results = pd.DataFrame(columns=['Method','Accuracy','Precision','Recall', 'F1_Score', 'Average','Normal_Detect_Rate','Atk_Detect_Rate'])

    # Calculate metrics - assumes your testes() and test_normal_atk() functions exist and work with these inputs
    acc, prec, rec, f1, avrg = testes(model, format_3d(X_balanced), y_balanced, y_pred)
    norm, atk = test_normal_atk(y_balanced, y_pred)

    new_row = pd.DataFrame([{
        'Method': 'GRU',
        'Accuracy': acc,
        'Precision': prec,
        'F1_Score': f1,
        'Recall': rec,
        'Average': avrg,
        'Normal_Detect_Rate': norm,
        'Atk_Detect_Rate': atk
    }])

    results = pd.concat([results, new_row], ignore_index=True)
    return results

def main():
    modelName = 'GRU'
    dataDir = os.getenv('DATA_DIR', '/platform/data')
    scratchDir = os.getenv('SCRATCH_DIR', '/platform/scratch')
    os.makedirs(scratchDir, exist_ok=True)

    maxEpoch = int(os.getenv('MAX_EPOCHS', str(defaultMaxEpoch)))
    minPeers = int(os.getenv('MIN_PEERS', str(defaultMinPeers)))

    # train_file = './01-12/export_dataframe_proc.csv'
    # test_file = './01-12/export_tests_proc.csv'
    # val_file = './01-12/export_vals_proc.csv'
    
    train_file = os.path.join(dataDir, trainFileName)
    test_file = os.path.join(dataDir, testFileName)
    val_file = os.path.join(dataDir, valFileName)


    X_train, y_train = train_samples(train_file)
    X_test, y_test = val_test_samples(test_file)
    X_val, y_val = val_test_samples(val_file)
    X_train, X_test, X_val = normalize_data(X_train, X_test, X_val)

    graph_path = os.path.join(scratchDir, modelName + '_training_graph.png')

     # ✅ Print preview of datasets
    print(f"\n--- Sample of Training Data ({train_file}) ---")
    print("X_train:\n", X_train[:3])
    print("y_train:\n", y_train[:3])

    print(f"\n--- Sample of Validation Data ({val_file}) ---")
    print("X_val:\n", X_val[:3])
    print("y_val:\n", y_val[:3])

    print(f"\n--- Sample of Test Data ({test_file}) ---")
    print("X_test:\n", X_test[:3])
    print("y_test:\n", y_test[:3])

    print('***** Starting model =', modelName)
    model_gru = GRU_model(X_train.shape[1])
    final_model = compile_train(model_gru,X_train,y_train, X_val, y_val, maxEpoch, minPeers, model_name='GRU', graph_path=graph_path)

    # Evaluate the model
    results = evaluate_model(final_model, X_test, y_test)
    print('***** Results:')
    print(results)

    model_path = os.path.join(scratchDir, modelName)
    final_model.save(os.path.join(model_path, modelName + '.h5'))
    print(f"Saved the trained model to: {model_path}")

if __name__ == "__main__":
    main()

