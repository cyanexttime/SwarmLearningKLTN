import pandas as pd
import numpy as np
from sklearn.utils import resample
from sklearn import preprocessing
# Import required libraries
from keras.models import Sequential

from keras.layers import Dense,GRU,Embedding,Dropout,Flatten,Conv1D,MaxPooling1D,LSTM
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
# Import `train_test_split` from `sklearn.model_selection`
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
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



def train_test(samples):
    
    
    # Specify the data 
    X=samples.iloc[:,0:(samples.shape[1]-1)]
    
    # Specify the target labels and flatten the array
    #y= np.ravel(amostras.type)
    y= samples.iloc[:,-1]
    
    # Split the data up in train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    
    return X_train, X_test, y_train, y_test


# normalize input data

def normalize_data(X_train,X_test):
    # Import `StandardScaler` from `sklearn.preprocessing`
    from sklearn.preprocessing import StandardScaler,MinMaxScaler
    
    # Define the scaler 
    #scaler = StandardScaler().fit(X_train)
    scaler = MinMaxScaler(feature_range=(-1, 1)).fit(X_train)
    
    # Scale the train set
    X_train = scaler.transform(X_train)
    
    # Scale the test set
    X_test = scaler.transform(X_test)
    
    return X_train, X_test

# Reshape data input

def format_3d(df):
    
    X = np.array(df)
    return np.reshape(X, (X.shape[0], X.shape[1], 1))

def format_2d(df):
    
    X = np.array(df)
    return np.reshape(X, (X.shape[0], X.shape[1]))

def compile_train(model, X_train, y_train, deep=True, model_name=None):
    """
    Compile and train the model, then save it to a models folder.
    
    Parameters:
    -----------
    model : model object
        The machine learning model to train
    X_train : array-like
        Training data features
    y_train : array-like
        Training data target
    deep : bool, default=True
        Whether this is a deep learning model requiring compilation
    model_name : str, default=None
        Name to use when saving the model. If None, will try to use model's class name.
    
    Returns:
    --------
    model : trained model object
    """
    import os
    
    # Create models directory if it doesn't exist
    if not os.path.exists('models'):
        os.makedirs('models')
        print("Created 'models' directory")
    
    # Get model name if not provided
    if model_name is None:
        model_name = model.__class__.__name__
    
    if deep:
        # For deep learning models
        model.compile(loss='binary_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])
        
        history = model.fit(X_train, y_train, epochs=10, batch_size=256, verbose=1)
        
        # summarize history for accuracy
        plt.figure(figsize=(10, 4))
        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'])  # Updated from 'acc' to 'accuracy'
        plt.title('Model Accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['train'], loc='upper left')
        
        # summarize history for loss
        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'])
        plt.title('Model Loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['train'], loc='upper left')
        plt.tight_layout()
        plt.show()

        print(model.metrics_names)
        
        # Save the deep learning model
        model_path = os.path.join('models', f"{model_name}.h5")
        model.save(model_path)
        print(f"Deep learning model saved to {model_path}")
    
    else:
        # For non-deep learning models
        model.fit(X_train, y_train)
        
        # Save the model using joblib
        try:
            import joblib
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
    from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, cohen_kappa_score, accuracy_score
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

    X_train, X_test, y_train, y_test = train_test(samples)


    #junta novamente pra aumentar o numero de normais
    X = pd.concat([X_train, y_train], axis=1)

    # separate minority and majority classes
    is_benign = X[' Label']==0 #base de dados toda junta

    normal = X[is_benign]
    ddos = X[~is_benign]

    # upsample minority
    normal_upsampled = resample(normal,
                            replace=True, # sample with replacement
                            n_samples=len(ddos), # match number in majority class
                            random_state=27) # reproducible results

    # combine majority and upsampled minority
    upsampled = pd.concat([normal_upsampled, ddos])

    # Specify the data 
    X_train=upsampled.iloc[:,0:(upsampled.shape[1]-1)]    #DDoS
    y_train= upsampled.iloc[:,-1]  #DDoS

    input_size = (X_train.shape[1], 1)
    print('input_size:',input_size)
    print('X_train:',X_train.shape[1])
    return X_train, y_train

def test_samples(input_file):
    tests = pd.read_csv(input_file, sep=',')

    # X_test = np.concatenate((X_test,(tests.iloc[:,0:(tests.shape[1]-1)]).to_numpy())) # testar 33% + dia de testes
    # y_test = np.concatenate((y_test,tests.iloc[:,-1]))

    X_test = tests.iloc[:,0:(tests.shape[1]-1)]                        
    y_test = tests.iloc[:,-1]

    print((y_test.shape))
    print((X_test.shape))

    # X_train, X_test = normalize_data(X_train,X_test)
    return X_test, y_test

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(format_3d(X_test)) 
    y_pred = y_pred.round()
    results = pd.DataFrame(columns=['Method','Accuracy','Precision','Recall', 'F1_Score', 'Average','Normal_Detect_Rate','Atk_Detect_Rate'])
    acc, prec, rec, f1, avrg = testes(model, format_3d(X_test), y_test, y_pred)
    norm, atk = test_normal_atk(y_test, y_pred)

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
    train_file = './01-12/export_dataframe_proc.csv'
    test_file = './01-12/export_tests_proc.csv'
    X_train, y_train = train_samples(train_file)
    X_test, y_test = test_samples(test_file)
    X_train, X_test = normalize_data(X_train,X_test)

    model_gru = GRU_model(82)
    model_gru = compile_train(model_gru,format_3d(X_train),y_train,model_name='GRU')

    evaluate_model(model_gru, X_test, y_test)

main()

