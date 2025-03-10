# importing the libraries
import pandas as pd

import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, LeakyReLU, Dropout, BatchNormalization
from keras.regularizers import l2
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from sklearn.model_selection import KFold

def ann(input_dim, hidden_layers=[12, 6], activation='relu', dropout_rate=0.1, lr_reg=0.001, stop=True):
    # define model
    model = Sequential()

    # Input layer
    model.add(Dense(hidden_layers[0], input_dim=input_dim, activation=activation, activity_regularizer=l2(lr_reg)))
    model.add(BatchNormalization())
    model.add(Dropout(dropout_rate))

    # Hidden layers
    for neurons in hidden_layers[1:]:
        model.add(Dense(neurons, activation=activation, activity_regularizer=l2(lr_reg)))
        model.add(BatchNormalization())
        model.add(Dropout(dropout_rate))

    # Output layer
    model.add(Dense(1, activation='linear', activity_regularizer=l2(lr_reg)))

    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mean_absolute_error'])

    # EarlyStopping
    if stop:
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    return model, early_stopping

def cross_validation_train(X, y, n_splits=5, epochs=100, batch_size=32):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    best_model = None
    best_history = None
    best_val_loss = float('inf')

    for train_index, val_index in kf.split(X):
        X_train, X_val = X[train_index], X[val_index]
        y_train, y_val = y[train_index], y[val_index]

        model, early_stopping = ann(input_dim=X_train.shape[1])

        history = model.fit(X_train, y_train, 
                            epochs=epochs, 
                            batch_size=batch_size, 
                            validation_data=(X_val, y_val), 
                            callbacks=[early_stopping], 
                            verbose=0)

        # Guardar el mejor modelo
        current_val_loss = model.evaluate(X_val, y_val)[0]
        if current_val_loss < best_val_loss:
            best_val_loss = current_val_loss
            best_model = model
            best_history = history.history

    # Puedes acceder al mejor modelo después de completar todas las iteraciones
    return best_model, best_history

# Defining a function to find the best parameters for ANN
def FunctionFindBestParams(X_train, y_train, X_val, y_val):
    
    # Defining the list of hyper parameters to try
    batch_size_list = [50, 100, 150, 200]
    epoch_list = [50, 100, 150, 200]
    regularizers = [0.001, 0.05, 0.01, 0.5, 0.1]
    n = len(batch_size_list)*len(epoch_list)*len(regularizers)
    
    SearchResultsData=pd.DataFrame(columns=['TrialNumber', 'batch', 'epochs', 'reg', 'val_loss', 'val_MAE'])
    
    # initializing the trials
    TrialNumber=0
    for batch in batch_size_list:
        for epochs in epoch_list:
            for l2_reg in regularizers:
                TrialNumber+=1
            
                # model
                ann_model = ann(X_train.shape[1], l2_reg)
            
                # Fitting the ANN to the Training set
                callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)
                ann_history = ann_model.fit(X_train, y_train,
                                            epochs=epochs,
                                            batch_size=batch, 
                                            callbacks=[callback],
                                            verbose=0,
                                            validation_data=(X_val, y_val))
            
                # printing the results of the current iteration
                print(f'{TrialNumber}/{n} -- batch_size: {batch} - epochs: {epochs} - reg: {l2_reg} - loss: {ann_history.history["val_loss"][-1]:.4f} - MAE: {ann_history.history["val_mean_absolute_error"][-1]:.4f}')
            
                SearchResultsData=SearchResultsData.append(pd.DataFrame(data=[[TrialNumber, batch, epochs, l2_reg, ann_history.history['val_loss'][-1], ann_history.history['val_mean_absolute_error'][-1]]],
                                                                    columns=['TrialNumber', 'batch', 'epochs', 'reg', 'val_loss', 'val_MAE']))
    return(SearchResultsData)