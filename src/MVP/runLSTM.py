def runLSTM_1D_1S (df, target = 'Close', window = 10, model='New', File = 'undefined', train_split = 0.8):

    #Import libraries

    import pandas as pd
    import numpy as np
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Dropout, LSTM
    from tensorflow.keras.callbacks import EarlyStopping
    from sklearn.preprocessing import MinMaxScaler, StandardScaler
    from sklearn.metrics import mean_squared_error
    from sklearn.model_selection import TimeSeriesSplit

    #drop dividend, index, symbol
    df = df.drop(['Dividends', 'Stock Splits', 'symbol'], axis=1)

    # Set X , ensuring 'Target' is the last column (set X without then concat back in)
    X = df.drop(target, axis=1)
    y = df[target]
    X = pd.concat([X, y], axis=1)

    # Scale the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X = pd.DataFrame(columns=X.columns
                    , data=X_scaled
                    , index=X.index)

    # Store the shape of the data into scaler_shape
    scaler_shape = X.shape

    def lstm_split(data, n_steps):
        X, y = [], []
        for i in range(len(data) - n_steps):
            X.append(data[i:i + n_steps,]) # Putting all features from all rows specified by n_steps into X (including target)
            y.append(data[i + n_steps, -1]) # Putting the next row's target value into y

        return np.array(X), np.array(y)

    X1, y1 = lstm_split(X.values, n_steps = window)

    split_idx = int(np.ceil(len(X1)*train_split))
    date_index = df.index

    X_train, X_test = X1[:split_idx], X1[split_idx:]
    y_train, y_test = y1[:split_idx], y1[split_idx:]
    X_train_date, X_test_date = date_index[:split_idx], date_index[split_idx:]

    # Create X_Predict
    X_Predict = []
    X_Predict.append(X.values[len(X) - window + 1:, :-1])
    X_Predict = np.array(X_Predict)


    if model == 'New':
        #Build the model
        lstm = Sequential()
        lstm.add(LSTM(120, activation='relu', return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
                
        lstm.add(LSTM(60, activation='relu', return_sequences=True))
                
        lstm.add(LSTM(120, activation='relu', return_sequences=True))
                
        lstm.add(LSTM(60, activation='relu', return_sequences=False))
        lstm.add(Dropout(rate=0.2))

        lstm.add(Dense(1))
        lstm.compile(optimizer='adam', loss='mse')
        # lstm.summary()

        # Train the model
        lstm.fit(X_train, y_train, epochs=25, batch_size=5, verbose=1, shuffle=False)

        # Save trained model (pickle) and add date to file name
        import pickle
        import datetime
        
        now = datetime.datetime.now()
        now = now.strftime("%Y-%m-%d %H:%M")
        filename = 'LSTM_' + File + '.sav'
        pickle.dump(lstm, open('../models/'+filename))
    
    else:
        # load model (pickle)
        import pickle
        lstm = pickle.load(open(model, 'rb'))

    # Test model
    y_pred = lstm.predict(X_test)

    #Predict future values
    y_pred_future = lstm.predict(X_Predict)

    # unscaling the data
    #reshape y_test and y_pred
    y_test_reshaped = y_test.reshape(-1,1)
    y_pred_reshaped = y_pred.reshape(-1,1)
    y_pred_future_reshaped = y_pred_future.reshape(-1,1)

    #add missing dimensions y_test and y_pred match scaler_shape
    y_test_dimensions = np.repeat(y_test_reshaped, scaler_shape[1], axis=1)
    y_pred_dimensions = np.repeat(y_pred_reshaped, scaler_shape[1], axis=1)
    y_pred_future_dimensions = np.repeat(y_pred_future_reshaped, scaler_shape[1], axis=1)

    y_test_unscaled = scaler.inverse_transform(y_test_dimensions)
    y_pred_unscaled = scaler.inverse_transform(y_pred_dimensions)
    y_pred_future_unscaled = scaler.inverse_transform(y_pred_future_dimensions)

    # set y_test_unscaled and y_pred_unscaled to 1D
    y_test_unscaled = y_test_unscaled[:,0]
    y_pred_unscaled = y_pred_unscaled[:,0] 
    y_pred_future_unscaled = y_pred_future_unscaled[:,0]

    rmse = np.sqrt(mean_squared_error(y_test_unscaled, y_pred_unscaled))

    return y_pred_unscaled, y_pred_future_unscaled, rmse
    