def getTicker(ticker):
    import pandas as pd
    import yfinance as yf
    
    df = pd.DataFrame()
    
    # use yfinance to get the ticker history
    df = yf.Ticker(ticker).history(period='max')

    # add symbol column
    df['Symbol'] = ticker

    # Format the date
    df.reset_index(inplace=True)
    df['Date'] = pd.to_datetime(df['Date'], utc=True).dt.date
    df.set_index('Date', inplace=True)

    if not df.empty:
        return df
    else:
        print('Error getting historical data for ' + ticker)


def featureEngineering(df):

    import pandas as pd
    import numpy as np

    
    # Formatting the Date, there's probably a better way to do this, but this works I guess..
    df.reset_index(inplace=True)
    df['Date'] = pd.to_datetime(df['Date'], utc=True)
    df['Date'] = pd.to_datetime(df['Date']).dt.date
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    
    #Switching NaNs to zeros so we dont get infinites
    df.fillna(0, inplace=True)
    

    # Add a column called Change for the main value, which is the percent change from the previous day's close
    df['Close_Change'] = df['Close'].pct_change()
    df['Open_Change'] = df['Open'].pct_change()
    df['High_Change'] = df['High'].pct_change()
    df['Low_Change'] = df['Low'].pct_change()
    df['Volume_Change'] = df['Volume'].pct_change() / 100 # Dividing by 100 as we will not be scaling the data
    

    #drop dividends and splits
    df.drop(['Dividends', 'Stock Splits'], axis=1, inplace=True)

    ## Feature Engineering using TA Library
    #import ta library
    import ta

    #Awesome Oscillator
    from ta.momentum import AwesomeOscillatorIndicator
    df['AO'] = ta.momentum.AwesomeOscillatorIndicator(df['High'], df['Low'], 5, 34).awesome_oscillator() / 10 # Dividing by 10 as we will not be scaling the data

    # Rate of Change
    from ta.momentum import ROCIndicator
    df['ROC'] = ta.momentum.ROCIndicator(df['Close'], 12).roc() / 100 # Dividing by 100 as we will not be scaling the data

    # Relative Strength Index
    from ta.momentum import RSIIndicator
    df['RSI'] = ta.momentum.RSIIndicator(df['Close'], 14).rsi() / 100 # Dividing by 100 as we will not be scaling the data

    # Stochastic RSI
    from ta.momentum import StochRSIIndicator
    df['StochRSI'] = ta.momentum.StochRSIIndicator(df['Close'], 14, 3, 3).stochrsi()

    # Stochastic Oscillator
    from ta.momentum import StochasticOscillator
    df['StochOsc'] = ta.momentum.StochasticOscillator(df['High'], df['Low'], df['Close'], 14, 3, 3).stoch() / 100 # Dividing by 100 as we will not be scaling the data

    # True Strength Index
    from ta.momentum import TSIIndicator
    df['TSI'] = ta.momentum.TSIIndicator(df['Close'], 25, 13).tsi() / 100 # Dividing by 100 as we will not be scaling the data

    # Chaikin Money Flow
    from ta.volume import ChaikinMoneyFlowIndicator
    df['CMF'] = ta.volume.ChaikinMoneyFlowIndicator(df['High'], df['Low'], df['Close'], df['Volume'], 20).chaikin_money_flow()

    # Average True Range
    from ta.volatility import AverageTrueRange
    df['ATR'] = ta.volatility.AverageTrueRange(df['High'], df['Low'], df['Close'], 14).average_true_range() / 10 # Dividing by 10 as we will not be scaling the data

    #Bollinger Bands
    from ta.volatility import BollingerBands
    df['BB_H'] = ta.volatility.BollingerBands(df['Close'], 20, 2).bollinger_hband()
    df['BB_L'] = ta.volatility.BollingerBands(df['Close'], 20, 2).bollinger_lband()
    df['BB_M'] = ta.volatility.BollingerBands(df['Close'], 20, 2).bollinger_mavg()
    df['BB_W'] = ta.volatility.BollingerBands(df['Close'], 20, 2).bollinger_wband()

    # Take the distance between BB_H/L with Close price and calculate its ratio with the width of the Bollinger Bands 
    df['BB_HL'] = (df['BB_H'] - df['Close']) / df['BB_W']
    df['BB_LH'] = (df['Close'] - df['BB_L']) / df['BB_W']

    #Drop bollinger bands
    df.drop(['BB_H', 'BB_L', 'BB_M', 'BB_W'], axis=1, inplace=True)

    # Exponential Moving Average (5, 20, 50, 100, 200)
    from ta.trend import EMAIndicator
    df['EMA_5'] = ta.trend.EMAIndicator(df['Close'], 5).ema_indicator()
    df['EMA_20'] = ta.trend.EMAIndicator(df['Close'], 20).ema_indicator()
    df['EMA_50'] = ta.trend.EMAIndicator(df['Close'], 50).ema_indicator()
    df['EMA_100'] = ta.trend.EMAIndicator(df['Close'], 100).ema_indicator()
    df['EMA_200'] = ta.trend.EMAIndicator(df['Close'], 200).ema_indicator()

    # EMA ratios
    df['Ratio_EMA5'] = df['Close'] / df['EMA_5']
    df['Ratio_EMA20'] = df['EMA_5'] / df['EMA_20']


    # Ratio of the change between Ratio_EMA and its previous value
    df['Ratio_EMA5_Change'] = df['Ratio_EMA5'].pct_change()
    df['Ratio_EMA20_Change'] = df['Ratio_EMA20'].pct_change()

    #drop EMAs
    df.drop(['EMA_5', 'EMA_20', 'EMA_50', 'EMA_100', 'EMA_200'], axis=1, inplace=True)

    # MACD
    from ta.trend import MACD
    df['MACD'] = ta.trend.MACD(df['Close'], 12, 26, 9).macd() / 10 # Dividing by 10 as we will not be scaling the data
    df['MACD_Signal'] = ta.trend.MACD(df['Close'], 12, 26, 9).macd_signal() / 10 # Dividing by 10 as we will not be scaling the data
    df['MACD_Diff'] = ta.trend.MACD(df['Close'], 12, 26, 9).macd_diff() / 10 # Dividing by 10 as we will not be scaling the data

    # Create Median Price
    df['Median_Price'] = (df['High'] + df['Low']) / 2
    
    # Drop Volume
    df.drop(['Volume'], axis=1, inplace=True)

    # Some technical indicators require a minimum of 40 periods of data so we will drop the first 40 rows
    df.drop(df.index[:40], inplace=True)

    # Drop NA -> Maybe 0s instead?
    df.fillna(0, inplace=True)
    
    df.replace([np.inf, -np.inf], 0, inplace=True)
    
    return df

def newLSTM (df, target = 'Close', window = 20, train_split = 0.8):

    # Import libraries
    import pandas as pd
    import numpy as np
    import keras
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Dropout, LSTM
    from sklearn.metrics import mean_squared_error
    from sklearn.model_selection import TimeSeriesSplit

    #Get the ticker symbol
    ticker = df['Symbol'][0].replace('.TO', '')

    # Print what symbol you are training and deleting the column after. We only kept it for the print statement
    print('Symbol: ', ticker)
    
    # Deleting Close, Open, High, Low, Median_Price columns unless they are the target
    if target != 'Close':
        df.drop(['Close'], axis=1, inplace=True)
    
    if target != 'Open':
        df.drop(['Open'], axis=1, inplace=True)
    
    if target != 'High':
        df.drop(['High'], axis=1, inplace=True)
    
    if target != 'Low':
        df.drop(['Low'], axis=1, inplace=True)
    
    if target != 'Median_Price':
        df.drop(['Median_Price'], axis=1, inplace=True)

    # Set X , ensuring 'Target' is the last column (set X without then concat back in)
    X = df.drop([target, 'Symbol'], axis=1)
    y = df[target]
    X = pd.concat([X, y], axis=1)

    #==================================
    # Notice we are not scaling the data. The reason is because we are modifying the data to be a percentage change from the previous day's close when we are formatting the data for the LSTM below. The remainder of the data has been manually scaled during Feature Engineering.
    #==================================

    def lstm_split(data, n_steps):
        X, y, temp = [], [], []
        for i in range(len(data) - n_steps):
            
            # In order to increase the accuracy of our model, we will be normalizing the data of every window by the last value of the window. This will result in the last value of the window being 1.0 and the predictions will be a percentage change from the last value of the window. This will also help the model to learn the trend of the stock better.
            
            # Get the last value of the window
            CentralValue = data[i + n_steps -1 ,-1] 
            
            # Get the values of the predicted value in window and divide by the last value of the window
            temp = data[i:i + n_steps, -1]
            temp = temp / CentralValue

            #add data[i:i + n_steps, :-1] & temp to X
            X.append(np.concatenate((data[i:i + n_steps, :-1], temp.reshape(-1,1)), axis=1))

            y.append(data[i + n_steps, -1] / CentralValue) # Putting the next row's target value into y
            
        return np.array(X), np.array(y)

    # Call the function to format the data for the LSTM
    X1, y1 = lstm_split(X.values, n_steps = window)
    
    # Split the data into train and test
    split_idx = int(np.ceil(len(X1)*train_split))
    date_index = df.index

    X_train, X_test = X1[:split_idx], X1[split_idx:]
    y_train, y_test = y1[:split_idx], y1[split_idx:]
    X_train_date, X_test_date = date_index[:split_idx], date_index[split_idx:]

    # Create X_Predict
    X_Predict = []
    X_Predict.append(X.values[len(X) - window:, ])
    X_Predict = np.array(X_Predict)

    #Build the model
    lstm = Sequential()
    lstm.add(LSTM(50, activation='relu', return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
    # lstm.add(LSTM(50, activation='relu', return_sequences=True))       
    # lstm.add(LSTM(120, activation='relu', return_sequences=True))
    lstm.add(LSTM(50, activation='relu', return_sequences=False))
    lstm.add(Dropout(rate=0.2))
    lstm.add(Dense(1))
    lstm.compile(optimizer='adam', loss='mse')

    # Train the model
    lstm.fit(X_train, y_train, epochs=25, batch_size=5, verbose=1, shuffle=False)

    # Save trained model
    lstm.save('../models/LSTM_' + ticker + '_' + target + '.h5')      

    # Test model
    y_pred = lstm.predict(X_test)

    #Predict future values
    y_pred_future = lstm.predict(X_Predict)

    # Calculate RMSE
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    return y_test, y_pred, y_pred_future, rmse

def addPredictions(df, true, test, rmse, target):
    # Add the predictions to the dataframe
    df[target + '_Predictions'] = np.insert(test, 0, np.zeros(len(df) - len(test)))
    df[target + '_True'] = np.insert(true, 0, np.zeros(len(df) - len(true)))

    # Add the RMSE to the dataframe
    df[target + '_Predictions_High'] = df[target + '_Predictions'] + rmse
    df[target + '_Predictions_Low'] = df[target + '_Predictions'] - rmse

    return df

def addFuturePredictions(Predictions, pred, rmse, target, symbol):
    import pandas as pd
    
    temp = pd.DataFrame(pred, columns=['Prediction'])
    temp['RMSE'] = rmse
    temp['Symbol'] = symbol
    temp['Target'] = target
    
    Predictions = pd.concat([Predictions, temp])

    return Predictions

def getPredictions(df):
    import pandas as pd

    Predictions = pd.DataFrame()
    symbol = df['Symbol'][0]

    # Predict Close
    true, test, pred, rmse = newLSTM(df, target = 'Close', window = 20, train_split = 0.8)
    df = addPredictions(df, true, test, pred, rmse, 'Close')
    Predictions = addFuturePredictions(Predictions, pred, rmse, 'Close', symbol)
    
    # Predict High
    true, test, pred, rmse = newLSTM(df, target = 'High', window = 20, train_split = 0.8)
    df = addPredictions(df, true, test, pred, rmse, 'High')
    Predictions = addFuturePredictions(Predictions, pred, rmse, 'High', symbol)

    # Predict Low
    true, test, pred, rmse = newLSTM(df, target = 'Low', window = 20, train_split = 0.8)
    df = addPredictions(df, true, test, pred, rmse, 'Low')
    Predictions = addFuturePredictions(Predictions, pred, rmse, 'Low', symbol)
    
    # Predict Open
    true, test, pred, rmse = newLSTM(df, target = 'Open', window = 20, train_split = 0.8)
    df = addPredictions(df, true, test, pred, rmse, 'Open')
    Predictions = addFuturePredictions(Predictions, pred, rmse, 'Open', symbol)

    # Predict Median_Price
    true, test, pred, rmse = newLSTM(df, target = 'Median_Price', window = 20, train_split = 0.8)
    df = addPredictions(df, true, test, pred, rmse, 'Median_Price')
    Predictions = addFuturePredictions(Predictions, pred, rmse, 'Median_Price', symbol)

    return df, Predictions
