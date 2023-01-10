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

    # Drop rows where Open and Close are both 0 (halted trading)
    df2 = df[(df['Open'] != 0) & (df['Close'] != 0)]    
    df = df2.copy() # Doing it this way so we don't get a warning saying we're setting a value on a copy of a slice from a dataframe

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
    # print('Symbol: ', ticker) # Kept for archive purposes, but now I'm printing the ticker in the main function
    
    # Set X , ensuring 'Target' is the last column (set X without then concat back in)
    #removing Close, Open, High, Low, Median_Price columns, unless they are the target
    if target == 'Close':
        X = df.drop([target, 'Symbol', 'Open', 'High', 'Low', 'Median_Price'], axis=1)
        
    if target == 'High':
        X = df.drop([target, 'Symbol', 'Close', 'Open', 'Low', 'Median_Price'], axis=1)

    if target == 'Low':
        X = df.drop([target, 'Symbol', 'Close', 'Open', 'High', 'Median_Price'], axis=1)

    if target == 'Open':
        X = df.drop([target, 'Symbol', 'Close', 'High', 'Low', 'Median_Price'], axis=1)

    if target == 'Median_Price':
        X = df.drop([target, 'Symbol', 'Close', 'Open', 'High', 'Low'], axis=1)    
    
    
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

            # Encountered a CentralValue of 0, but noticed later that the stock was halted. I prefer to remove these days during Feature Engineering, but I will leave this here in case I run into issues in the future
            # if CentralValue == 0:
            #     CentralValue = 1
            #     print('CentralValue = 0', i)
            
            # Get the values of the predicted value in window and divide by the last value of the window
            temp = data[i:i + n_steps, -1]
            temp = temp / CentralValue

            #add data[i:i + n_steps, :-1] & temp to X
            X.append(np.concatenate((data[i:i + n_steps, :-1], temp.reshape(-1,1)), axis=1))

            y.append(data[i + n_steps, -1] / CentralValue) # Putting the next row's target value into y

        # #replace NANs with 0
        # X = np.nan_to_num(X)
        # y = np.nan_to_num(y)
        
        # #replace inf with 0
        # from numpy import inf
        # X[X==inf]=0
        # X[X==-inf]=0

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

    #set the last value of X as the central value
    CentralValue = X.values[-1, -1]

    # Encountered a CentralValue of 0, but noticed later that the stock was halted. I prefer to remove these days during Feature Engineering, but I will leave this here in case I run into issues in the future
    # if CentralValue == 0:
    #     CentralValue = 1
    #     print('CentralValue = 0', "X_Predict", )

    temp = X.values[len(X) - window:, -1]
    temp = temp / CentralValue
    
    # Encountered a CentralValue of 0, but noticed later that the stock was halted. I prefer to remove these days during Feature Engineering, but I will leave this here in case I run into issues in the future
    # #replace NANs and inf with 0
    # temp = np.nan_to_num(temp)
    # from numpy import inf
    # temp[temp==inf]=0
    # temp[temp==-inf]=0

    X_Predict.append(np.concatenate((X.values[len(X) - window:, :-1], temp.reshape(-1,1)), axis=1))
    X_Predict = np.array(X_Predict)

    # Check if we already have a trained model (check path)
    try:

        from keras.models import load_model

        lstm = load_model('../models/LSTM_' + ticker + '_' + target + '.h5')
        print('Model Loaded')
    
    except:
    
        print('Model Not Found, training new model')

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

def addPredictions(lenDF, true, test, rmse, target):
    import numpy as np
    import pandas as pd

    df = pd.DataFrame()

    # Add the predictions to the dataframe
    df[target + '_Predictions'] = np.insert(test, 0, np.zeros(lenDF - len(test)))
    df[target + '_True'] = np.insert(true, 0, np.zeros(lenDF - len(true)))

    # Set all values of column RMSE to rmse
    # df['RMSE'] = rmse

    # df[target + '_Predictions_High'] = df[target + '_Predictions'] + df['RMSE']
    # df[target + '_Predictions_Low'] = df[target + '_Predictions'] - df['RMSE']

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

    #print Symbol
    print("Symbol: ", symbol)

    # Predict Close
    print('Predicting Close...')
    true, test, pred, rmse = newLSTM(df, target = 'Close', window = 20, train_split = 0.8)
    Close = addPredictions(len(df), true, test, rmse, 'Close')
    Predictions = addFuturePredictions(Predictions, pred, rmse, 'Close', symbol)
    
    # Predict High
    print('Predicting High...')
    true, test, pred, rmse = newLSTM(df, target = 'High', window = 20, train_split = 0.8)
    High = addPredictions(len(df), true, test, rmse, 'High')
    Predictions = addFuturePredictions(Predictions, pred, rmse, 'High', symbol)

    # Predict Low
    print('Predicting Low...')
    true, test, pred, rmse = newLSTM(df, target = 'Low', window = 20, train_split = 0.8)
    Low = addPredictions(len(df), true, test, pred, 'Low')
    Predictions = addFuturePredictions(Predictions, pred, rmse, 'Low', symbol)
    
    # Predict Open
    print('Predicting Open...')
    true, test, pred, rmse = newLSTM(df, target = 'Open', window = 20, train_split = 0.8)
    Open = addPredictions(len(df), true, test, pred, 'Open')
    Predictions = addFuturePredictions(Predictions, pred, rmse, 'Open', symbol)

    # Predict Median_Price
    print('Predicting Median Price...')
    true, test, pred, rmse = newLSTM(df, target = 'Median_Price', window = 20, train_split = 0.8)
    Median = addPredictions(len(df), true, test, pred, 'Median_Price')
    Predictions = addFuturePredictions(Predictions, pred, rmse, 'Median_Price', symbol)

    # Add Close, High, Low, Open, and Median_Price to df
    temp = pd.concat([Close, High, Low, Open, Median], axis=1)

    # temp and df should have different lengths, so we need to add the extra rows to temp
    if len(df) > len(temp):
        temp = pd.concat([temp, pd.DataFrame(np.zeros((len(df) - len(temp), 10)), columns=temp.columns)], axis=0)

    # Add temp as additional columns to df
    df = df.reset_index(drop=False) #we don't have the dates as an index for temp, so we need to reset the index to merge
    df = pd.concat([df, temp], axis=1)
    df = df.set_index('Date')

    return df, Predictions

def formatPredictions(Ticker_pred):    
    # Putting the predictions back to the original price value
    Ticker_pred['Open_Predictions'] = Ticker_pred['Open_Predictions'] * Ticker_pred['Open'].shift(1)
    Ticker_pred['Open_True'] = Ticker_pred['Open_True'] * Ticker_pred['Open'].shift(1)

    Ticker_pred['Close_Predictions'] = Ticker_pred['Close_Predictions'] * Ticker_pred['Close'].shift(1)
    Ticker_pred['Close_True'] = Ticker_pred['Close_True'] * Ticker_pred['Close'].shift(1)

    Ticker_pred['High_Predictions'] = Ticker_pred['High_Predictions'] * Ticker_pred['High'].shift(1)
    Ticker_pred['High_True'] = Ticker_pred['High_True'] * Ticker_pred['High'].shift(1)

    Ticker_pred['Low_Predictions'] = Ticker_pred['Low_Predictions'] * Ticker_pred['Low'].shift(1)
    Ticker_pred['Low_True'] = Ticker_pred['Low_True'] * Ticker_pred['Low'].shift(1)

    Ticker_pred['Median_Price_Predictions'] = Ticker_pred['Median_Price_Predictions'] * Ticker_pred['Median_Price'].shift(1)
    Ticker_pred['Median_Price_True'] = Ticker_pred['Median_Price_True'] * Ticker_pred['Median_Price'].shift(1)

    return Ticker_pred


def plotlyGraph(target, Ticker_pred, Ticker_Future, ticker):
    import plotly.graph_objects as go
    
    print(target + ':')

    #Get RMSE value, which is the value of RMSE column in Ticker_Future where Target = target and symbol = ticker
    RMSE = Ticker_Future.loc[(Ticker_Future['Target'] == target) & (Ticker_Future['Symbol'] == ticker), 'RMSE'].values[0]

    fig = go.Figure()   
    fig.add_trace(go.Scatter(x=Ticker_pred.index, y=Ticker_pred[target + '_True'], mode='lines', name='True'))
    # fig.add_trace(go.Scatter(x=Ticker_pred.index, y=Ticker_pred[target + '_Predictions'], mode='lines', name='Predictions'))

    #make a shaded area for the prediction (width = rmse on each side)
    fig.add_trace(go.Scatter(x=Ticker_pred.index, y=Ticker_pred[target + '_Predictions'] * (1 + RMSE), fill=None, mode='lines', line_color='rgba(0,100,80,0.2)', name='Prediction_High'))
    fig.add_trace(go.Scatter(x=Ticker_pred.index, y=Ticker_pred[target + '_Predictions'] * (1 - RMSE), fill='tonexty', mode='lines', line_color='rgba(0,100,80,0.2)', name='Prediction_Band'))
    fig.show()
    print()