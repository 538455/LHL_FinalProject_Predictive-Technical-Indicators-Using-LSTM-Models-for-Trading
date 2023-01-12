def dailyPrediction(ticker, accuracyPlot = True):
    
    # Get the stock historical data
    import pandas as pd
    # from Functions import getTicker_daily
    Ticker = getTicker_daily(ticker)

    # Save a copy of Ticker (backup if library is not working)
    Ticker.reset_index(inplace=True)
    Ticker.to_csv('../../data/1D/' + ticker + '.csv', index=False)

    Ticker = pd.read_csv('../../data/1D/' + ticker + '.csv', index_col=0)

    # Feature Engineering
    # from Functions import featureEngineering
    Ticker_FE = featureEngineering(Ticker)

    # Run the model to get the predictions
    # from Functions import getPredictions
    Ticker_pred, Ticker_Future = getPredictions(Ticker_FE, Predict='1D')

    # Reformat the Prediction Prices
    # from Functions import formatPredictions
    Ticker_pred = formatPredictions(Ticker_pred)

    if accuracyPlot == True:
        # Plot the predictions
        # from Functions import plotlyGraph
        plotlyGraph('Open', Ticker_pred, Ticker_Future, ticker)
        plotlyGraph('Close', Ticker_pred, Ticker_Future, ticker)
        plotlyGraph('High', Ticker_pred, Ticker_Future, ticker)
        plotlyGraph('Low', Ticker_pred, Ticker_Future, ticker)
        plotlyGraph('Median_Price', Ticker_pred, Ticker_Future, ticker)

    # Format the predictions
    Ticker_D_Pred = format_D_Pred(Ticker_pred, Ticker_Future)

    # Add the investment columns
    Ticker_D_Pred = addInvestmentcols(Ticker_D_Pred)

    #count the number of unique rows where bought == 1
    bought = len(Ticker_D_Pred.loc[Ticker_D_Pred['bought'] == 1])
    sold = len(Ticker_D_Pred.loc[Ticker_D_Pred['sold'] == 1])

    # Print the accuracy and keep only one decimal
    print()

    TestLen = len(Ticker_D_Pred[Ticker_D_Pred['Predicted_Profit'].notna()]) / 365
    print("General Performance - Last ~", round(TestLen, 1) ,"year(s):")

    # Ticker_D_Pred['profit'] = Ticker_D_Pred['profit'] / 100 # Divide profit by 100 for Profit Readability
    print("Profit:", round(Ticker_D_Pred['profit'].sum(), 2), "%               Yearly avg.: ", round(Ticker_D_Pred['profit'].sum() / TestLen, 2), "%")
    # Ticker_D_Pred['profit'] = Ticker_D_Pred['profit'] * 100 # Multiply profit by 100 to get it back to original value
    
    print("# Transactions:", len(Ticker_D_Pred.loc[Ticker_D_Pred['bought'] == 1]), "          Yearl avg: ", round(len(Ticker_D_Pred.loc[Ticker_D_Pred['bought'] == 1]) / TestLen, 2))

    print("Success rate:", round(sold / bought * 100, 2), "%")
    
    # Print the sum of profit
    print('Daily Average when successful: ', round(Ticker_D_Pred[(Ticker_D_Pred['bought'] == 1) & (Ticker_D_Pred['sold'] == 1)]['profit'].mean(), 2) , "%")
    print('Daily Average when unsuccessful: ', round(Ticker_D_Pred[(Ticker_D_Pred['bought'] == 1) & (Ticker_D_Pred['sold'] == 0)]['profit'].mean(), 2) , "%")

    print()
    # Print the number of rows where Ticker_D_Pred['bought'] == 1 & sold == 0 & profit > 1
    # missed = len(Ticker_D_Pred[(Ticker_D_Pred['bought'] == 1) & (Ticker_D_Pred['sold'] == 0)])
    # loss = len(Ticker_D_Pred[(Ticker_D_Pred['bought'] == 1) & (Ticker_D_Pred['sold'] == 0) & (Ticker_D_Pred['profit'] < 1)])
    # gain = len(Ticker_D_Pred[(Ticker_D_Pred['bought'] == 1) & (Ticker_D_Pred['sold'] == 0) & (Ticker_D_Pred['profit'] > 1)])

    # print('When Missed:')
    # print('Take a loss: ', round(loss / missed * 100), "% of the time...     ", 'Average loss: ', round((Ticker_D_Pred[(Ticker_D_Pred['bought'] == 1) & (Ticker_D_Pred['sold'] == 0) & (Ticker_D_Pred['profit'] < 1)]['profit'].mean())), "%")
    # print('Make a gain: ', round(gain / missed * 100), "% of the time...     ", 'Average gain: ', round((Ticker_D_Pred[(Ticker_D_Pred['bought'] == 1) & (Ticker_D_Pred['sold'] == 0) & (Ticker_D_Pred['profit'] >= 1)]['profit'].mean())), "%")

    # print()
    # print("Total historical profit:", round(Ticker_D_Pred['profit'].sum()), "%")

    #Make a copy of Ticker_D_Pred
    Ticker_D_Pred_copy = Ticker_D_Pred.copy()
    # reset the index
    Ticker_D_Pred_copy.reset_index(inplace=True, drop=False)
    # rename the index column to Date and format it to datetime
    Ticker_D_Pred_copy.rename(columns={'index': 'Date'}, inplace=True)
    Ticker_D_Pred_copy['Date'] = pd.to_datetime(Ticker_D_Pred_copy['Date'])
    #Filter Date to only include year 2022
    Ticker_D_Pred_copy = Ticker_D_Pred_copy[Ticker_D_Pred_copy['Date'].dt.year == 2022]
    
    print("2022 Performance:")
    
    # Ticker_D_Pred_copy['profit'] = Ticker_D_Pred_copy['profit'] / 100 # Divide profit by 100 for Profit Readability
    print("Profit:", round(Ticker_D_Pred_copy['profit'].sum(), 2), "%")
    # Ticker_D_Pred_copy['profit'] = Ticker_D_Pred_copy['profit'] * 100 # Multiply profit by 100 to get it back to original value

    print("# Transactions:", len(Ticker_D_Pred_copy.loc[Ticker_D_Pred_copy['bought'] == 1]))

    sold = len(Ticker_D_Pred_copy.loc[Ticker_D_Pred_copy['sold'] == 1])
    bought = len(Ticker_D_Pred_copy.loc[Ticker_D_Pred_copy['bought'] == 1])
    print("Success rate:", round(sold / bought * 100, 2), "%")

    # Print the sum of profit
    print('Daily Average when successful: ', round(Ticker_D_Pred_copy[(Ticker_D_Pred_copy['bought'] == 1) & (Ticker_D_Pred_copy['sold'] == 1)]['profit'].mean(), 2) , "%")
    print('Daily Average when unsuccessful: ', round(Ticker_D_Pred_copy[(Ticker_D_Pred_copy['bought'] == 1) & (Ticker_D_Pred_copy['sold'] == 0)]['profit'].mean(), 2) , "%")


    return Ticker_D_Pred

def addInvestmentcols(Ticker_D_Pred):
    import pandas as pd
    import numpy as np

    # RMSE = High_RMSE's last value
    RMSE = 1 - (Ticker_D_Pred['High_RMSE'].iloc[-1] * 0) # In the end, it rem
    MinPredProfit = .25

    # LimitBuy = Low_Predictions
    # LimitSell = High_Predictions - RMSE
    # StopLoss = Low_Predictions * 0.97

    # Create a new column in Ticker_D_Pred called Predicted_Profit and set it to 1 if Low_Predictions > (High_Predictions * 1.03) and 0 otherwise
    # Ticker_D_Pred['Predicted_Profit'] = (1 - (Ticker_D_Pred['High_Predictions'] * RMSE) / Ticker_D_Pred['Open_Predictions'] ) * 100 #backup.. I think that's a mistake
    Ticker_D_Pred['Predicted_Profit'] = (Ticker_D_Pred['High_Predictions'] * RMSE) / Ticker_D_Pred['Open_Predictions'] * 100 - 100

    # Create a new column in Ticker_D_P
    # Ticker_D_Pred['True_Profit'] = (1 - (Ticker_D_Pred['High_True'] / Ticker_D_Pred['Open_True']) ) * 100 #backup.. I think that's a mistake
    Ticker_D_Pred['True_Profit'] = (Ticker_D_Pred['High_True'] / Ticker_D_Pred['Open_True']) * 100 - 100

    # Create a new column in Ticker_D_Pred called bought and set it to 1 if Predicted_Profit > 1, Low_Predictions > Low_True and 0 otherwise
    Ticker_D_Pred['bought'] = np.where((Ticker_D_Pred['Predicted_Profit'] > MinPredProfit) & (Ticker_D_Pred['Open_Predictions'] > Ticker_D_Pred['Open_True']), 1, 0)

    #=====================================================================================================================================================================
    # I found some volatile situations where the models usually don't profit well, so I'm going to remove them.
    # Ticker_D_Pred['bought'] = np.where((Ticker_D_Pred['bought'] == 1) & (Ticker_D_Pred['Open_Predictions'] < Ticker_D_Pred['Close_True'].shift(1)), 0, Ticker_D_Pred['bought']) # Indicative of a downward trend and model is likely predicting a deadcat bounce.
    Ticker_D_Pred['bought'] = np.where((Ticker_D_Pred['bought'] == 1) & (Ticker_D_Pred['Open_Predictions'] < Ticker_D_Pred['Low_True'].shift(1)), 0, Ticker_D_Pred['bought']) # Indicative of a downward trend and model is likely predicting a deadcat bounce. 
    # Ticker_D_Pred['bought'] = np.where((Ticker_D_Pred['bought'] == 1) & (Ticker_D_Pred['Median_Price_Predictions'] < Ticker_D_Pred['Open_Predictions']), 0, Ticker_D_Pred['bought']) # Model is predicting that there is more downside than upside. If I miss the High, there's a high chance to lose significantly.
    # Ticker_D_Pred['bought'] = np.where((Ticker_D_Pred['bought'] == 1) & (Ticker_D_Pred['Close_Predictions'] > Ticker_D_Pred['High_Predictions']), 0, Ticker_D_Pred['bought']) # This is an impossible scenario. If the model predicts that the close will be higher than the high, it's likely a very volatile situation.
    # Ticker_D_Pred['bought'] = np.where((Ticker_D_Pred['bought'] == 1) & (abs(Ticker_D_Pred['Median_Price_Predictions'] - Ticker_D_Pred['Low_Predictions']) < abs(Ticker_D_Pred['Median_Price_Predictions'] - Ticker_D_Pred['High_Predictions'])), 0, Ticker_D_Pred['bought']) # Could be considered a bearsish signal as average profit is lower in those situations.
    #=====================================================================================================================================================================

    # Create PrevDayAccuracyClose
    SBB['Prev_Day_Accuracy_Close'] = SBB['Close_True'] / SBB['Close_Predictions'] *100 - 100

    # Create a new column in Ticker_D_Pred called sold and set it to 1 if bought == 1 and High_Predictions < High_True and 0 otherwise
    Ticker_D_Pred['sold'] = np.where((Ticker_D_Pred['bought'] == 1) & ((Ticker_D_Pred['High_Predictions'] * RMSE) < Ticker_D_Pred['High_True']), 1, 0)

    # Create a new column in Ticker_D_Pred called profit and: 
    # If bought == 1 and sold == 1, set it to High_Predictions - Low_Predictions
    # If bought == 1 and sold == 0, set it to Close_True - Low_Predictions
    # If bought == 0 and sold == 0, set it to 0
    # Ticker_D_Pred['profit'] = np.where((Ticker_D_Pred['bought'] == 1) & (Ticker_D_Pred['sold'] == 1), (Ticker_D_Pred['High_Predictions'] * RMSE) / Ticker_D_Pred['Open_Predictions'], np.where((Ticker_D_Pred['bought'] == 1) & (Ticker_D_Pred['sold'] == 0), Ticker_D_Pred['Close_True'] / Ticker_D_Pred['Open_Predictions'], 0))
    Ticker_D_Pred['profit'] = np.where((Ticker_D_Pred['bought'] == 1) & (Ticker_D_Pred['sold'] == 1), (Ticker_D_Pred['High_Predictions'] * RMSE) / Ticker_D_Pred['Open_True'], np.where((Ticker_D_Pred['bought'] == 1) & (Ticker_D_Pred['sold'] == 0), Ticker_D_Pred['Close_True'] / Ticker_D_Pred['Open_True'], 0))
    
    # Where profit is not 0, subtract * 100 and subtract 100
    Ticker_D_Pred['profit'] = round(np.where(Ticker_D_Pred['profit'] != 0, Ticker_D_Pred['profit'] * 100 - 100, 0),2)

    # Add stoploss. In profit, if value is between 0.01 and 0.97, set it to 0.97
    # Ticker_D_Pred['profit'] = np.where((Ticker_D_Pred['profit'] > 0.01) & (Ticker_D_Pred['profit'] < 0.97), 0.97, Ticker_D_Pred['profit'])

    return Ticker_D_Pred


def format_D_Pred(Ticker_D_Pred, Ticker_D_Future):
    import pandas as pd
    import numpy as np


    #Get Open, Close, High, Low, Volume, and Median Price of the last day in Ticker_D_Pred and Ticker_W_Pred
    DOpen = Ticker_D_Pred['Open'].iloc[-1]
    DClose = Ticker_D_Pred['Close'].iloc[-1]
    DHigh = Ticker_D_Pred['High'].iloc[-1]
    DLow = Ticker_D_Pred['Low'].iloc[-1]
    DMedianPrice = Ticker_D_Pred['Median_Price'].iloc[-1]

    # Add an empty row to the end of Ticker_D_Pred
    Ticker_D_Pred.loc[len(Ticker_D_Pred)] = np.nan

    # set the index of the new row to the next weekday from today
    next_weekday = pd.to_datetime('today') + pd.offsets.BDay(1)
    Ticker_D_Pred.reset_index(inplace=True, drop=False)
    Ticker_D_Pred.loc[Ticker_D_Pred.index[-1], 'Date'] = next_weekday
    Ticker_D_Pred['Date'] = pd.to_datetime(Ticker_D_Pred['Date']).dt.date
    Ticker_D_Pred.set_index(['Date'], inplace=True)

    # Format the future predictions
    Ticker_D_Future.loc[Ticker_D_Future['Target'] == 'Open', 'Prediction'] = Ticker_D_Future.loc[Ticker_D_Future['Target'] == 'Open', 'Prediction'] * DOpen
    Ticker_D_Future.loc[Ticker_D_Future['Target'] == 'Close', 'Prediction'] = Ticker_D_Future.loc[Ticker_D_Future['Target'] == 'Close', 'Prediction'] * DClose
    Ticker_D_Future.loc[Ticker_D_Future['Target'] == 'High', 'Prediction'] = Ticker_D_Future.loc[Ticker_D_Future['Target'] == 'High', 'Prediction'] * DHigh
    Ticker_D_Future.loc[Ticker_D_Future['Target'] == 'Low', 'Prediction'] = Ticker_D_Future.loc[Ticker_D_Future['Target'] == 'Low', 'Prediction'] * DLow
    Ticker_D_Future.loc[Ticker_D_Future['Target'] == 'Median_Price', 'Prediction'] = Ticker_D_Future.loc[Ticker_D_Future['Target'] == 'Median_Price', 'Prediction'] * DMedianPrice

    # Add the future predictions to the last row of Ticker_D_Pred
    Ticker_D_Pred.loc[Ticker_D_Pred.index[-1], 'Open_Predictions'] = Ticker_D_Future.loc[Ticker_D_Future['Target'] == 'Open', 'Prediction'].values
    Ticker_D_Pred.loc[Ticker_D_Pred.index[-1], 'Close_Predictions'] = Ticker_D_Future.loc[Ticker_D_Future['Target'] == 'Close', 'Prediction'].values
    Ticker_D_Pred.loc[Ticker_D_Pred.index[-1], 'High_Predictions'] = Ticker_D_Future.loc[Ticker_D_Future['Target'] == 'High', 'Prediction'].values
    Ticker_D_Pred.loc[Ticker_D_Pred.index[-1], 'Low_Predictions'] = Ticker_D_Future.loc[Ticker_D_Future['Target'] == 'Low', 'Prediction'].values
    Ticker_D_Pred.loc[Ticker_D_Pred.index[-1], 'Median_Price_Predictions'] = Ticker_D_Future.loc[Ticker_D_Future['Target'] == 'Median_Price', 'Prediction'].values

    #Create Open_RMSE, Close_RMSE, High_RMSE, Low_RMSE, Median_Price_RMSE columns
    Ticker_D_Pred['Open_RMSE'] = np.nan
    Ticker_D_Pred['Close_RMSE'] = np.nan
    Ticker_D_Pred['High_RMSE'] = np.nan
    Ticker_D_Pred['Low_RMSE'] = np.nan
    Ticker_D_Pred['Median_Price_RMSE'] = np.nan

    # Add the RMSE values to the last row of Ticker_D_Pred
    Ticker_D_Pred.loc[Ticker_D_Pred.index[-1], 'Open_RMSE'] = Ticker_D_Future.loc[Ticker_D_Future['Target'] == 'Open', 'RMSE'].values
    Ticker_D_Pred.loc[Ticker_D_Pred.index[-1], 'Close_RMSE'] = Ticker_D_Future.loc[Ticker_D_Future['Target'] == 'Close', 'RMSE'].values
    Ticker_D_Pred.loc[Ticker_D_Pred.index[-1], 'High_RMSE'] = Ticker_D_Future.loc[Ticker_D_Future['Target'] == 'High', 'RMSE'].values
    Ticker_D_Pred.loc[Ticker_D_Pred.index[-1], 'Low_RMSE'] = Ticker_D_Future.loc[Ticker_D_Future['Target'] == 'Low', 'RMSE'].values
    Ticker_D_Pred.loc[Ticker_D_Pred.index[-1], 'Median_Price_RMSE'] = Ticker_D_Future.loc[Ticker_D_Future['Target'] == 'Median_Price', 'RMSE'].values

    # Delete Close_Change, High_Change, Low_Change, Open_Change, Volume_Change, AO,	ROC,RSI,StochRSI,StochOsc,TSI,CMF,ATR,BB_HL,BB_LH,Ratio_EMA5,Ratio_EMA20,Ratio_EMA5_Change,Ratio_EMA20_Change,MACD,MACD_Signal,MACD_Diff,Median_Price
    Ticker_D_Pred.drop(['Close_Change', 'High_Change', 'Low_Change', 'Open_Change', 'Volume_Change', 'AO', 'ROC', 'RSI', 'StochRSI', 'StochOsc', 'TSI', 'CMF', 'ATR', 'BB_HL', 'BB_LH', 'Ratio_EMA5', 'Ratio_EMA20', 'Ratio_EMA5_Change', 'Ratio_EMA20_Change', 'MACD', 'MACD_Signal', 'MACD_Diff', 'Median_Price'], axis=1, inplace=True)

    return Ticker_D_Pred


def getTicker_daily(ticker):
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
    
    #Delete rows with NaNs
    df.dropna(inplace=True)
    
    # #Switching NaNs to zeros so we dont get infinites
    # df.fillna(0, inplace=True)
    

    # Add a column called Change for the main value, which is the percent change from the previous day's close
    df['Close_Change'] = df['Close'].pct_change()
    df['Open_Change'] = df['Open'].pct_change()
    df['High_Change'] = df['High'].pct_change()
    df['Low_Change'] = df['Low'].pct_change()
    df['Volume_Change'] = df['Volume'].pct_change() / 100 # Dividing by 100 as we will not be scaling the data
    
    #drop dividends and splits
    df.drop(['Dividends', 'Stock Splits'], axis=1, inplace=True)

    # Drop rows where Open, Close, High,  or Low == 0
    df2 = df[(df[['Open', 'Close', 'High', 'Low']] != 0).all(axis=1)]
    df = df2.copy() # Copying the dataframe to avoid the SettingWithCopyWarning

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

def runLSTM (df, target = 'Close', window = 20, train_split = 0.8, predict = '1D'):

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

        lstm = load_model('../models/' + predict + '/LSTM_' + ticker + '_' + target + '.h5')
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
        lstm.fit(X_train, y_train, epochs=100, batch_size=5, verbose=1, shuffle=False)

        # Save trained model
        lstm.save('../models/' + predict + '/LSTM_' + ticker + '_' + target + '.h5')      

    # Test model
    y_pred = lstm.predict(X_test, verbose=0)

    #Predict future values
    y_pred_future = lstm.predict(X_Predict, verbose=0)

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

def getPredictions(df, Predict = '1D'):
    import pandas as pd

    Predictions = pd.DataFrame()
    symbol = df['Symbol'][0]

    #print Symbol
    print("Symbol: ", symbol)

    # Predict Close
    print('Predicting Close...')
    true, test, pred, rmse = runLSTM(df, target = 'Close', window = 20, train_split = 0.8, predict = Predict)
    Close = addPredictions(len(df), true, test, rmse, 'Close')
    Predictions = addFuturePredictions(Predictions, pred, rmse, 'Close', symbol)
    
    # Predict High
    print('Predicting High...')
    true, test, pred, rmse = runLSTM(df, target = 'High', window = 20, train_split = 0.8, predict = Predict)
    High = addPredictions(len(df), true, test, rmse, 'High')
    Predictions = addFuturePredictions(Predictions, pred, rmse, 'High', symbol)

    # Predict Low
    print('Predicting Low...')
    true, test, pred, rmse = runLSTM(df, target = 'Low', window = 20, train_split = 0.8, predict = Predict)
    Low = addPredictions(len(df), true, test, pred, 'Low')
    Predictions = addFuturePredictions(Predictions, pred, rmse, 'Low', symbol)
    
    # Predict Open
    print('Predicting Open...')
    true, test, pred, rmse = runLSTM(df, target = 'Open', window = 20, train_split = 0.8, predict = Predict)
    Open = addPredictions(len(df), true, test, pred, 'Open')
    Predictions = addFuturePredictions(Predictions, pred, rmse, 'Open', symbol)

    # Predict Median_Price
    print('Predicting Median Price...')
    true, test, pred, rmse = runLSTM(df, target = 'Median_Price', window = 20, train_split = 0.8, predict = Predict)
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