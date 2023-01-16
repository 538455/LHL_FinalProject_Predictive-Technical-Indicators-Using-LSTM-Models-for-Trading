def refreshPredictions():
    import pandas as pd
    
    # Import tickerList
    tickerList = []
    with open('../../data/tickerList.txt') as f:
        for line in f:
            tickerList.append(line.strip())

    # Import wishlist
    wishlist = []
    with open('../../data/wishlist.txt') as f:
        for line in f:
            wishlist.append(line.strip())

    # Create an empty dataframe called Predictions
    Predictions = pd.DataFrame()

    # ==================================================================================================

    # Refresh the historical performance and prediction for each ticker
    
    # from Functions import dailyPrediction
    #loop thru tickerList
    for ticker in tickerList:
        historicalPerformance, prediction = dailyPrediction(ticker)

        # Save the historical performance to a csv file
        historicalPerformance.to_csv('../../data/historicalPerformance/' + ticker + '.csv')

        # Append prediction as a new row to the Predictions dataframe using pd.concat
        Predictions = pd.concat([Predictions, prediction])

    # Order the Predictions dataframe by the prediction
    Predictions = Predictions.sort_values(by='Predicted_Profit', ascending=False)

    # Save the Predictions dataframe to a csv file add the date to the file name
    import datetime
    Predictions.to_csv('../../data/Predictions/' + str(datetime.date.today()) + '_1D_Predictions.csv') # Save to the local drive, used to archive predictions
    Predictions.to_csv('Z:/+StockPredictions/Predictions.csv') # Save to the network drive, used to generate daily report

    #==================================================================================================

    # Now that the predictions have been made, we will train 2x tickers from the wishlist and add them to the tickerList
    for i in range(15):
        # Select the first ticker from the wishlist, on error, exit the loop
        try:
            ticker = wishlist[0]

            if len(ticker) > 0:

                # Train the ticker
                historicalPerformance, prediction = dailyPrediction(ticker)

                #==================================================================================================
                # We're not going to save the historical performance to a csv file because it will not be delivered in a timely manner. Predictions will be available starting the next day.
                #==================================================================================================

                # Update wishlist and tickerList
                wishlist.remove(ticker)
                tickerList.append(ticker)
        
        except:
            break

    # Update the wishlist txt file
    with open('../../data/wishlist.txt', 'w') as f:
        for item in wishlist:
            f.write("%s\r" % item)

    # Update the tickerList txt file
    with open('../../data/tickerList.txt', 'w') as f:
        for item in tickerList:
            f.write("%s\r" % item)


def dailyPrediction(ticker):
    import pandas as pd

    # Get the stock historical data
    # from Functions import getTicker_daily
    Ticker = getTicker_daily(ticker)

    # Save a copy of Ticker (backup if library is not working)
    # if len(Ticker > 1):
    #     Ticker.reset_index(inplace=True)
    #     Ticker.to_csv('../../data/1D/' + ticker + '.csv', index=False)
    # else:
    #     # Ticker = pd.read_csv('../../data/1D/' + ticker + '.csv', index_col=0) # If the library is not working, use the saved copy
    #     raise ValueError(ticker, 'is empty... There could to be an issue with the yfinance library')

    # Feature Engineering
    # from Functions import featureEngineering
    Ticker_FE = featureEngineering(Ticker)

    # Run the model to get the predictions
    # from Functions import getPredictions
    historicalPerformance, Prediction = getPredictions(Ticker_FE, Predict='1D')

    ### Format the historicalPerformance dataframe
    # Add future predictions to the historicalPerformance dataframe
    historicalPerformance = formatReturns(historicalPerformance, Prediction)

    # Add the investment columns
    openMargin=1.005 # How much higher than the predicted open price do you want to buy at? (1=100%)
    highMargin=.985 # How much lower than the predicted high price do you want to sell at? (1=100%)
    minProfit=2.5 # What is the minimum profit you want to make on a trade? (%)
    historicalPerformance = addInvestmentcols(historicalPerformance, openMargin, highMargin, minProfit)

    # Final Cleanup
    historicalPerformance = historicalPerformance[['Symbol', 'Open_Predictions', 'Open_True', 'High_Predictions', 'High_True', 'bought', 'sold', 'profit', 'Predicted_Profit', 'True_Profit', 'Low_Predictions', 'Low_True', 'Close_Predictions', 'Close_True', 'Median_Price_Predictions', 'Median_Price_True']]
    historicalPerformance = historicalPerformance[(historicalPerformance[['Open_Predictions', 'High_Predictions', 'Low_Predictions', 'Close_Predictions', 'Median_Price_Predictions']] != 0).all(axis=1)]
    historicalPerformance = historicalPerformance.dropna(subset=['Open_Predictions', 'High_Predictions', 'Low_Predictions', 'Close_Predictions', 'Median_Price_Predictions'])

    ### Add Historical Performance to Prediction
    # from Functions import formatPrediction
    Predictions = formatPrediction(historicalPerformance, openMargin, highMargin)

    return historicalPerformance, Predictions

def formatPrediction(historicalPerformance, buyMargin, sellMargin):
    import pandas as pd
    import numpy as np
    
    # Set Predictions to the last row of the historicalPerformance dataframe
    Prediction = historicalPerformance.copy()
    Prediction = Prediction.tail(1)
    Prediction = Prediction.drop(['True_Profit', 'bought', 'sold', 'profit', 'Open_True', 'High_True', 'Low_True', 'Close_True', 'Median_Price_True'], axis=1)
    
    #Create Recommendation columns and set to first
    Recommendation = Prediction.copy()
    Recommendation['buyPrice'] = round(Recommendation['Open_Predictions'] * buyMargin,2)
    Recommendation['sellPrice'] = round(Recommendation['High_Predictions'] * sellMargin, 2)
    Recommendation = Recommendation[['Predicted_Profit', 'buyPrice', 'sellPrice']]
    Prediction = Prediction.drop(['Predicted_Profit'], axis=1)
    Prediction = pd.concat([Recommendation, Prediction], axis=1)

    ## Add Length of test data
    # Count the number of unique rows where bought == 1
    bought = len(historicalPerformance.loc[historicalPerformance['bought'] == 1])
    sold = len(historicalPerformance.loc[historicalPerformance['sold'] == 1])

    # Add the length of the test data to the Prediction dataframe
    TestLen = len(historicalPerformance[historicalPerformance['Predicted_Profit'].notna()]) / 365
    Prediction['yearsTested'] = round(TestLen, 1)
    
    # Add totalProfit to the Prediction dataframe
    Prediction['totalProfit(%)'] = round(historicalPerformance['profit'].sum(), 2)

    # Add avgProfit to the Prediction dataframe
    Prediction['avgProfit(%/yr)'] = round(historicalPerformance['profit'].sum() / TestLen, 2)

    # Add totalTransactions to the Prediction dataframe
    Prediction['totalTransactions'] = len(historicalPerformance.loc[historicalPerformance['bought'] == 1])

    # Add avgTransactions to the Prediction dataframe
    Prediction['avgTransactions(yr)'] = round(len(historicalPerformance.loc[historicalPerformance['bought'] == 1]) / TestLen, 2)

    # Add successRate to the Prediction dataframe
    if bought == 0:
        Prediction['successRate(%)'] = 0
    else:
        Prediction['successRate(%)'] = round(sold / bought * 100, 2)

    # Add avgProfitWhenSuccessful to the Prediction dataframe
    Prediction['avgProfit_Successful(%)'] = round(historicalPerformance[(historicalPerformance['bought'] == 1) & (historicalPerformance['sold'] == 1)]['profit'].mean(), 2)

    # Add avgProfitWhenUnsuccessful to the Prediction dataframe
    Prediction['avgProfit_Unsuccessful(%)'] = round(historicalPerformance[(historicalPerformance['bought'] == 1) & (historicalPerformance['sold'] == 0)]['profit'].mean(), 2)
    
    ## 2022 Performance
    # #Filter to only include year 2022
    historicalPerformance_copy = historicalPerformance.copy()
    historicalPerformance_copy.reset_index(inplace=True, drop=False)
    historicalPerformance_copy.rename(columns={'index': 'Date'}, inplace=True)
    historicalPerformance_copy['Date'] = pd.to_datetime(historicalPerformance_copy['Date'])
    historicalPerformance_copy = historicalPerformance_copy[historicalPerformance_copy['Date'].dt.year == 2022]

    # Add profit2022 to the Prediction dataframe
    Prediction['profit2022(%)'] = round(historicalPerformance_copy['profit'].sum(), 2)

    # Add transactions2022 to the Prediction dataframe
    Prediction['transactions2022'] = len(historicalPerformance_copy.loc[historicalPerformance_copy['bought'] == 1])

    # Add successRate2022 to the Prediction dataframe
    if len(historicalPerformance_copy.loc[historicalPerformance_copy['bought'] == 1]) == 0:
        Prediction['successRate2022(%)'] = 0
    else:
        Prediction['successRate2022(%)'] = round(len(historicalPerformance_copy.loc[historicalPerformance_copy['sold'] == 1]) / len(historicalPerformance_copy.loc[historicalPerformance_copy['bought'] == 1]) * 100, 2)

    # Add avgProfit_Successful2022 to the Prediction dataframe
    Prediction['avgProfit_Successful2022(%)'] = round(historicalPerformance_copy[(historicalPerformance_copy['bought'] == 1) & (historicalPerformance_copy['sold'] == 1)]['profit'].mean(), 2)

    # Add avgProfit_Unsuccessful2022 to the Prediction dataframe
    Prediction['avgProfit_Unsuccessful2022(%)'] = round(historicalPerformance_copy[(historicalPerformance_copy['bought'] == 1) & (historicalPerformance_copy['sold'] == 0)]['profit'].mean(), 2)

    return Prediction

def addInvestmentcols(historicalPerformance, openMargin, highMargin, minProfit):
    import pandas as pd
    import numpy as np

    highMargin = .985
    openMargin = 1.05
    minProfit = 3.5

    # Predicted Profit
    historicalPerformance['Predicted_Profit'] = (historicalPerformance['High_Predictions']) / (historicalPerformance['Open_Predictions']) * 100 - 100

    # True Profit
    historicalPerformance['True_Profit'] = (historicalPerformance['High_True'] / historicalPerformance['Open_True']) * 100 - 100

    # Create a new column in historicalPerformance called bought and set it to 1 if Predicted_Profit > 1, Low_Predictions > Low_True and 0 otherwise
    historicalPerformance['bought'] = np.where((historicalPerformance['Predicted_Profit'] > minProfit) & ((historicalPerformance['Open_Predictions'] * (openMargin)) > historicalPerformance['Open_True']), 1, 0)

    #=====================================================================================================================================================================
    # Potential filters to add to bought
    # historicalPerformance['bought'] = np.where((historicalPerformance['bought'] == 1) & (historicalPerformance['Open_Predictions'] < historicalPerformance['Close_True'].shift(1)), 0, historicalPerformance['bought']) # Indicative of a downward trend and model is likely predicting a deadcat bounce.
    # historicalPerformance['bought'] = np.where((historicalPerformance['bought'] == 1) & (historicalPerformance['Open_Predictions'] < historicalPerformance['Low_True'].shift(1)), 0, historicalPerformance['bought']) # Indicative of a downward trend and model is likely predicting a deadcat bounce. 
    # historicalPerformance['bought'] = np.where((historicalPerformance['bought'] == 1) & (historicalPerformance['Median_Price_Predictions'] < historicalPerformance['Open_Predictions']), 0, historicalPerformance['bought']) # Model is predicting that there is more downside than upside. If I miss the High, there's a high chance to lose significantly.
    # historicalPerformance['bought'] = np.where((historicalPerformance['bought'] == 1) & (historicalPerformance['Close_Predictions'] > historicalPerformance['High_Predictions']), 0, historicalPerformance['bought']) # This is an impossible scenario. If the model predicts that the close will be higher than the high, it's likely a very volatile situation.
    # historicalPerformance['bought'] = np.where((historicalPerformance['bought'] == 1) & (abs(historicalPerformance['Median_Price_Predictions'] - historicalPerformance['Low_Predictions']) < abs(historicalPerformance['Median_Price_Predictions'] - historicalPerformance['High_Predictions'])), 0, historicalPerformance['bought']) # Could be considered a bearsish signal as average profit is lower in those situations.
    #=====================================================================================================================================================================

    # Column indicating whether or not the investment was sold
    historicalPerformance['sold'] = np.where((historicalPerformance['bought'] == 1) & ((historicalPerformance['High_Predictions'] * highMargin) < historicalPerformance['High_True']), 1, 0)

    # Column calculating the profit of the investment: 
    # If bought == 1 and sold == 1, set it to High_Predictions - Low_Predictions
    # If bought == 1 and sold == 0, set it to Close_True - Low_Predictions
    # If bought == 0 and sold == 0, set it to 0
    historicalPerformance['profit'] = np.where((historicalPerformance['bought'] == 1) & (historicalPerformance['sold'] == 1), (historicalPerformance['High_Predictions'] * highMargin) / historicalPerformance['Open_True'] * 100 -100, np.where((historicalPerformance['bought'] == 1) & (historicalPerformance['sold'] == 0), ( historicalPerformance['Close_True'] / historicalPerformance['Open_True'] * 100 - 100), 0))

    # Stop loss, if ever used
    # historicalPerformance['profit'] = np.where((historicalPerformance['profit'] > 0.01) & (historicalPerformance['profit'] < 0.97), 0.97, historicalPerformance['profit'])

    return historicalPerformance

def formatReturns(historicalPerformance, Predictions):
    import pandas as pd
    import numpy as np

    # Putting the predictions back to the original price value
    historicalPerformance['Open_Predictions'] = historicalPerformance['Open_Predictions'] * historicalPerformance['Open'].shift(1)
    historicalPerformance['Open_True'] = historicalPerformance['Open_True'] * historicalPerformance['Open'].shift(1)

    historicalPerformance['Close_Predictions'] = historicalPerformance['Close_Predictions'] * historicalPerformance['Close'].shift(1)
    historicalPerformance['Close_True'] = historicalPerformance['Close_True'] * historicalPerformance['Close'].shift(1)

    historicalPerformance['High_Predictions'] = historicalPerformance['High_Predictions'] * historicalPerformance['High'].shift(1)
    historicalPerformance['High_True'] = historicalPerformance['High_True'] * historicalPerformance['High'].shift(1)

    historicalPerformance['Low_Predictions'] = historicalPerformance['Low_Predictions'] * historicalPerformance['Low'].shift(1)
    historicalPerformance['Low_True'] = historicalPerformance['Low_True'] * historicalPerformance['Low'].shift(1)

    historicalPerformance['Median_Price_Predictions'] = historicalPerformance['Median_Price_Predictions'] * historicalPerformance['Median_Price'].shift(1)
    historicalPerformance['Median_Price_True'] = historicalPerformance['Median_Price_True'] * historicalPerformance['Median_Price'].shift(1)

    #Get Open, Close, High, Low, Volume, and Median Price of the last day in historicalPerformance and Ticker_W_Pred
    DOpen = historicalPerformance['Open'].iloc[-1]
    DClose = historicalPerformance['Close'].iloc[-1]
    DHigh = historicalPerformance['High'].iloc[-1]
    DLow = historicalPerformance['Low'].iloc[-1]
    DMedianPrice = historicalPerformance['Median_Price'].iloc[-1]

    # Add an empty row to the end of historicalPerformance
    historicalPerformance.loc[len(historicalPerformance)] = np.nan

    # set the index of the new row to the next weekday from today
    next_weekday = pd.to_datetime('today') + pd.offsets.BDay(1)
    historicalPerformance.reset_index(inplace=True, drop=False)
    historicalPerformance.loc[historicalPerformance.index[-1], 'Date'] = next_weekday
    historicalPerformance['Date'] = pd.to_datetime(historicalPerformance['Date']).dt.date
    historicalPerformance.set_index(['Date'], inplace=True)

    # Add the symbol to the new row by copying the symbol from the last row
    historicalPerformance.loc[historicalPerformance.index[-1], 'Symbol'] = historicalPerformance['Symbol'].iloc[-2]

    # Add the Prediction Price for Open to ['Open_Predictions'] last row of historicalPerformance
    historicalPerformance.loc[historicalPerformance.index[-1], 'Open_Predictions'] = Predictions.loc[Predictions['Target'] == 'Open', 'Prediction'].iloc[0] * DOpen

    # Add the Prediction Price for Close to ['Close_Predictions'] last row of historicalPerformance
    historicalPerformance.loc[historicalPerformance.index[-1], 'Close_Predictions'] = Predictions.loc[Predictions['Target'] == 'Close', 'Prediction'].iloc[0] * DClose

    # Add the Prediction Price for High to ['High_Predictions'] last row of historicalPerformance
    historicalPerformance.loc[historicalPerformance.index[-1], 'High_Predictions'] = Predictions.loc[Predictions['Target'] == 'High', 'Prediction'].iloc[0] * DHigh

    # Add the Prediction Price for Low to ['Low_Predictions'] last row of historicalPerformance
    historicalPerformance.loc[historicalPerformance.index[-1], 'Low_Predictions'] = Predictions.loc[Predictions['Target'] == 'Low', 'Prediction'].iloc[0] * DLow

    # Add the Prediction Price for Median Price to ['Median_Price_Predictions'] last row of historicalPerformance
    historicalPerformance.loc[historicalPerformance.index[-1], 'Median_Price_Predictions'] = Predictions.loc[Predictions['Target'] == 'Median_Price', 'Prediction'].iloc[0] * DMedianPrice


    # # Format the future predictions
    # Predictions.loc[Predictions['Target'] == 'Open', 'Prediction'] = Predictions.loc[Predictions['Target'] == 'Open', 'Prediction'] * DOpen
    # Predictions.loc[Predictions['Target'] == 'Close', 'Prediction'] = Predictions.loc[Predictions['Target'] == 'Close', 'Prediction'] * DClose
    # Predictions.loc[Predictions['Target'] == 'High', 'Prediction'] = Predictions.loc[Predictions['Target'] == 'High', 'Prediction'] * DHigh
    # Predictions.loc[Predictions['Target'] == 'Low', 'Prediction'] = Predictions.loc[Predictions['Target'] == 'Low', 'Prediction'] * DLow
    # Predictions.loc[Predictions['Target'] == 'Median_Price', 'Prediction'] = Predictions.loc[Predictions['Target'] == 'Median_Price', 'Prediction'] * DMedianPrice

    # # Add the future predictions to the last row of historicalPerformance
    # historicalPerformance.loc[historicalPerformance.index[-1], 'Open_Predictions'] = Predictions.loc[Predictions['Target'] == 'Open', 'Prediction'].values
    # historicalPerformance.loc[historicalPerformance.index[-1], 'Close_Predictions'] = Predictions.loc[Predictions['Target'] == 'Close', 'Prediction'].values
    # historicalPerformance.loc[historicalPerformance.index[-1], 'High_Predictions'] = Predictions.loc[Predictions['Target'] == 'High', 'Prediction'].values
    # historicalPerformance.loc[historicalPerformance.index[-1], 'Low_Predictions'] = Predictions.loc[Predictions['Target'] == 'Low', 'Prediction'].values
    # historicalPerformance.loc[historicalPerformance.index[-1], 'Median_Price_Predictions'] = Predictions.loc[Predictions['Target'] == 'Median_Price', 'Prediction'].values

    #Create Open_RMSE, Close_RMSE, High_RMSE, Low_RMSE, Median_Price_RMSE columns and set all values to the first value in Predictions.loc[Predictions['Target'] == 'Open', 'RMSE'].values
    historicalPerformance['Open_RMSE'] = np.nan
    historicalPerformance['Close_RMSE'] = np.nan
    historicalPerformance['High_RMSE'] = np.nan
    historicalPerformance['Low_RMSE'] = np.nan
    historicalPerformance['Median_Price_RMSE'] = np.nan

    # Add the RMSE values to the last row of historicalPerformance
    historicalPerformance.loc[historicalPerformance.index[-1], 'Open_RMSE'] = Predictions.loc[Predictions['Target'] == 'Open', 'RMSE'].values
    historicalPerformance.loc[historicalPerformance.index[-1], 'Close_RMSE'] = Predictions.loc[Predictions['Target'] == 'Close', 'RMSE'].values
    historicalPerformance.loc[historicalPerformance.index[-1], 'High_RMSE'] = Predictions.loc[Predictions['Target'] == 'High', 'RMSE'].values
    historicalPerformance.loc[historicalPerformance.index[-1], 'Low_RMSE'] = Predictions.loc[Predictions['Target'] == 'Low', 'RMSE'].values
    historicalPerformance.loc[historicalPerformance.index[-1], 'Median_Price_RMSE'] = Predictions.loc[Predictions['Target'] == 'Median_Price', 'RMSE'].values

    # Delete Close_Change, High_Change, Low_Change, Open_Change, Volume_Change, AO,	ROC,RSI,StochRSI,StochOsc,TSI,CMF,ATR,BB_HL,BB_LH,Ratio_EMA5,Ratio_EMA20,Ratio_EMA5_Change,Ratio_EMA20_Change,MACD,MACD_Signal,MACD_Diff,Median_Price
    historicalPerformance.drop(['Close_Change', 'High_Change', 'Low_Change', 'Open_Change', 'Volume_Change', 'AO', 'ROC', 'RSI', 'StochRSI', 'StochOsc', 'TSI', 'CMF', 'ATR', 'BB_HL', 'BB_LH', 'Ratio_EMA5', 'Ratio_EMA20', 'Ratio_EMA5_Change', 'Ratio_EMA20_Change', 'MACD', 'MACD_Signal', 'MACD_Diff', 'Median_Price'], axis=1, inplace=True)

    return historicalPerformance

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

        # Because we have an already trained model, we will not be training all of the data. Rather, we will be training the data that we have not trained yet.
        
        # Load previous split_idx value
        filename = '../models/' + predict + '/split_idx_' + ticker + '_' + target + '.txt'

        # Open txt file and read the value of split_idx
        with open(filename, 'r') as f:
            split_idx_old = int(f.read())
        
        # By comparing the split_idx_old and split_idx, we can determine the rows that we have not trained yet.
        if split_idx_old != split_idx:

            X_train = X1[split_idx_old:split_idx]
            y_train = y1[split_idx_old:split_idx]
            X_train_date = date_index[split_idx_old:split_idx]

            # Train the model
            lstm.fit(X_train, y_train, epochs=100, batch_size=5, verbose=1, shuffle=False)

            # Save trained model
            lstm.save('../models/' + predict + '/LSTM_' + ticker + '_' + target + '.h5')

            # Save the value of split_idx in a txt file, if it does not exist, create it
            with open(filename, 'w') as f:
                f.write(str(split_idx))

    
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
        
        # Save the value of split_idx in a txt file, if it does not exist, create it
        filename = '../models/' + predict + '/split_idx_' + ticker + '_' + target + '.txt'

        with open(filename, 'w') as f:
            f.write(str(split_idx))     

    # Test model
    y_pred = lstm.predict(X_test, verbose=0)

    #Predict future values
    y_pred_future = lstm.predict(X_Predict, verbose=0)

    # Calculate RMSE
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    # print(rmse) #I remove the RMSE from the tables, but I will leave this here in case I want to use it in the future for testing
    
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