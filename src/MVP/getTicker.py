def getTickerhistory(TSX_wishlist):
    
    import pandas as pd
    import yfinance as yf

    TSX_history = pd.DataFrame()

    for index, row in TSX_wishlist.iterrows():
    
        df = pd.DataFrame()
        
        try:
            #use yfinance to get the ticker history
            df = yf.Ticker(row['symbol']).history(period='max')

            #add symbol column
            df['symbol'] = row['symbol']

            #add date column
            df['Date'] = df.index

            
        except:
            print('Error getting ticker for ' + row['symbol'])

        #if df is not empty
        if not df.empty:

            TSX_history= pd.concat([TSX_history, df], axis=0, ignore_index=True)

    return TSX_history

#=========================================================================================

def tickerInfo(ticker):
    # get the stock info, if there's an error, set info to None

    #Import yfinance
    import yfinance as yf
    import pandas as pd

    try:
        info = yf.Ticker(ticker).info
        # convert info to a df
        temp = pd.DataFrame.from_dict(info, orient='index')

        #above is the wrong orientation, so transpose it
        temp = temp.T
    
    except:
        # set temp to empty df
        temp = pd.DataFrame()    

    return temp

def getTickerinfo(TSX_Tickers):
    
    #Import yfinance
    import yfinance as yf
    import pandas as pd

    # Create a df to hold the info
    TSX_info = pd.DataFrame()

    # Loop through the tickers and get the info
    for ticker in TSX_Tickers:

        # get the info for the ticker            
        temp = pd.DataFrame()
        temp = tickerInfo(ticker)

        #if temp is empty, skip the ticker
        if temp.empty:
            continue

        else: 
            TSX_info = pd.concat([TSX_info, temp], axis=0)

    return TSX_info

