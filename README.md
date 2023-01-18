# Final Project - Predictive Technical Indicator for Trading

This project was completed by:
- Sébastien Garneau - GitHub: [538455](https://github.com/538455)

For more details, I invite you to view my full presentation [here](Presentation.pptx) or feel free to browse thru the code!

## Project / Goals
The goal of this project was to build a predictive technical indicator for trading that would beat the performance of the S&P 500 index. The S&P 500 is a US index that features 500 leading companies and is often regarded as one of the best gauges of the stock market overall.

## Process
The project consisted of three main steps:

Data collection: I used the python library yfinance to access stock market data freely available on Yahoo Finance.

Feature engineering: I used another python library called TA-Lib to calculate a dozen different technical indicators to use as input for the model.

Model training: The core of the project was a multitude of LSTM models. For each stock, I trained an LSTM model to predict the Open Price, another for the Closing Price, and two more for the High and Low Price. Each model required an average of 30 minutes to train, so we’re looking at roughly 2hrs of initial training time per stock.

To increase the accuracy of the model, I researched how LSTM learn and found an important factor: LSTM learn and make predictions based on a window that we’re setting. In my case, the model was trying to predict the next day’s closing price, by looking at the price and technical indicators of the last 20 days. To solve this, I set the last value of the window to 1, and adjusted the previous values of the window AND the predicted value to be in relation to that value of 1.

## Results
The predictions were not perfect and there’s still room for improvement. Despite, I have been able to develop an investment strategy based on this indicator. By simply following all buy/sell triggers provided by this indicator, we have an average performance (profit) of -11% in a simulated environment using a sample of 23x stocks. Although this doesn't beat the average performance of the S&P 500; it was to be expected, technical indicators are a good tool for investors, but they are not replacing the investors themselves. This means that although following blindly the buy/sell triggers of this indicator would not beat the S&P 500, it would still be a good tool for investors to use in their decision making process, which would likely lead to better results.

## Future Goals
In terms of next steps, I’d like to further increase the accuracy of the model, and explore other ways of using the predictive candle in an investment strategy. However, it's important to note that technical indicators are a good tool for investors, but they are not replacing the investors themselves.