# Stock Analyzer
Replacing technical, fundamental, and sentimental stock analysis with Machine Learning Models

## Table of Contents
- [Project Purpose](#Project-Purpose)
- [Project Overview](#Project-Overview)
- [Data Sources](#Data-Sources)
- [Data Gathering and Cleaning](#Data-Gathering-and-Cleaning)
- [Exploring Data](#Exploring-Data)
- [Modeling](#Modeling)
- [Classification](#Classification)
- [Time Series](#Time-Series)
- [Sentimental](#Sentimental)
- [Frontend and Presentation](#Frontend-and-Presentation)
- [Next Steps](#Next-Steps)

## Project Purpose
Three of the most popular stock analysis methods are technical, fundamental, and sentimental analysis.  Perhaps Machine Learning can be used to remove the human element from these methods of analysis.  For fundamental analysis, classification models were used to determine if a stock would be a Buy, Sell, or Hold.  For technical analysis, time series models were used to forecast the general direction of the stock price.  For sentiment analysis, NLP tools were used such as NLTK's VADER to determine the public's opinion and feeling towards a stock.  The hope would be that these machine learning models would enhance and improve the current methods of analysis. 

## Project Overview
1. Collected data from three different web sources by using webscraping or API calls.
  - Yahoo Finance for Time Series data
  - Stockpup for Fundamental data
  - Twitter for sentimental data
2. Data cleaning and Exploratory Data Analysis
  - Fundamental data was cleaned and formatted into a Pandas DataFrame.
  - Time series data was downloaded as daily data then resampled into weekly and monthly intervals.
  - Sentimental data was formatted into a Pandas DataFrame.
3. Modeling
  - Fundamental data was then used to train several different classification models.
  - Time series data was fitted and trained to two time series models.
  - Sentimental data did not require any modeling.
4. Presentation and Frontend
  - Streamlit was used to create a frontend for each form of analysis with their respective machine learning models.
5. Next Steps

## Data Sources
Data sources for each different method of stock analysis required different websources:
- **Stockpup.com** contained 765 stock tickers and quarterly reports for each.  This was webscraped using `BeautifulSoup` and stored as separate DataFrames in a dictionary then pickled for later use.

- **Yahoo Finance** contained the daily prices for each stock searched and could be downloaded individually as a CSV file.  `Selenium` was used to automate the downloading process for any stock entered.

- **Twitter** was used to source 200 tweets about any stock desired.  `Twint` was used to simplify the tweet webscraping process instead of Twitter's API due to the API's limited nature.

## Data Gathering and Cleaning
- __BeautifulSoup__ was used to scrape *Stockpup.com*: [Retrieving Data](Classification/Retrieving_Data.ipynb)

- __Selenium__ was used to scrape *Yahoo Finance*: [Price Scraping](Time_Series/Closing_Price_Scraping.ipynb)

- __Twint__ was used to scrape *Twitter*: [Twitter Scraping](Sentiment/Sentiment_Twitter.ipynb)

#### Data Cleaning
Cleaning Fundamental data from Stockpup.com



## Exploring Data

## Modeling

### Classification

### Time Series

### Sentimental

## Frontend and Presentation

## Next Steps
