# Stock Analyzer
Replacing technical, fundamental, and sentimental stock analysis with Machine Learning Models

## Table of Contents
- [Project Purpose](#Project-Purpose)
- [Project Overview](#Project-Overview)
- [Data Sources](#Data-Sources)
- [Data Gathering](#Data-Gathering)
- [Exploring Data](#Exploring-Data)
- [Modeling](#Modeling)
- [Classification](#Classification)
- [Time Series](#Time-Series)
- [Sentimental](#Sentimental)
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

## Data Gathering

## Exploring Data

## Modeling

### Classification

### Time Series

### Sentimental

## Next Steps
