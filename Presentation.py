import matplotlib.pyplot as plt
from matplotlib import rcParams
import _pickle as pickle
import numpy as np
import pandas as pd
from joblib import load
from sklearn.preprocessing import StandardScaler
import streamlit as st
from PIL import Image

# Preventing error from occuring: XGBoost causes kernel to die.
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
from xgboost import XGBClassifier
import xgboost as xgb


from fbprophet import Prophet as proph
import warnings
warnings.filterwarnings('ignore')

# Reregistering Matplotlib due to FBProphet
pd.plotting.register_matplotlib_converters()

# Loading Pretrained XGBoost Model
clf = load("Classification/classifier_model.joblib")

# Loading the top 10 features DF for scaling purposes
with open("Classification/top10_df.pkl", "rb") as fp:
    df = pickle.load(fp)
    
# Loading the new data for classification purposes
with open("Classification/stockpup.pkl", "rb") as fp:
    og_df = pickle.load(fp)
    
# Loading the ticker dictionary
with open("Classification/tickers.pkl", "rb") as fp:
    tickers = pickle.load(fp)
    
# Functions for Classifier:

def format_stock(df):
    """
    Formats the selected stock DF for use later on
    """
    
    # Setting the Datetime index
    df['Quarter end'] = pd.to_datetime(df['Quarter end'])
    df = df.set_index("Quarter end")
    
    # Replacing all "None" values with 0
    df = df.replace("None", 0)
    
    # Sorting the DF
    df = df.sort_index(ascending=True)
    
    # Converting to numeric values
    df = df.apply(pd.to_numeric)
    
    # Changing values to percent change
    for c in df.columns:
        df[c] = df[c].pct_change(1)*100
        
    # Replacing inf values with 1 and-1, as well as handling remaining NaNs
    df = df.replace([np.inf, -np.inf], [1,-1]).fillna(0)
    
    return df  


def classify_me(df1, df2, selected):
    """
    Returns the latest quarterly report available with the appropriate columns and scales it to the trained data
    """
    # Formatting the selected stock
    select_df = format_stock(df1[selected]).sort_index(ascending=True).tail(1)

    # Narrowing down the stock to the most recent QR with the neccessary columns
    select_df = select_df[[i for i in df2.columns if i != 'Decision']]

    # Resetting the the index to be able to append to the top 10 DF
    select_df = select_df.reset_index(drop=True)

    # Dropping the decision column from the top 10 DF
    dropped = df2.drop("Decision", 1)

    # Appending the new data to the top 10 DF for scaling purposes
    dropped = dropped.append(select_df)

    # Instantiating the scaler
    scaler = StandardScaler()

    # Scaling the features
    features_df = dropped

    scaled_df = pd.DataFrame(scaler.fit_transform(features_df), 
                                   index=features_df.index, 
                                   columns=features_df.columns)

    return scaled_df.iloc[[-1]]

def pie_stock(pred, name):
    """
    Creates a pie chart of the stock class predictions
    """
    rcParams['font.size'] = 30.0
    plt.style.use('bmh')
    plt.figure(figsize=(10,10))
    plt.pie(pred[0], 
            labels=['SELL', 'BUY','HOLD'], 
            shadow=True,
            explode=(0.03,.03,0.03),
            autopct='%1.1f%%')
    plt.title(name)
    plt.axis('equal')
    plt.tight_layout()
    st.pyplot()

# Functions for Time Series:

from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
import random
import glob
import os
import time

def closing_prices(stock):
    """
    Using Selenium to scrape the prices from Yahoo Finance
    """
    try:
        # Randomize wait times
        seq = [i/10 for i in range(5,12)]

        # Options to help load the page faster
        chromeOptions = Options()
        prefs = {"profile.managed_default_content_settings.images":2,
                 "profile.default_content_setting_values.notifications":2,
                 "profile.managed_default_content_settings.stylesheets":2,
                 "profile.managed_default_content_settings.cookies":1,
                 "profile.managed_default_content_settings.javascript":1,
                 "profile.managed_default_content_settings.plugins":2,
                 "profile.managed_default_content_settings.popups":2,
                 "profile.managed_default_content_settings.geolocation":2,
                 "profile.managed_default_content_settings.media_stream":2}
        chromeOptions.add_experimental_option("prefs",prefs)
        chromeOptions.add_extension(r"/Users/flatironschool/Desktop/extension_1_22_4_0.crx")

        # Opening a browser to google search
        driver = webdriver.Chrome(options=chromeOptions)
        driver.set_window_size(720,900)
        driver.implicitly_wait(3)
        driver.get("https://google.com")
        
        if len(stock) < 2:
            sterm = tickers[stock.upper()]
        else:
            sterm = stock.upper()
        
        # Searching for the stock by ticker's corresponding company name
        search = driver.find_element(By.XPATH, "//input[@class='gLFyf gsfi']")
        search.send_keys(f"{sterm} yahoo finance")
        search.send_keys(Keys.RETURN)
        time.sleep(random.choice(seq))

        # Clicking the top google result
        try:
            search_res = driver.find_element(By.XPATH, "//*[@id='rso']/div[1]/div/div/div/div/div[1]/a/h3")
            search_res.click()
        except:
            try:
                search_res = driver.find_element(By.XPATH, "//*[@id='rso']/div[1]/div/div/div/div[1]/a/h3")
                search_res.click()
            except:
                search_res = driver.find_element_by_tag_name('h3')
                search_res.click()
            
        

        # Clicking the historical data
        hist_but = driver.find_element(By.XPATH, "//*[@id='quote-nav']/ul/li[6]/a")
        hist_but.click()
        time.sleep(random.choice(seq))
        
        # Clicking the date range
        date_rg = driver.find_element(By.XPATH, "//input[@class='C(t) O(n):f Tsh($actionBlueTextShadow) Bd(n) Bgc(t) Fz(14px) Pos(r) T(-1px) Bd(n):f Bxsh(n):f Cur(p) W(190px)']")
        date_rg.click()
        time.sleep(random.choice(seq))

        # Clicking "Max", then "Done", then "Apply"
        max_but = driver.find_element(By.XPATH, "//*[@id='Col1-1-HistoricalDataTable-Proxy']/section/div[1]/div[1]/div[1]/span[2]/div/div[1]/span[8]")
        max_but.click()
        time.sleep(random.choice(seq))

        done = driver.find_element(By.XPATH, "//*[@id='Col1-1-HistoricalDataTable-Proxy']/section/div[1]/div[1]/div[1]/span[2]/div/div[3]/button[1]")
        done.click()
        time.sleep(random.choice(seq))

        apply = driver.find_element(By.XPATH, "//*[@id='Col1-1-HistoricalDataTable-Proxy']/section/div[1]/div[1]/button/span")
        apply.click()
        time.sleep(random.choice(seq[-3:]))

        # Finally downloading the CSV
        download = driver.find_element(By.XPATH, "//*[@id='Col1-1-HistoricalDataTable-Proxy']/section/div[1]/div[2]/span[2]/a/span")

        download.click()
        time.sleep(random.choice(seq[-3:]))

        # Closing the window
        driver.quit()
    
    except:
        driver.quit()
        raise ValueError()
        

def fbp(ts, stock, pers=90):
    """
    Plots the forecast of the FBProphet model
    """
    plt.style.use('fivethirtyeight')

    # Instantiating the FBP
    mod = proph(interval_width=.95, daily_seasonality=True)
    
    # Fitting the model
    mod.fit(ts)
    
    # Making future Dataframes
    future = mod.make_future_dataframe(periods=pers, freq='D')
    
    # Forecasting
    forecast = mod.predict(future)
        
    # Plotting the forecast
    mod.plot(forecast, uncertainty=True)
    
    plt.xlabel("Dates")
    plt.ylabel("Price")
    plt.xlim(['2018-10', None])
    st.pyplot()
    
    
def forecast_me(stock, pers):    
    """
    Runs the selenium script to download the closing prices, then formats the DF 
    and runs FBProphet to fit the newly formatted DF
    """
    try:
        if f"../../Downloads/{stock}.csv" not in glob.glob('../../Downloads/*.csv'):
            # Selenium script to download the latest closing prices
            closing_prices(stock)
            
            # Grabs the available stock csv file in the Downloads folder
            df = pd.read_csv(f'../../Downloads/{stock}.csv')

    except:
        return st.error("Unable to find selected stock. Please select a different one.")
    
    # Grabs the available stock csv file in the Downloads folder
    df = pd.read_csv(f'../../Downloads/{stock}.csv')
    
    # Selecting the necessary columns for FBP
    ts = df[['Date', 'Close']].tail(1000)

    # Renaming the columns for use in FB prophet
    ts.rename(columns={'Date': 'ds', 'Close': 'y'}, inplace=True)

    # Running the FBprophet model
    return fbp(ts, stock, pers=pers)

# Functions for Sentiment:

import twint
import asyncio

def twitter_scrape(ticker, tweet_cnt=200):
    """
    Scrapes the most recent tweets concerning the selected stock
    """
    # Prevents error: no current event loop in thread
    asyncio.set_event_loop(asyncio.new_event_loop())

    # Configuring Twint to search for the subject in the first specified city
    c = twint.Config()

    # Hiding the print output of tweets scraped
    c.Hide_output = True
    
    # The amount of tweets to return sorted by most recent
    c.Limit = tweet_cnt

    # Input parameters
    c.Search = '$'+str(ticker)

    # Removing retweets
    c.Filter_retweets = True

    # No pictures or video
    c.Media = False

    # English only
    c.Lang ='en'

    # Excluding tweets with links
    c.Links ='exclude'

    # Making the results pandas friendly
    c.Pandas = True

    twint.run.Search(c)

    # Assigning the DF
    df = twint.storage.panda.Tweets_df
    
    return df

def sentiment_class(score):
    """
    Labels each tweet based on its sentiment score
    """
    if score > 0:
        score = "POS+"
    elif score < 0:
        score ='NEG-'
    else:
        score = 'NEU'
        
    return score

from nltk.sentiment.vader import SentimentIntensityAnalyzer

def vader_scores(df):

    # Instantiating the sentiment analyzer
    sid = SentimentIntensityAnalyzer()

    # Grabbing the sentiment scores and assigning them to a new column
    df['sentiment'] = [sid.polarity_scores(df.tweet.iloc[i])['compound'] for i in range(len(df))]

    # Labeling the tweets in a new column
    df['feel'] = df.sentiment.apply(sentiment_class)
    
    return df

def tweet_donut(df, stock):
    plt.style.use('fivethirtyeight')
    plt.rcParams['font.size'] = 15
    fig, ax = plt.subplots(figsize=(5,5))

    ax.pie(list(df.feel.value_counts()), 
           labels=df.feel.value_counts().index, 
           autopct='%1.1f%%',
           wedgeprops = { 'linewidth': 7, 'edgecolor': 'whitesmoke' })

    circle = plt.Circle((0,0), 0.3, color='whitesmoke')
    fig = plt.gcf()
    fig.gca().add_artist(circle)

    ax.axis('equal')
    st.pyplot()
    
    
def tweet_hist(df, stock):
    plt.style.use('fivethirtyeight')

    fig, ax = plt.subplots(figsize=(10,8))

    # Plotting the sentiment scores
    ax.hist(df['sentiment'], bins=5)

    plt.title(f"Sentiment for {stock}")
    ax.set_xticks([-1,0,1])
    ax.set_xticklabels(['negative', 'neutral', 'positive'])
    plt.xlabel("Sentiment")
    plt.ylabel("# of Tweets")
    st.pyplot()
    

def create_sentiment(stock, tweet_cnt=200):
    """
    Runs all the required twitter scraping functions
    """
    # Creates a DF with tweets and sentiment scores and labels
    df = vader_scores(twitter_scrape(stock, tweet_cnt))
    
    # Creates a donut chart of the tweet count and labels
    tweet_donut(df, stock)
    
    st.subheader("Distribution of the Sentiment scores")
    
    # Creates a histogram of the sentiment scores
    tweet_hist(df, stock)
    
    
    
# Interactive Section
st.title("Stock Analyzer - Using Machine Learning")   

robot = Image.open('robot_trader.jpg')
st.image(robot, use_column_width=True)

st.header("Which stock would you like analyzed?")
selected = st.selectbox("Pick a stock", (list(tickers.keys()))).upper()
st.text("(Disclaimer: Not all stocks will be able to be shown)")


st.subheader("Select a Trading Method:")  

# Classifying the stock
if st.checkbox("Fundamental Analysis - Classification Modeling: (Observing Finances)"):
    "- Determining whether a stock is worth investing based on its financial health."
    
    # Image
    fund = Image.open('maxresdefault.jpg')
    st.image(fund, use_column_width=True)
    with st.spinner(f"Classifying {selected}..."):
        st.subheader("Classification Probability")
        
        # Assigning the DF of the newest scaled Quarterly report
        X = classify_me(og_df, df, selected)

        # Predicting the probabilities of each class
        prediction = clf.predict_proba(X)
        
        # Graphing the classes probability on a pie chart
        pie_stock(prediction, selected)
        
        st.write("_(Model Used: XGBoost)_")



# Time Series Analysis
if st.checkbox("Technical Analysis - Time Series Modeling: (Observing Price Patterns)"):
    "- Determining a stock's future price based on historical prices."
    
    # Image
    tech = Image.open('tech.png')
    st.image(tech, use_column_width=True, format='png')
    
    # Forecasting periods
    st.write("__How many days into the future would you like to forecast?__")
    periods = st.slider("Note: predictions become less accurate the further out they are.", 0, 365)

    if periods > 1:
    
        with st.spinner(f"Calculating the future of {selected}, this may take awhile..."):
            st.subheader(f"Forecasted Prices for {selected} in the next {periods} days")
            
            # Forecasting the prices
            forecast_me(selected, periods)
            
            st.text("Explanation:\n- Black dots represent actual closing prices \n- Blue line is the forecasted price \n- Blue shaded region is the confidence interval.")
            st.write("_(Model Used: Facebook Prophet)_")


# Sentiment Analysis
if st.checkbox("Sentiment Analysis - NLP on Twitter: (Observing General Opinion)"): 
    "- Determining the stock's future based on people's thoughts and opinions."
    
    # Image
    twitter = Image.open('twitter.png')
    st.image(twitter, use_column_width=True, format='png')
    
    with st.spinner(f"Getting tweets about {selected}, this may take awhile..."):
        st.subheader(f"200 Most Recent Tweets Regarding {selected}")
        
        # Graphs the donut chart and histogram of the sentiment values
        create_sentiment(selected)
        
        st.write("_(Using SentimentIntensityAnalyzer from NLTK.VADER)_")

    
    
    
    
    
    
    
    
    
    
    
    
    
    