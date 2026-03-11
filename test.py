import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import requests
from sklearn.linear_model import LinearRegression
from textblob import TextBlob

st.set_page_config(page_title="AI Finance Terminal", layout="wide")

# --------------------------------------------------
# BLOOMBERG STYLE THEME
# --------------------------------------------------

st.markdown("""
<style>

body {
background-color:#0a0f1c;
color:white;
}

.stApp {
background-color:#0a0f1c;
}

h1,h2,h3,h4{
color:#00ffcc;
}

[data-testid="stSidebar"]{
background-color:#111827;
}

</style>
""", unsafe_allow_html=True)

st.title("AI Finance Terminal")

# --------------------------------------------------
# SIDEBAR MENU
# --------------------------------------------------

menu = st.sidebar.selectbox(
"Select Tool",
[
"Market Dashboard",
"Stock Analysis Terminal",
"Undervalued Stock Finder",
"Portfolio Analyzer",
"Multibagger Predictor",
"News Sentiment Analyzer"
]
)

# --------------------------------------------------
# MARKET DASHBOARD
# --------------------------------------------------

if menu == "Market Dashboard":

    st.header("Global Market Overview")

    indices = {
    "NIFTY 50":"^NSEI",
    "SENSEX":"^BSESN",
    "NASDAQ":"^IXIC",
    "S&P 500":"^GSPC"
    }

    data=[]

    for name,ticker in indices.items():

        stock=yf.Ticker(ticker)
        hist=stock.history(period="1d")

        price=hist["Close"].iloc[-1]

        data.append([name,price])

    df=pd.DataFrame(data,columns=["Index","Price"])

    st.dataframe(df,use_container_width=True)

# --------------------------------------------------
# STOCK ANALYSIS
# --------------------------------------------------

elif menu=="Stock Analysis Terminal":

    st.header("Stock Analysis Terminal")

    ticker=st.text_input("Enter Stock Ticker","RELIANCE.NS")

    if ticker:

        stock=yf.Ticker(ticker)

        data=stock.history(period="1y")

        st.subheader("Price Chart")

        fig=go.Figure()

        fig.add_trace(go.Scatter(
        x=data.index,
        y=data["Close"],
        name="Close Price"
        ))

        st.plotly_chart(fig,use_container_width=True)

        st.subheader("Financial Metrics")

        info=stock.info

        col1,col2,col3=st.columns(3)

        col1.metric("PE Ratio",info.get("trailingPE"))
        col2.metric("Market Cap",info.get("marketCap"))
        col3.metric("Dividend Yield",info.get("dividendYield"))

# --------------------------------------------------
# UNDERVALUED STOCK FINDER
# --------------------------------------------------

elif menu=="Undervalued Stock Finder":

    st.header("Undervalued Stock Analyzer")

    ticker=st.text_input("Stock Ticker","INFY.NS")

    if ticker:

        stock=yf.Ticker(ticker)
        info=stock.info

        pe=info.get("trailingPE")
        pb=info.get("priceToBook")
        roe=info.get("returnOnEquity")

        st.write("PE Ratio:",pe)
        st.write("PB Ratio:",pb)
        st.write("ROE:",roe)

        score=0

        if pe and pe<20:
            score+=1

        if pb and pb<3:
            score+=1

        if roe and roe>0.15:
            score+=1

        if score>=2:

            st.success("Stock may be UNDERVALUED")

        else:

            st.warning("Stock may be fairly valued")

# --------------------------------------------------
# PORTFOLIO ANALYZER
# --------------------------------------------------

elif menu=="Portfolio Analyzer":

    st.header("Portfolio Risk Analyzer")

    stocks=st.text_area(
    "Enter Stocks (comma separated)",
    "RELIANCE.NS,TCS.NS,INFY.NS"
    )

    if stocks:

        tickers=stocks.split(",")

        data=yf.download(tickers,period="1y")["Close"]

        returns=data.pct_change().dropna()

        portfolio_return=returns.mean().mean()*252

        portfolio_risk=returns.std().mean()*np.sqrt(252)

        st.metric("Expected Annual Return",round(portfolio_return,2))
        st.metric("Portfolio Risk",round(portfolio_risk,2))

        st.line_chart(data)

# --------------------------------------------------
# MULTIBAGGER PREDICTOR
# --------------------------------------------------

elif menu=="Multibagger Predictor":

    st.header("AI Multibagger Stock Predictor")

    ticker=st.text_input("Ticker","TATAMOTORS.NS")

    if ticker:

        stock=yf.Ticker(ticker)

        data=stock.history(period="5y")

        data["days"]=np.arange(len(data))

        X=data[["days"]]

        y=data["Close"]

        model=LinearRegression()

        model.fit(X,y)

        future=np.array([[len(data)+365]])

        prediction=model.predict(future)

        current_price=data["Close"].iloc[-1]

        predicted_price=prediction[0]

        growth=((predicted_price-current_price)/current_price)*100

        st.write("Current Price:",round(current_price,2))
        st.write("Predicted Price (1 year):",round(predicted_price,2))
        st.write("Expected Growth %:",round(growth,2))

        if growth>100:

            st.success("Potential Multibagger")

        elif growth>40:

            st.info("High Growth Stock")

        else:

            st.warning("Moderate Growth")

        fig=go.Figure()

        fig.add_trace(go.Scatter(
        x=data.index,
        y=data["Close"],
        name="Historical Price"
        ))

        st.plotly_chart(fig,use_container_width=True)

# --------------------------------------------------
# NEWS SENTIMENT ANALYZER
# --------------------------------------------------

elif menu=="News Sentiment Analyzer":

    st.header("Financial News Sentiment")

    company=st.text_input("Company Name","Tesla")

    if company:

        url=f"https://newsapi.org/v2/everything?q={company}&apiKey=YOUR_API_KEY"

        try:

            response=requests.get(url)

            articles=response.json()["articles"][:5]

            for article in articles:

                title=article["title"]

                sentiment=TextBlob(title).sentiment.polarity

                if sentiment>0:

                    label="Positive"

                elif sentiment<0:

                    label="Negative"

                else:

                    label="Neutral"

                st.write(title)

                st.write("Sentiment:",label)

                st.write("---")

        except:

            st.warning("Add your NewsAPI key to enable news analysis")