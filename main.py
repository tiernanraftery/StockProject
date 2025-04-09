import streamlit as st
from datetime import date


import yfinance as yf
from prophet import Prophet

from prophet.plot import plot_plotly
from plotly import graph_objs as go

START = "2019-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

#give app a title
st.title('Stock  App')

#pick stocks to show
#show apple, google, microsoft, and gamestop...
stocks = ('GOOG', 'AAPL', 'MSFT', 'GME')
#select between differnt stocks
selected_stock = st.selectbox('Select stocks for prediction', stocks)


#add slider for number of years for prediction
n_years = st.slider('Years of prediction:', 1, 5)
#calculate the period 
period = n_years * 365

#load the stock data
@st.cache_data
def load_data(ticker):
    data = yf.download(ticker, START, TODAY)
    data.reset_index(inplace=True) #put the date in the first column
    return data

#
data_load_state = st.text('Loading data...')
data = load_data(selected_stock)
data_load_state.text('Loading data... done!')

#analize the data
st.subheader('Raw data')
st.write(data.tail())


# Plot raw data
def plot_raw_data():
	fig = go.Figure()
	fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name="stock_open"))
	fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name="stock_close"))
	fig.layout.update(title_text='Time Series data with Rangeslider', xaxis_rangeslider_visible=True)
	st.plotly_chart(fig)
      



	





