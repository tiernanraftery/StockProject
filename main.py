import streamlit as st
from datetime import date
import yfinance as yf
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go
import plotly.express as px

# Load example stock dataset from plotly
df_test = px.data.stocks()

# Get the stock columns (excluding the 'date')
stock_names = df_test.columns[1:]

# Create a dropdown for user to pick a stock
selected_stock = st.selectbox("Select stock from sample data:", stock_names)

# Create a line chart for the selected stock
fig = px.line(df_test, x='date', y=selected_stock, title=f'{selected_stock} Sample Plotly Chart')
st.plotly_chart(fig)

START = "2019-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

st.title("Stock Price Prediction App")

stocks = ("AAPL", "GOOG", "MSFT", "GME")
selected_stock = st.selectbox("Select stock:", stocks)

n_years = st.slider("Years to predict:", 1, 4)
period = n_years * 365

@st.cache_data
def load_data(ticker):
    data = yf.download(ticker, START, TODAY)
    data.reset_index(inplace=True)
    return data

st.text("Loading data...")
data = load_data(selected_stock)

if data.empty:
    st.error("No data found for this stock.")
    st.stop()

st.subheader("Raw Data")
st.write(data.tail())


df_test = px.data.stocks()
fig_test = px.line(df_test, x='date', y='GOOG', title='Test Plotly Chart')
st.plotly_chart(fig_test)

# Plot
def plot_raw_data():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name="Open"))
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name="Close"))
    fig.update_layout(title="Stock Prices", xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)

plot_raw_data()


df_train = data[["Date", "Close"]].rename(columns={"Date": "ds", "Close": "y"})
m = Prophet()
m.fit(df_train)
future = m.make_future_dataframe(periods=period)
forecast = m.predict(future)

st.subheader("Forecast Data")
st.write(forecast.tail())

st.write(f"Forecast plot for {n_years} years")
fig1 = plot_plotly(m, forecast)
st.plotly_chart(fig1)

st.write("Forecast components")
fig2 = m.plot_components(forecast)
st.write(fig2)
