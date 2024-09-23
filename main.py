import streamlit as st
import numpy as np
np.float_ = np.float64 #to resolve trouble with installing Prophet with using deprecated functions of numpy
import yfinance as yf
from datetime import date
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go

START = "2015-01-01" 
TODAY = date.today().strftime("%Y-%m-%d")
st.title("Cryptocurrencies Prediction App")

cryptocurrencies = ("BTC-USD", "ETH-USD", "BNB-USD", "SOL-USD")
selected_cryptocurrency = st.selectbox("Select dataset for prediction", cryptocurrencies)

n_years  = st.slider("Years of prediction:", 1, 4)
period = n_years * 365

def load_data(ticker):
    data = yf.download(ticker, START, TODAY)
    data.reset_index(inplace=True)
    return data

date_load_state = st.text("Load data...")
data = load_data(selected_cryptocurrency)
date_load_state.text("Loading data..done")

st.subheader('Raw data')
st.write(data.tail())

def plot_raw_data():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'], y = data['Open'], name='price_open'))
    fig.add_trace(go.Scatter(x=data['Date'], y = data['Close'], name='price_close'))
    fig.layout.update(title_text="Time Series Data", xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)
    
plot_raw_data()

# Forecasting
df_train = data[['Date', 'Close']]
df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})

## Creating model parameters
model_param ={
    "daily_seasonality": False,
    "weekly_seasonality":True,
    "yearly_seasonality":True,
    "seasonality_mode": "multiplicative",
    "growth": "logistic"
}

m = Prophet(**model_param)
# Setting a cap or upper limit for the forecast as we are using logistics growth
# The cap will be maximum value of target variable plus 5% of std.
df_train['cap']= df_train["y"].max() + df_train["y"].std() * 0.05 
m.fit(df_train)

future = m.make_future_dataframe(periods=period)
future['cap'] = df_train['cap'].max()
forecast = m.predict(future)
forecast.to_csv('forecast_course_'+selected_cryptocurrency + '_' + TODAY + '.csv' )

st.subheader('Forecast data')
st.write(forecast.tail())

st.write('forecast data')
fig1 = plot_plotly(m, forecast)
st.plotly_chart(fig1)

st.write('forecast components')
fig2 = m.plot_components(forecast)
st.write(fig2)