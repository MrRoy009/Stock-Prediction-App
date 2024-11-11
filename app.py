import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from tensorflow.keras.models import load_model
import streamlit as st
from PIL import Image

# Set up Streamlit layout
st.set_page_config(layout="wide", page_title="Stock Trend Prediction")

# Define the start and end dates
start = '2010-01-01'
end = '2024-12-31'

# App Title with a stylish header
st.markdown("""
    <style>
    .title {
        font-size: 45px;
        font-weight: bold;
        color: #4e8cc7;
        text-align: center;
        font-family: 'Verdana';
    }
    </style>
    <div class="title">📈 Stock Trend Prediction App</div>
    <br>
    """, unsafe_allow_html=True)

# Load Cartoon Images
st.sidebar.image("Images\img1.jpg", use_column_width=True)

# Sidebar for User Input with custom CSS for styling
st.sidebar.markdown("""
    <style>
    .sidebar-text {
        font-size: 18px;
        color: #333;
        font-weight: bold;
        font-family: 'Verdana';
    }
    </style>
    """, unsafe_allow_html=True)
st.sidebar.markdown('<p class="sidebar-text">Select Stock Ticker:</p>', unsafe_allow_html=True)
user_input = st.sidebar.text_input('', 'AAPL')

# Fetch data directly using yfinance
df = yf.download(user_input, start=start, end=end)

# Data Summary Section
st.subheader('Data Summary from 2010-2024')
st.write(df.describe())

# Visualization: Closing Price vs Time Chart with frames
st.subheader('Closing Price vs Time Chart')
if 'Close' in df.columns and not df['Close'].empty:
    fig = plt.figure(figsize=(12, 6))
    plt.plot(df['Close'], label='Closing Price', color='blue')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.title(f'Closing Price of {user_input} Over Time')
    plt.legend()
    st.markdown("""
        <style>
        .graph-frame {
            border: 2px solid #4e8cc7;
            padding: 10px;
            border-radius: 15px;
        }
        </style>
        """, unsafe_allow_html=True)
    st.markdown('<div class="graph-frame">', unsafe_allow_html=True)
    st.pyplot(fig)
    st.markdown('</div>', unsafe_allow_html=True)
else:
    st.error("No closing price data available to plot.")

# Moving Average Chart (100-day)
st.subheader('Closing Price vs Time Chart with 100-Day Moving Average (100MA)')
ma100 = df['Close'].rolling(100).mean()
fig = plt.figure(figsize=(12, 6))
plt.plot(df['Close'], label='Closing Price', color='blue', alpha=0.5)
plt.plot(ma100, label='100-Day MA', color='red')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.markdown('<div class="graph-frame">', unsafe_allow_html=True)
st.pyplot(fig)
st.markdown('</div>', unsafe_allow_html=True)

# Moving Average Chart (100-day & 200-day)
st.subheader('Closing Price vs Time Chart with 100MA & 200MA')
ma200 = df['Close'].rolling(200).mean()
fig = plt.figure(figsize=(12, 6))
plt.plot(df['Close'], label='Closing Price', color='blue', alpha=0.5)
plt.plot(ma100, label='100-Day MA', color='red')
plt.plot(ma200, label='200-Day MA', color='green')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.markdown('<div class="graph-frame">', unsafe_allow_html=True)
st.pyplot(fig)
st.markdown('</div>', unsafe_allow_html=True)

# Splitting Data into Training and Testing
data_training = pd.DataFrame(df['Close'][0:int(len(df) * 0.70)])
data_testing = pd.DataFrame(df['Close'][int(len(df) * 0.70):])

# Show data shapes for clarity
st.sidebar.write(f"Training Data Shape: {data_training.shape}")
st.sidebar.write(f"Testing Data Shape: {data_testing.shape}")

from sklearn.preprocessing import MinMaxScaler #MinMaxScaler is a tool used for normalizing data. It scales the data to a specific range, making it easier to work with, especially in ML models.
scaler = MinMaxScaler(feature_range=(0, 1)) #--> scales all the values  of the dataset to be between 0 and 1.

data_training_array = scaler.fit_transform(data_training)

# Load the trained model
model = load_model('my_model.keras')

# Prepare testing data
past_100_days = data_training.tail(100)
final_df = pd.concat([past_100_days, data_testing], ignore_index=True)

input_data = scaler.fit_transform(final_df)

x_test = [] #--> This will contain sequences of 100 consecutive data points each, which will be used as inputs to the model
y_test = [] #--> This will contain the corresponding target values(the acutal stock price on the day right after those 100 days) that the model will try to predict
 
for i in range(100, input_data.shape[0]):
    x_test.append(input_data[i - 100:i]) #--> This takes the 100 data points right before the current i(from i-100 to i-1) and appends them as a single sequence to x_train
    y_test.append(input_data[i, 0]) #--> This takes the actual stock price at index i and appends it to y_train as the corresponding output y_train as the correspoinding output(what the model should predict).

x_test, y_test = np.array(x_test), np.array(y_test)
y_predicted = model.predict(x_test)

# Scale back to original values
scaler = scaler.scale_
scale_factor = 1 / scaler[0]
y_predicted = y_predicted * scale_factor
y_test = y_test * scale_factor

# Predictions vs Original Price Chart
st.subheader('Predictions vs Original Prices')
fig2 = plt.figure(figsize=(12, 6))
plt.plot(y_test, 'b', label='Original Price')
plt.plot(y_predicted, 'r', label='Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.title(f'Prediction vs Actual Price for {user_input}')
plt.legend()
st.markdown('<div class="graph-frame">', unsafe_allow_html=True)
st.pyplot(fig2)
st.markdown('</div>', unsafe_allow_html=True)

# Footer with cartoon figure
st.markdown("<hr>", unsafe_allow_html=True)

st.write("Developed by Swastayan. Powered by Machine Learning & Streamlit.")


