🚗 Tesla Stock Price Prediction using LSTM

GitHub ID: sunkireddy-Barath

📌 Project Description

This project focuses on predicting Tesla (TSLA) stock prices using a Deep Learning–based Long Short-Term Memory (LSTM) network.
LSTM models are well suited for time-series forecasting as they can learn long-term dependencies from historical data.

The model is trained on historical Tesla stock market data to forecast future closing prices and analyze market trends.

🎯 Project Objectives

Understand and apply LSTM networks for time-series forecasting

Preprocess real-world stock market data

Train and evaluate a deep learning model

Visualize predicted vs actual stock prices

Gain hands-on experience with financial data analysis using DL

📂 Dataset Information

Stock: Tesla Inc. (TSLA)

Source: Yahoo Finance

Data Type: Daily historical stock prices

Features Included:

Open – Price at market opening

High – Highest price during the day

Low – Lowest price during the day

Close – Price at market closing (target variable)

Adjusted Close – Close price adjusted for splits/dividends

Volume – Number of shares traded

📌 The dataset is included directly in the repository as a CSV file.

🧠 Methodology
1️⃣ Data Preprocessing

Handle missing values

Select relevant features

Normalize data using MinMaxScaler

Convert data into time-series sequences suitable for LSTM

2️⃣ Model Architecture

LSTM-based Deep Learning model

Learns temporal patterns from historical price data

Designed to predict future closing prices

3️⃣ Model Training

Dataset split into training and testing sets

Model trained using backpropagation

Loss function used: Mean Squared Error (MSE)

4️⃣ Model Evaluation

Evaluation Metrics:

Mean Squared Error (MSE)

Root Mean Squared Error (RMSE)

Visual comparison of:

Actual vs Predicted Stock Prices

🛠️ Technologies & Libraries Used

Python

TensorFlow / Keras

NumPy

Pandas

Matplotlib

Scikit-learn

Jupyter Notebook

▶️ How to Run the Project
1️⃣ Clone the Repository
git clone https://github.com/sunkireddy-Barath/tesla-stock-prediction-lstm.git

2️⃣ Install Dependencies
pip install -r requirements.txt

3️⃣ Run the Notebook
jupyter notebook


Open the notebook file and run all cells sequentially.

📊 Results & Observations

The LSTM model successfully captures trend patterns in Tesla stock prices

Predictions closely follow actual closing prices for short-term forecasting

Model performance can be improved with more features and tuning

⚠️ Limitations

Stock prices are influenced by external factors such as:

News

Market sentiment

Economic conditions

The model relies only on historical price data

Not suitable for long-term or real-time trading decisions

🔮 Future Enhancements

Add technical indicators (RSI, MACD, Moving Averages)

Integrate news or sentiment analysis

Compare with other models (GRU, ARIMA)

Deploy as a web application for live prediction

📌 Conclusion

This project demonstrates how Deep Learning (LSTM) can be effectively applied to financial time-series forecasting. It provides practical experience in data preprocessing, neural network modeling, and performance evaluation using real-world stock market data.
