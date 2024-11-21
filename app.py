import os
import numpy as np
import pandas as pd
import requests
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from io import BytesIO
import base64
from flask import Flask, render_template, jsonify, request, redirect, url_for
from apscheduler.schedulers.background import BackgroundScheduler
import threading
import hashlib
import string
import random

os.environ['MPLCONFIGDIR'] = os.getcwd() + "/configs/"

app = Flask(__name__)

global_data = {"current_week_data": [], "future_data": [], "plot_url": "", "backtest_results": {}}

def fetch_bitcoin_data():
    try:
        url = 'https://api.coingecko.com/api/v3/coins/bitcoin/market_chart'
        params = {'vs_currency': 'usd', 'days': '365', 'interval': 'daily'}  
        response = requests.get(url, params=params)
        response.raise_for_status()  
    
        try:
            data = response.json() 
        except ValueError:
            print(f"Error parsing JSON response: {response.text}")
            return None

        prices = data.get('prices', [])
        if not prices:
            print("No price data found in response.")
            return None
    
        df = pd.DataFrame(prices, columns=['timestamp', 'price'])
        df['date'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('date', inplace=True)
        return df
    except requests.exceptions.RequestException as e:
        print(f"Error fetching data: {e}")
        return None

def plot_series(time, series, format="-", start=0, end=None):
    plt.plot(time[start:end], series[start:end], format)
    plt.xlabel("Time")
    plt.ylabel("Price (USD)")
    plt.grid(True)

def predict_bitcoin_prices(df):
    df['days_since'] = (df.index - df.index.min()).days
    X = df[['days_since']]
    y = df['price']

    model = LinearRegression()
    model.fit(X, y)

    future_days = pd.DataFrame({'days_since': range(df['days_since'].max() + 1, df['days_since'].max() + 8)})
    predictions = model.predict(future_days)

    future_dates = [df.index[-1] + pd.Timedelta(days=i) for i in range(1, 8)]
    future_df = pd.DataFrame({
        'date': future_dates,
        'predicted_price': predictions
    })

    return future_df

def backtest_model(df):
    df['predicted_price'] = np.nan
    for i in range(len(df) - 1):
        future_df = predict_bitcoin_prices(df.iloc[:i+1])
        df.loc[df.index[i+1], 'predicted_price'] = future_df['predicted_price'].iloc[0]

    df['position'] = 0  
    df['position'][df['predicted_price'] > df['price']] = 1  
    df['position'][df['predicted_price'] < df['price']] = -1  

    df['profit'] = df['position'].shift(1) * (df['price'].pct_change()) 

    df['cumulative_profit'] = (1 + df['profit']).cumprod()

    total_profit = df['cumulative_profit'].iloc[-1] - 1
    num_trades = df['position'].abs().sum() 

    return total_profit, num_trades, df

def update_data():
    global global_data
    print("Updating data...")
    df = fetch_bitcoin_data()
    if df is not None:
        total_profit, num_trades, backtest_df = backtest_model(df)
        print(f"Total profit from backtesting: {total_profit:.2%}")
        print(f"Number of trades: {num_trades}")
        
        future_df = predict_bitcoin_prices(df)
        plot_url, current_week_data, future_data = generate_plot_and_data(future_df, df)

        global_data = {
            "plot_url": plot_url,
            "current_week_data": current_week_data,
            "future_data": future_data,
            "backtest_results": {
                "total_profit": total_profit,
                "num_trades": num_trades
            }
        }
    else:
        print("Failed to update data.")

def generate_plot_and_data(future_df, df):
    plt.figure(figsize=(10, 6))

    plot_series(df.index, df['price'], format="-", start=0)

    plot_series(future_df['date'], future_df['predicted_price'], format="--", start=0)
    
    plt.legend(["Actual", "Predicted"])
    plt.title('Bitcoin Price Prediction (Next 7 Days)')

    buf = BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    plot_url = base64.b64encode(buf.read()).decode('utf-8')
    buf.close()

    current_week_data = df.tail(7)[['price']].round(2).reset_index().to_dict(orient='records')
    
    future_data = future_df[['date', 'predicted_price']].round(2).reset_index(drop=True).to_dict(orient='records')

    return plot_url, current_week_data, future_data

scheduler = BackgroundScheduler()
scheduler.add_job(update_data, 'interval', hours=1)
scheduler.start()

def initialize_data():
    update_data()

def convert_to_native(data):
    if isinstance(data, pd.DataFrame):
        return data.applymap(lambda x: x.item() if isinstance(x, np.generic) else x).to_dict(orient='records')
    elif isinstance(data, pd.Series):
        return data.apply(lambda x: x.item() if isinstance(x, np.generic) else x).to_dict()
    elif isinstance(data, list):
        return [convert_to_native(item) for item in data]
    elif isinstance(data, dict):
        return {key: convert_to_native(value) for key, value in data.items()}
    elif isinstance(data, np.generic): 
        return data.item()  
    else:
        return data

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/data_analysis', methods=['POST'])
def data_analysis():
    global global_data

    native_global_data = convert_to_native(global_data)
    
    return jsonify(native_global_data)

@app.route('/generate_sha256_key')
def generate_sha256_key():
    random_string = ''.join(random.choices(string.ascii_letters + string.digits, k=32))
    sha256_key = hashlib.sha256(random_string.encode()).hexdigest()  
    return jsonify({"key": sha256_key})

@app.route('/validate_key', methods=['POST'])
def validate_key():
    private_key = request.form['private_key']

    if len(private_key) == 64: 
        return redirect(url_for('index'))
    else:
        return 'Invalid Key', 400

if __name__ == '__main__':
    initialize_data()  
    app.run(debug=True) 