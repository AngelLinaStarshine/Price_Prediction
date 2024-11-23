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
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, r2_score

os.environ['MPLCONFIGDIR'] = os.getcwd() + "/configs/"

app = Flask(__name__)

global_data = {"current_week_data": [], "future_data": [], "plot_url": "", "backtest_results": {}}

def fetch_bitcoin_data():
    try:
        url = 'https://api.coingecko.com/api/v3/coins/bitcoin/market_chart'
        params = {'vs_currency': 'usd', 'days': '90', 'interval': 'daily'} 
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
        df['date'] = pd.to_datetime(df['timestamp'], unit='ms').dt.date 
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
    df = df.copy()  

    df.index = pd.to_datetime(df.index)

    df['days_since'] = (df.index - df.index.min()).days  
    
    X = df[['days_since']] 
    y = df['price']  

    df_clean = pd.concat([X, y], axis=1).dropna()
    X_clean = df_clean[['days_since']]
    y_clean = df_clean['price']
    
 
    X_clean = X_clean.reset_index(drop=True)
    y_clean = y_clean.reset_index(drop=True)
    
    if len(X_clean) != len(y_clean):
        raise ValueError(f"Inconsistent lengths after dropping NaNs: X ({len(X_clean)}), y ({len(y_clean)})")

    model = LinearRegression()
    model.fit(X_clean, y_clean)

    future_days = pd.DataFrame({'days_since': range(df['days_since'].max() + 1, df['days_since'].max() + 31)})
    predictions = model.predict(future_days)

    future_dates = [df.index[-1] + pd.Timedelta(days=i) for i in range(1, 31)]  
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

    actual = df['price'].iloc[1:]  
    predicted = df['predicted_price'].iloc[1:]

    if len(actual) != len(predicted):
        raise ValueError(f"Inconsistent number of samples: Actual ({len(actual)}) and Predicted ({len(predicted)})")

    mse = mean_squared_error(actual, predicted)
    rmse = np.sqrt(mse)
    mape = mean_absolute_percentage_error(actual, predicted)
    r2 = r2_score(actual, predicted)

    print(f"Mean Squared Error (MSE): {mse:.4f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
    print(f"Mean Absolute Percentage Error (MAPE): {mape:.4f}")
    print(f"R-squared: {r2:.4f}")

    return mse, rmse, mape, r2

def backtest_model_with_trading(df):
    df = df.copy()
    df['predicted_price'] = np.nan

    for i in range(len(df) - 1):
        future_df = predict_bitcoin_prices(df.iloc[:i+1])
        df.loc[df.index[i+1], 'predicted_price'] = future_df['predicted_price'].iloc[0]

    df_clean = df[['price', 'predicted_price']].dropna()

    rmse = np.sqrt(mean_squared_error(df_clean['price'], df_clean['predicted_price']))
    r2 = r2_score(df_clean['price'], df_clean['predicted_price'])
    mape = mean_absolute_percentage_error(df_clean['price'], df_clean['predicted_price'])

    df['position'] = 0  
    df['position'][df['predicted_price'] > df['price']] = 1  
    df['position'][df['predicted_price'] < df['price']] = -1  

    df['profit'] = df['position'].shift(1) * (df['price'].pct_change()) 
    df['cumulative_profit'] = (1 + df['profit']).cumprod()

    total_profit = df['cumulative_profit'].iloc[-1] - 1
    num_trades = df['position'].abs().sum() 

    return total_profit, num_trades, rmse, r2, mape, df

def update_data():
    global global_data
    print("Updating data...")
    df = fetch_bitcoin_data()
    if df is not None:
        total_profit, num_trades, rmse, r2, mape, backtest_df = backtest_model_with_trading(df)
        print(f"Total profit from backtesting: {total_profit:.2%}")
        print(f"Number of trades: {num_trades}")
        print(f"RMSE: {rmse:.2f}")
        print(f"R-squared: {r2:.2f}")
        print(f"MAPE: {mape:.2f}%")
        
        future_df = predict_bitcoin_prices(df)
        plot_url, current_week_data, future_data = generate_plot_and_data(future_df, df)

        global_data = {
            "plot_url": plot_url,
            "current_week_data": current_week_data,
            "future_data": future_data,
            "backtest_results": {
                "total_profit": total_profit,
                "num_trades": num_trades,
                "rmse": rmse,
                "r2": r2,
                "mape": mape
            }
        }
    else:
        print("Failed to update data.")


def generate_plot_and_data(future_df, df):
    plt.figure(figsize=(10, 6))

    plot_series(df.index, df['price'], format="-", start=0)

    plot_series(future_df['date'], future_df['predicted_price'], format="--", start=0)
    
    plt.legend(["Actual", "Predicted"])
    plt.title('Bitcoin Price Prediction (Next 30 Days)')

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

@app.route('/update')
def update():
    update_data()
    return redirect(url_for('index'))

@app.route('/download', methods=['POST'])
def download():
    global global_data

    # Retrieve SHA256 key from the request
    user_key = request.form.get('sha256_key')

    # Define the correct SHA256 key
    correct_key = hashlib.sha256("your_secret_key".encode()).hexdigest()

    if user_key != correct_key:
        return jsonify({"error": "Invalid SHA256 key"}), 403

    # Prepare the file content
    backtest_results = global_data.get("backtest_results", {})
    file_content = (
        f"Total profit from backtesting: {backtest_results.get('total_profit', 0):.2%}\n"
        f"Number of trades: {backtest_results.get('num_trades', 0)}\n"
        f"RMSE: {backtest_results.get('rmse', 0):.2f}\n"
        f"R-squared: {backtest_results.get('r2', 0):.2f}\n"
        f"MAPE: {backtest_results.get('mape', 0):.2%}\n"
    )

    # Send the file as a response
    return (
        file_content,
        200,
        {"Content-Disposition": "attachment; filename=backtest_results.txt"},
    )


if __name__ == "__main__":
    initialize_data()
    app.run(debug=True, use_reloader=False)