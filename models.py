import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from io import BytesIO
import base64
import requests

def fetch_bitcoin_data():
    url = 'https://api.coingecko.com/api/v3/coins/bitcoin/market_chart'
    params = {'vs_currency': 'usd', 'days': '30', 'interval': 'daily'}
    response = requests.get(url, params=params)
    if response.status_code == 200:
        data = response.json()
        prices = data['prices']
        df = pd.DataFrame(prices, columns=['timestamp', 'price'])
        df['date'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('date', inplace=True)
        return df
    return None

def plot_series(time, series, format="-", start=0, end=None):
    plt.plot(time[start:end], series[start:end], format)
    plt.xlabel("Time")
    plt.ylabel("Price (USD)")
    plt.grid(True)

def predict_bitcoin_prices():
    df = fetch_bitcoin_data()

    if df is None:
        print("Failed to fetch data.")
        return None, None, None

    current_week_df = df.tail(7)
    current_week_df['date'] = current_week_df.index.strftime('%Y-%m-%d')

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

    plt.figure(figsize=(10, 6))
    plot_series(current_week_df.index, current_week_df['price'], format="-", start=0, end=7)
    plot_series(future_df['date'], future_df['predicted_price'], format="--", start=0, end=7)
    plt.legend(["Actual", "Predicted"])
    plt.title('Bitcoin Price Prediction (Next 7 Days)')

    buf = BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    plot_url = base64.b64encode(buf.read()).decode('utf-8')
    buf.close()

    current_week_data = current_week_df[['date', 'price']].to_dict(orient='records')
    future_data = future_df[['date', 'predicted_price']].to_dict(orient='records')

    return plot_url, current_week_data, future_data
