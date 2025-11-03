import yfinance as yf
import pandas as pd
import numpy as np

tickers_map = {
    'トヨタ': '7203.T',
    'ソニー': '6758.T',
    'ソフトバンクG': '9984.T',
    '三菱UFJ': '8306.T',
    'NTT': '9432.T'
}
tickers = list(tickers_map.values())

# --- 1. データの取得 ---
try:
    print("過去1年間の株価データをyfinanceから取得中...")
    # auto_adjust=True がデフォルトのため、Adj Close は含まれませんが、
    # Close が調整済みになります。
    data = yf.download(tickers, period='1y') 

    # 調整済みの 'Close' (終値) を使用
    close_prices = data['Close']

    # カラム名を見やすいように企業名に置き換え
    close_prices.rename(columns={v: k for k, v in tickers_map.items()}, inplace=True)

    # --- 2. 日次対数リターン (Daily Log Returns) の計算 ---
    daily_log_returns = np.log(close_prices / close_prices.shift(1))

    # 最初の行（リターンが計算できない日）を削除
    returns_df = daily_log_returns.dropna()

    # --- 3. 結果の表示と保存 ---
    print("\n--- 日次対数リターン (Daily Log Returns) の最初の5行 ---")
    print(returns_df.head())

    print("\n--- 日次リターンの基本統計情報 ---")
    print(returns_df.describe())

    csv_file = "stock_returns_fixed.csv"
    returns_df.to_csv(csv_file)
    print(f"\nリターンデータは '{csv_file}' に保存されました。")

except Exception as e:
    print(f"\nデータの取得または処理中にエラーが発生しました。")
    print(f"エラー内容: {e}")