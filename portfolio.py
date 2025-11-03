import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 銘柄設定
tickers_map = {
    'トヨタ': '7203.T',
    'ソニー': '6758.T',
    'ソフトバンクG': '9984.T',
    '三菱UFJ': '8306.T',
    'NTT': '9432.T'
}
tickers = list(tickers_map.values())
num_assets = len(tickers)

# 定数設定
NUM_PORTFOLIOS = 50000  # 生成するポートフォリオの数
RISK_FREE_RATE = 0.00   # 無リスク金利 (Sharpe比計算用)
TRADING_DAYS = 250      # 年間の営業日数
PERIOD = '5y'           # 過去5年間のデータを使用

def get_portfolio_analysis():
    print(f"過去 {PERIOD} の株価データを取得中...")
    try:
        data = yf.download(tickers, period=PERIOD)
        close_prices = data['Close']
        
        close_prices.rename(columns={v: k for k, v in tickers_map.items()}, inplace=True)
        
    except Exception as e:
        print(f"データの取得に失敗しました: {e}")
        return

    # --- 1. 年率リターンと共分散行列の計算 ---
    daily_log_returns = np.log(close_prices / close_prices.shift(1)).dropna()
    annual_returns = daily_log_returns.mean() * TRADING_DAYS
    annual_cov_matrix = daily_log_returns.cov() * TRADING_DAYS
    
    # --- 2. モンテカルロ・シミュレーションの実行 ---
    # ★★★ 修正箇所: 3 + num_assets に変更 ★★★
    # results配列の行数: リターン(1) + リスク(1) + Sharpe比(1) + 重み(5) = 8
    results = np.zeros((3 + num_assets, NUM_PORTFOLIOS)) 

    print(f"モンテカルロ・シミュレーションを実行中... (ポートフォリオ数: {NUM_PORTFOLIOS})")
    for i in range(NUM_PORTFOLIOS):
        weights = np.random.random(num_assets)
        weights /= np.sum(weights)

        portfolio_return = np.sum(annual_returns * weights) * 100 
        portfolio_stddev = np.sqrt(np.dot(weights.T, np.dot(annual_cov_matrix, weights))) * 100 
        sharpe_ratio = (portfolio_return / 100 - RISK_FREE_RATE) / (portfolio_stddev / 100)
        
        # 結果を格納
        results[0, i] = portfolio_return
        results[1, i] = portfolio_stddev
        results[2, i] = sharpe_ratio
        
        # 重みを格納
        for j in range(num_assets):
            results[j+3, i] = weights[j]

    # --- 3. 最適ポートフォリオの特定 ---
    # DataFrameのカラム名は ['Return', 'Risk', 'Sharpe Ratio'] (3つ) + 銘柄名 (5つ) = 8
    results_df = pd.DataFrame(results.T, columns=['Return', 'Risk', 'Sharpe Ratio'] + list(tickers_map.keys()))
    
    max_sharpe_portfolio = results_df.loc[results_df['Sharpe Ratio'].idxmax()]
    
    # --- 4. 結果の表示 ---
    print("\n" + "="*50)
    print("🏆 最大シャープレシオ・ポートフォリオ")
    print("="*50)
    print(f"年率リターン: {max_sharpe_portfolio['Return']:.2f} %")
    print(f"年率ボラティリティ（リスク）: {max_sharpe_portfolio['Risk']:.2f} %")
    print(f"シャープレシオ: {max_sharpe_portfolio['Sharpe Ratio']:.4f}")
    print("\n--- 銘柄の重み ---")
    
    weights_output = max_sharpe_portfolio[list(tickers_map.keys())] * 100 
    print(weights_output.map('{:.2f}%'.format).to_string())

    # --- 5. 効率的フロンティアのプロット ---
    plt.figure(figsize=(12, 8))
    
    scatter = plt.scatter(results_df['Risk'], results_df['Return'], 
                          c=results_df['Sharpe Ratio'], 
                          cmap='viridis', marker='o')
    
    plt.scatter(max_sharpe_portfolio['Risk'], max_sharpe_portfolio['Return'], 
                marker='*', color='r', s=500, label='Max Sharpe Ratio Portfolio')
    
    plt.title('効率的フロンティアとモンテカルロ・シミュレーション')
    plt.xlabel('年率ボラティリティ (リスク) [%]')
    plt.ylabel('年率リターン [%]')
    plt.colorbar(scatter, label='Sharpe Ratio')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(labelspacing=0.8)
    plt.show()

if __name__ == "__main__":
    get_portfolio_analysis()