#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
黄金价格预测程序
================
功能：
1. 获取过去一年的金价走势
2. 整合多个影响因子（美元指数、利率、VIX等）
3. 使用机器学习模型预测明天的金价

作者：AI Assistant
日期：2026-01-24
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial Unicode MS', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False


class GoldPricePredictor:
    """黄金价格预测器"""
    
    def __init__(self):
        self.data = None
        self.model = None
        self.scaler = StandardScaler()
        self.feature_columns = []
        
    def fetch_data(self, period_days=365):
        """
        获取黄金价格及相关因子数据
        
        因子说明：
        - GLD: 黄金ETF（代表金价）
        - DX-Y.NYB: 美元指数（负相关）
        - ^TNX: 10年期美债收益率（负相关）
        - ^VIX: 恐慌指数（正相关，避险情绪）
        - TIP: 通胀保值债券ETF（通胀预期）
        - USO: 原油ETF（通胀关联）
        - SLV: 白银ETF（贵金属联动）
        - SPY: 标普500 ETF（风险偏好）
        """
        print("=" * 60)
        print("正在获取市场数据...")
        print("=" * 60)
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=period_days + 60)  # 多取60天用于计算技术指标
        
        # 定义要获取的标的
        tickers = {
            'GLD': '黄金ETF (金价代理)',
            'DX-Y.NYB': '美元指数',
            '^TNX': '10年期美债收益率',
            '^VIX': 'VIX恐慌指数',
            'TIP': '通胀保值债券ETF',
            'USO': '原油ETF',
            'SLV': '白银ETF',
            'SPY': '标普500 ETF'
        }
        
        # 获取数据
        data_frames = {}
        for ticker, desc in tickers.items():
            try:
                df = yf.download(ticker, start=start_date, end=end_date, progress=False)
                if not df.empty:
                    # 处理多层列名
                    if isinstance(df.columns, pd.MultiIndex):
                        df.columns = df.columns.get_level_values(0)
                    data_frames[ticker] = df['Close']
                    print(f"  ✓ {desc} ({ticker}): {len(df)} 条数据")
                else:
                    print(f"  ✗ {desc} ({ticker}): 无数据")
            except Exception as e:
                print(f"  ✗ {desc} ({ticker}): 获取失败 - {e}")
        
        # 合并数据
        combined = pd.DataFrame(data_frames)
        combined = combined.dropna()
        
        # 重命名列
        column_mapping = {
            'GLD': 'gold_price',
            'DX-Y.NYB': 'usd_index',
            '^TNX': 'us10y_yield',
            '^VIX': 'vix',
            'TIP': 'tip_inflation',
            'USO': 'oil_price',
            'SLV': 'silver_price',
            'SPY': 'sp500'
        }
        combined = combined.rename(columns=column_mapping)
        
        print(f"\n合并后数据: {len(combined)} 条记录")
        print(f"时间范围: {combined.index.min().strftime('%Y-%m-%d')} 至 {combined.index.max().strftime('%Y-%m-%d')}")
        
        self.data = combined
        return combined
    
    def create_features(self):
        """创建特征工程"""
        print("\n" + "=" * 60)
        print("正在创建特征...")
        print("=" * 60)
        
        df = self.data.copy()
        
        # 1. 价格变化率（日收益率）
        for col in ['gold_price', 'usd_index', 'silver_price', 'sp500', 'oil_price']:
            if col in df.columns:
                df[f'{col}_return'] = df[col].pct_change()
        
        # 2. 移动平均线
        for window in [5, 10, 20, 50]:
            df[f'gold_ma{window}'] = df['gold_price'].rolling(window=window).mean()
            df[f'gold_ma{window}_ratio'] = df['gold_price'] / df[f'gold_ma{window}']
        
        # 3. 波动率（20日滚动标准差）
        df['gold_volatility'] = df['gold_price'].pct_change().rolling(window=20).std()
        
        # 4. RSI指标（相对强弱指数）
        delta = df['gold_price'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['gold_rsi'] = 100 - (100 / (1 + rs))
        
        # 5. MACD指标
        exp1 = df['gold_price'].ewm(span=12, adjust=False).mean()
        exp2 = df['gold_price'].ewm(span=26, adjust=False).mean()
        df['gold_macd'] = exp1 - exp2
        df['gold_macd_signal'] = df['gold_macd'].ewm(span=9, adjust=False).mean()
        
        # 6. 金银比
        if 'silver_price' in df.columns:
            df['gold_silver_ratio'] = df['gold_price'] / df['silver_price']
        
        # 7. 相对于美元指数的强弱
        if 'usd_index' in df.columns:
            df['gold_usd_ratio'] = df['gold_price'] / df['usd_index']
        
        # 8. VIX变化率
        if 'vix' in df.columns:
            df['vix_change'] = df['vix'].pct_change()
        
        # 9. 滞后特征（用于预测）
        for lag in [1, 2, 3, 5]:
            df[f'gold_lag{lag}'] = df['gold_price'].shift(lag)
            df[f'gold_return_lag{lag}'] = df['gold_price_return'].shift(lag)
        
        # 10. 目标变量：明天的金价
        df['target'] = df['gold_price'].shift(-1)
        
        # 删除NaN
        df = df.dropna()
        
        # 只保留最近一年的数据
        df = df.tail(252)  # 约一年交易日
        
        self.data = df
        
        # 定义特征列
        self.feature_columns = [
            'gold_price', 'usd_index', 'us10y_yield', 'vix', 
            'tip_inflation', 'oil_price', 'silver_price', 'sp500',
            'gold_price_return', 'usd_index_return', 'silver_price_return',
            'gold_ma5_ratio', 'gold_ma10_ratio', 'gold_ma20_ratio',
            'gold_volatility', 'gold_rsi', 'gold_macd',
            'gold_silver_ratio', 'gold_usd_ratio', 'vix_change',
            'gold_lag1', 'gold_lag2', 'gold_lag3',
            'gold_return_lag1', 'gold_return_lag2'
        ]
        
        # 确保所有特征列都存在
        self.feature_columns = [col for col in self.feature_columns if col in df.columns]
        
        print(f"创建了 {len(self.feature_columns)} 个特征")
        print(f"特征列表: {', '.join(self.feature_columns[:10])}...")
        
        return df
    
    def train_model(self):
        """训练预测模型"""
        print("\n" + "=" * 60)
        print("正在训练模型...")
        print("=" * 60)
        
        df = self.data.copy()
        
        X = df[self.feature_columns]
        y = df['target']
        
        # 分割训练集和测试集
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, shuffle=False  # 时间序列不打乱
        )
        
        # 标准化
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # 尝试多个模型
        models = {
            'Linear Regression': LinearRegression(),
            'Ridge Regression': Ridge(alpha=1.0),
            'Random Forest': RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42),
            'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, max_depth=5, random_state=42)
        }
        
        best_model = None
        best_score = -np.inf
        results = []
        
        for name, model in models.items():
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            
            mse = mean_squared_error(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            results.append({
                'model': name,
                'MSE': mse,
                'MAE': mae,
                'R2': r2
            })
            
            print(f"\n{name}:")
            print(f"  MSE: {mse:.4f}")
            print(f"  MAE: {mae:.4f}")
            print(f"  R²:  {r2:.4f}")
            
            if r2 > best_score:
                best_score = r2
                best_model = model
                self.best_model_name = name
        
        self.model = best_model
        print(f"\n✓ 最佳模型: {self.best_model_name} (R² = {best_score:.4f})")
        
        return results
    
    def predict_tomorrow(self):
        """预测明天的金价"""
        print("\n" + "=" * 60)
        print("预测明天金价...")
        print("=" * 60)
        
        # 获取最新一天的特征
        latest = self.data[self.feature_columns].iloc[-1:].copy()
        latest_scaled = self.scaler.transform(latest)
        
        # 预测
        prediction = self.model.predict(latest_scaled)[0]
        
        # 获取当前价格
        current_price = self.data['gold_price'].iloc[-1]
        change = prediction - current_price
        change_pct = (change / current_price) * 100
        
        # 获取日期
        current_date = self.data.index[-1]
        tomorrow = current_date + timedelta(days=1)
        # 跳过周末
        while tomorrow.weekday() >= 5:
            tomorrow += timedelta(days=1)
        
        result = {
            'current_date': current_date.strftime('%Y-%m-%d'),
            'current_price': current_price,
            'prediction_date': tomorrow.strftime('%Y-%m-%d'),
            'predicted_price': prediction,
            'change': change,
            'change_pct': change_pct
        }
        
        print(f"\n当前日期: {result['current_date']}")
        print(f"当前金价 (GLD): ${current_price:.2f}")
        print(f"\n预测日期: {result['prediction_date']}")
        print(f"预测金价 (GLD): ${prediction:.2f}")
        print(f"预测变化: {'+' if change >= 0 else ''}{change:.2f} ({'+' if change_pct >= 0 else ''}{change_pct:.2f}%)")
        
        # 给出方向建议
        if change_pct > 0.5:
            direction = "📈 看涨"
        elif change_pct < -0.5:
            direction = "📉 看跌"
        else:
            direction = "➡️ 震荡"
        
        print(f"\n趋势判断: {direction}")
        
        return result
    
    def analyze_factors(self):
        """分析各因子对金价的影响"""
        print("\n" + "=" * 60)
        print("因子影响力分析...")
        print("=" * 60)
        
        df = self.data.copy()
        
        # 计算相关性
        correlations = {}
        target = df['gold_price']
        
        factor_names = {
            'usd_index': '美元指数',
            'us10y_yield': '10年期美债收益率',
            'vix': 'VIX恐慌指数',
            'tip_inflation': '通胀预期(TIP)',
            'oil_price': '原油价格',
            'silver_price': '白银价格',
            'sp500': '标普500'
        }
        
        print("\n各因子与金价的相关性:")
        print("-" * 40)
        
        for col, name in factor_names.items():
            if col in df.columns:
                corr = df[col].corr(target)
                correlations[name] = corr
                bar = "█" * int(abs(corr) * 20)
                sign = "+" if corr > 0 else "-"
                print(f"{name:20s}: {sign}{abs(corr):.3f} {bar}")
        
        # 如果是树模型，显示特征重要性
        if hasattr(self.model, 'feature_importances_'):
            print("\n\n特征重要性 (来自模型):")
            print("-" * 40)
            
            importances = self.model.feature_importances_
            feature_imp = list(zip(self.feature_columns, importances))
            feature_imp.sort(key=lambda x: x[1], reverse=True)
            
            for feat, imp in feature_imp[:10]:
                bar = "█" * int(imp * 100)
                print(f"{feat:25s}: {imp:.3f} {bar}")
        
        return correlations
    
    def plot_analysis(self, save_path='gold_analysis.png'):
        """生成分析图表"""
        print("\n" + "=" * 60)
        print("生成分析图表...")
        print("=" * 60)
        
        df = self.data.copy()
        
        fig, axes = plt.subplots(3, 2, figsize=(16, 14))
        fig.suptitle('Gold Price Analysis & Prediction\n黄金价格分析与预测', fontsize=16, fontweight='bold')
        
        # 1. 金价走势 + 移动平均线
        ax1 = axes[0, 0]
        ax1.plot(df.index, df['gold_price'], label='GLD Price', color='gold', linewidth=2)
        ax1.plot(df.index, df['gold_ma20'], label='MA20', color='blue', linestyle='--', alpha=0.7)
        ax1.plot(df.index, df['gold_ma50'], label='MA50', color='red', linestyle='--', alpha=0.7)
        ax1.set_title('Gold Price (GLD) with Moving Averages\n金价走势与移动平均线')
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Price ($)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)
        
        # 2. 金价 vs 美元指数
        ax2 = axes[0, 1]
        ax2_twin = ax2.twinx()
        ax2.plot(df.index, df['gold_price'], label='Gold (GLD)', color='gold', linewidth=2)
        ax2_twin.plot(df.index, df['usd_index'], label='USD Index', color='green', linewidth=2)
        ax2.set_title('Gold vs USD Index\n金价 vs 美元指数')
        ax2.set_xlabel('Date')
        ax2.set_ylabel('Gold Price ($)', color='gold')
        ax2_twin.set_ylabel('USD Index', color='green')
        ax2.legend(loc='upper left')
        ax2_twin.legend(loc='upper right')
        ax2.grid(True, alpha=0.3)
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)
        
        # 3. RSI指标
        ax3 = axes[1, 0]
        ax3.plot(df.index, df['gold_rsi'], label='RSI(14)', color='purple', linewidth=1.5)
        ax3.axhline(y=70, color='red', linestyle='--', alpha=0.7, label='Overbought (70)')
        ax3.axhline(y=30, color='green', linestyle='--', alpha=0.7, label='Oversold (30)')
        ax3.fill_between(df.index, 30, 70, alpha=0.1, color='gray')
        ax3.set_title('RSI Indicator\nRSI相对强弱指标')
        ax3.set_xlabel('Date')
        ax3.set_ylabel('RSI')
        ax3.set_ylim(0, 100)
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        ax3.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45)
        
        # 4. VIX恐慌指数
        ax4 = axes[1, 1]
        ax4.fill_between(df.index, df['vix'], alpha=0.3, color='red')
        ax4.plot(df.index, df['vix'], label='VIX', color='red', linewidth=1.5)
        ax4.axhline(y=20, color='orange', linestyle='--', alpha=0.7, label='Normal Level')
        ax4.set_title('VIX Fear Index\nVIX恐慌指数')
        ax4.set_xlabel('Date')
        ax4.set_ylabel('VIX')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        ax4.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        plt.setp(ax4.xaxis.get_majorticklabels(), rotation=45)
        
        # 5. 相关性热力图
        ax5 = axes[2, 0]
        corr_cols = ['gold_price', 'usd_index', 'us10y_yield', 'vix', 'silver_price', 'sp500', 'oil_price']
        corr_cols = [c for c in corr_cols if c in df.columns]
        corr_matrix = df[corr_cols].corr()
        sns.heatmap(corr_matrix, annot=True, cmap='RdYlGn', center=0, ax=ax5, fmt='.2f')
        ax5.set_title('Factor Correlation Matrix\n因子相关性矩阵')
        
        # 6. 特征重要性
        ax6 = axes[2, 1]
        if hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
            feature_imp = list(zip(self.feature_columns, importances))
            feature_imp.sort(key=lambda x: x[1], reverse=True)
            top_features = feature_imp[:10]
            
            features = [f[0] for f in top_features]
            values = [f[1] for f in top_features]
            
            bars = ax6.barh(range(len(features)), values, color='steelblue')
            ax6.set_yticks(range(len(features)))
            ax6.set_yticklabels(features)
            ax6.invert_yaxis()
            ax6.set_title('Top 10 Feature Importance\n前10个重要特征')
            ax6.set_xlabel('Importance')
        else:
            ax6.text(0.5, 0.5, 'Feature importance\nnot available for\nthis model type', 
                    ha='center', va='center', fontsize=12)
            ax6.set_title('Feature Importance\n特征重要性')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ 图表已保存至: {save_path}")
        
        return fig
    
    def generate_report(self, prediction_result):
        """生成分析报告"""
        print("\n" + "=" * 60)
        print("生成分析报告...")
        print("=" * 60)
        
        df = self.data
        
        # 计算统计数据
        latest_price = df['gold_price'].iloc[-1]
        price_1m_ago = df['gold_price'].iloc[-22] if len(df) > 22 else df['gold_price'].iloc[0]
        price_3m_ago = df['gold_price'].iloc[-66] if len(df) > 66 else df['gold_price'].iloc[0]
        price_1y_ago = df['gold_price'].iloc[0]
        
        return_1m = (latest_price / price_1m_ago - 1) * 100
        return_3m = (latest_price / price_3m_ago - 1) * 100
        return_1y = (latest_price / price_1y_ago - 1) * 100
        
        max_price = df['gold_price'].max()
        min_price = df['gold_price'].min()
        avg_price = df['gold_price'].mean()
        
        latest_rsi = df['gold_rsi'].iloc[-1]
        latest_vix = df['vix'].iloc[-1]
        latest_usd = df['usd_index'].iloc[-1]
        
        report = f"""
╔══════════════════════════════════════════════════════════════════╗
║                    黄金价格分析报告                              ║
║                Gold Price Analysis Report                        ║
╠══════════════════════════════════════════════════════════════════╣
║  报告生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}                           ║
╚══════════════════════════════════════════════════════════════════╝

【一、价格概览】
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  当前价格 (GLD): ${latest_price:.2f}
  过去1个月涨跌:  {'+' if return_1m >= 0 else ''}{return_1m:.2f}%
  过去3个月涨跌:  {'+' if return_3m >= 0 else ''}{return_3m:.2f}%
  过去1年涨跌:    {'+' if return_1y >= 0 else ''}{return_1y:.2f}%
  
  一年最高价: ${max_price:.2f}
  一年最低价: ${min_price:.2f}
  一年平均价: ${avg_price:.2f}

【二、技术指标】
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  RSI(14):     {latest_rsi:.1f} {'(超买区域⚠️)' if latest_rsi > 70 else '(超卖区域⚠️)' if latest_rsi < 30 else '(正常区间)'}
  VIX恐慌指数: {latest_vix:.1f} {'(市场恐慌⚠️)' if latest_vix > 30 else '(情绪稳定)'}
  美元指数:    {latest_usd:.2f}

【三、明日预测】
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  预测模型:   {self.best_model_name}
  预测日期:   {prediction_result['prediction_date']}
  预测价格:   ${prediction_result['predicted_price']:.2f}
  预测涨跌:   {'+' if prediction_result['change'] >= 0 else ''}{prediction_result['change']:.2f} ({'+' if prediction_result['change_pct'] >= 0 else ''}{prediction_result['change_pct']:.2f}%)
  
  趋势判断:   {'📈 看涨' if prediction_result['change_pct'] > 0.5 else '📉 看跌' if prediction_result['change_pct'] < -0.5 else '➡️ 震荡'}

【四、关键影响因子】
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  · 美元指数 (负相关):   当前 {latest_usd:.2f}
  · 美债收益率 (负相关): 当前 {df['us10y_yield'].iloc[-1]:.2f}%
  · VIX恐慌 (正相关):    当前 {latest_vix:.1f}
  · 白银价格 (正相关):   当前 ${df['silver_price'].iloc[-1]:.2f}

【五、风险提示】
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  ⚠️  本预测仅供参考，不构成投资建议
  ⚠️  模型基于历史数据，无法预测黑天鹅事件
  ⚠️  金价受地缘政治、央行政策等多重因素影响
  ⚠️  建议结合基本面分析综合判断

══════════════════════════════════════════════════════════════════
"""
        print(report)
        
        # 保存报告
        with open('gold_analysis_report.txt', 'w', encoding='utf-8') as f:
            f.write(report)
        print("✓ 报告已保存至: gold_analysis_report.txt")
        
        return report


def main():
    """主函数"""
    print("""
    ╔═══════════════════════════════════════════════════════════╗
    ║           黄金价格预测系统 Gold Price Predictor           ║
    ║                      v1.0                                 ║
    ╚═══════════════════════════════════════════════════════════╝
    """)
    
    # 创建预测器
    predictor = GoldPricePredictor()
    
    # 1. 获取数据
    predictor.fetch_data(period_days=365)
    
    # 2. 创建特征
    predictor.create_features()
    
    # 3. 训练模型
    predictor.train_model()
    
    # 4. 分析因子
    predictor.analyze_factors()
    
    # 5. 预测明天价格
    prediction = predictor.predict_tomorrow()
    
    # 6. 生成图表
    predictor.plot_analysis()
    
    # 7. 生成报告
    predictor.generate_report(prediction)
    
    print("\n" + "=" * 60)
    print("分析完成!")
    print("=" * 60)
    print("\n生成的文件:")
    print("  - gold_analysis.png (分析图表)")
    print("  - gold_analysis_report.txt (分析报告)")
    
    return predictor, prediction


if __name__ == "__main__":
    predictor, prediction = main()
