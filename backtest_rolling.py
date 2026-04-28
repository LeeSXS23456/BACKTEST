import pandas as pd
import numpy as np
import logging
from datetime import datetime
import os

os.makedirs('logs', exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'logs/backtest_rolling_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

srcdir = "E:/SJTU/实习/国泰海通/业绩回测/nav"
resdir = "E:/SJTU/实习/国泰海通/业绩回测/result"

def load_nav_data(filepath):
    try:
        df = pd.read_excel(filepath, index_col=0)
        logger.info(f"成功加载数据，形状: {df.shape}")
        logger.info(f"指数列表: {df.columns.tolist()}")
        return df
    except Exception as e:
        logger.error(f"加载数据失败: {e}")
        raise

def calculate_drawdown(nav):
    max_value = nav.cummax()
    drawdown = (nav - max_value) / max_value
    return drawdown

def calculate_backtest_metrics(nav_series):
    rf = 0.015
    if len(nav_series) < 2:
        return None
    
    returns = nav_series.pct_change().dropna()
    
    if len(returns) == 0:
        return None
    
    total_return = nav_series.iloc[-1] / nav_series.iloc[0] - 1
    days = len(nav_series)
    annualized_return = (1 + total_return) ** (240 / days) - 1
    
    volatility = returns.std() * np.sqrt(240)
    
    sharpe_ratio = (annualized_return - rf) / volatility if volatility != 0 else np.nan
    
    drawdown = calculate_drawdown(nav_series)
    max_drawdown = drawdown.min()
    
    calmar_ratio = (annualized_return - rf) / abs(max_drawdown) if max_drawdown != 0 else np.nan
    
    win_rate = (returns > 0).mean()
    
    return {
        '年化收益': annualized_return,
        '累计收益': total_return,
        '年化波动率': volatility,
        '夏普比率': sharpe_ratio,
        '最大回撤': max_drawdown,
        '卡玛比率': calmar_ratio,
        '胜率': win_rate
    }

def rolling_backtest(nav_df, window=240*3):
    results_dict = {col: [] for col in nav_df.columns}
    total_days = len(nav_df)
    
    for i in range(total_days - window + 1):
        start_idx = i
        end_idx = i + window
        
        start_date = nav_df.index[start_idx]
        end_date = nav_df.index[end_idx - 1]
        
        for col in nav_df.columns:
            nav_series = nav_df[col].iloc[start_idx:end_idx]
            metrics = calculate_backtest_metrics(nav_series)
            
            if metrics is not None:
                metrics['持有开始日期'] = start_date
                metrics['持有结束日期'] = end_date
                results_dict[col].append(metrics)
        
        if (i + 1) % 50 == 0:
            logger.info(f"已处理 {i + 1}/{total_days - window + 1} 个窗口")
    
    for col in results_dict:
        df = pd.DataFrame(results_dict[col])
        df.set_index('持有结束日期', inplace=True)
        df = df[['持有开始日期', '累计收益', '年化收益', '年化波动率', '夏普比率', '最大回撤', '卡玛比率', '胜率']]
        results_dict[col] = df
    
    return results_dict

def aggregate_results(results_dict):
    agg_results = {}
    
    for index_name, df in results_dict.items():
        agg_results[index_name] = {}
        
        for metric_name in ['累计收益', '年化收益', '年化波动率', '夏普比率', '最大回撤', '卡玛比率', '胜率']:
            values = df[metric_name].dropna()
            
            if len(values) == 0:
                agg_results[index_name][metric_name] = np.nan
                continue
            
            if metric_name == '最大回撤':
                agg_results[index_name][f'{metric_name}_平均'] = values.mean()
                agg_results[index_name][f'{metric_name}_最小'] = values.min()
            else:
                agg_results[index_name][metric_name] = values.mean()
        
        positive_windows = (df['年化收益'] > 0).sum()
        total_windows = len(df['年化收益'].dropna())
        agg_results[index_name]['总胜率'] = positive_windows / total_windows if total_windows > 0 else np.nan
    
    agg_df = pd.DataFrame(agg_results).T
    return agg_df

def main():
    logger.info("开始回测流程")
    
    nav_file = f"{srcdir}/nav_index.xlsx"
    nav_df = load_nav_data(nav_file)
    
    window = 240 * 3
    logger.info(f"回测窗口: {window} 个交易日")
    
    results_dict = rolling_backtest(nav_df, window)
    logger.info(f"滚动回测完成，共 {len(results_dict)} 个指数")
    
    agg_df = aggregate_results(results_dict)
    logger.info("汇总计算完成")
    
    output_file = f"{resdir}/backtest_results_{nav_df.columns.tolist()}.xlsx"
    with pd.ExcelWriter(output_file) as writer:
        agg_df.to_excel(writer, sheet_name='汇总结果')
        
        for index_name, df in results_dict.items():
            sheet_name = index_name[:31]
            df.to_excel(writer, sheet_name=sheet_name)
            logger.info(f"已写入 {sheet_name} 数据")
    
    logger.info(f"结果已保存到 {output_file}")
    logger.info("回测流程结束")

if __name__ == "__main__":
    import os
    os.makedirs('logs', exist_ok=True)
    main()