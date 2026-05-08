import pandas as pd
import numpy as np
import logging
from datetime import datetime
import os

os.makedirs('logs', exist_ok=True)
os.makedirs('result', exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'logs/backtest_barra_alpha_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

srcdir = "E:/SJTU/实习/国泰海通/业绩回测/nav"
#expdir =  "E:/SJTU/实习/国泰海通/barra因子/result/管理人暴露/exposure" #excess_exposure
exrdir = "E:/SJTU/实习/国泰海通/barra因子/result/管理人暴露/weight_result/EXreturns"
desdir = "E:/SJTU/实习/国泰海通/业绩回测/result"

FACTOR_COLUMNS = [
    'fund_ret', 'pure_alpha', 'style_ret', 'beta', 'book_to_price', 'comovement', 
    'earnings_yield', 'growth', 'leverage', 'liquidity', 
    'momentum', 'non_linear_size', 'residual_volatility', 'size'
]

STYLE_FACTORS = [
    'beta', 'book_to_price', 'earnings_yield', 'growth', 
    'leverage', 'liquidity', 'momentum', 'non_linear_size', 'residual_volatility', 'size'
] #"comovement"

RF = 0.015

def load_data(filepath):
    try:
        df = pd.read_excel(filepath, index_col=0)
        df.index = pd.to_datetime(df.index)
        logger.info(f"成功加载数据，形状: {df.shape}")
        return df
    except Exception as e:
        logger.error(f"加载数据失败: {e}")
        raise

def calculate_drawdown(nav):
    max_value = nav.cummax()
    drawdown = (nav - max_value) / max_value
    return drawdown

def calculate_metrics(nav_series):
    if len(nav_series) < 2:
        return None
    
    returns = nav_series.pct_change().dropna()
    
    if len(returns) == 0:
        return None
    
    total_return = nav_series.iloc[-1] / nav_series.iloc[0] - 1
    weeks = len(nav_series)
    annualized_return = (1 + total_return) ** (52 / weeks) - 1
    
    volatility = returns.std() * np.sqrt(52)
    
    sharpe_ratio = (annualized_return - RF) / volatility if volatility != 0 else np.nan
    
    drawdown = calculate_drawdown(nav_series)
    max_drawdown = drawdown.min()
    
    calmar_ratio = (annualized_return - RF) / abs(max_drawdown) if max_drawdown != 0 else np.nan
    
    tracking_error = volatility
    information_ratio = annualized_return / tracking_error if tracking_error != 0 else np.nan
    
    return {
        '年化收益': annualized_return,
        '年化波动率': volatility,
        '夏普比率': sharpe_ratio,
        '最大回撤': max_drawdown,
        '卡玛比率': calmar_ratio,
        '跟踪误差': tracking_error,
        '信息比率': information_ratio
    }

def generate_nav_from_returns(returns_series):
    nav = (1 + returns_series).cumprod()
    nav = nav / nav.iloc[0]
    return nav

def process_fund(fund_data, benchmark_returns=None):
    fund_results = {}
    
    for factor in FACTOR_COLUMNS:
        if factor not in fund_data.columns:
            logger.warning(f"因子 {factor} 不在数据中")
            continue
        
        returns = fund_data[factor].dropna()
        if len(returns) == 0:
            continue
        
        nav = generate_nav_from_returns(returns)
        
        use_benchmark = (factor == 'fund_ret') and (benchmark_returns is not None)
        
        fund_results[factor] = {
            'nav': nav,
            'full_period': calculate_metrics(nav, benchmark_returns if use_benchmark else None)
        }
        
        years = sorted(nav.index.year.unique())
        
        for i, year in enumerate(years):
            if i == 0:
                year_nav = nav[nav.index.year == year]
            else:
                prev_year = years[i - 1]
                prev_last_date = nav[nav.index.year == prev_year].index[-1]
                curr_last_date = nav[nav.index.year == year].index[-1]
                year_nav = nav[(nav.index >= prev_last_date) & (nav.index <= curr_last_date)]
            
            year_metrics = calculate_metrics(year_nav, benchmark_returns if use_benchmark else None)
            fund_results[factor][f'year_{year}'] = year_metrics
    
    return fund_results


def main():
    logger.info("开始处理超额收益回测")
    
    import glob
    
    excess_files = glob.glob(f"{exrdir}/*_超额收益与因子贡献.xlsx")
    logger.info(f"发现 {len(excess_files)} 个超额收益文件")
    
    if len(excess_files) == 0:
        logger.warning("未找到任何超额收益文件")
        return
    
    all_years = set()
    
    # 使用局部变量存储所有需要处理的因子
    all_factors = STYLE_FACTORS.copy()
    factor_results = {factor: {} for factor in all_factors}
    
    for file_path in excess_files:
        fund_name = os.path.basename(file_path).replace('_超额收益与因子贡献.xlsx', '')
        logger.info(f"处理基金: {fund_name}")
        
        try:
            df = pd.read_excel(file_path, index_col=0)
            df.index = pd.to_datetime(df.index)
            
            # 计算额外的列
            df["style_excess_ret"] = df[STYLE_FACTORS].sum(axis=1)
            
            # 动态添加需要处理的因子
            current_factors = STYLE_FACTORS.copy()
            if 'excess_return' in df.columns:
                current_factors.append('excess_return')
            current_factors.append('style_excess_ret')
            
            all_years.update(df.index.year.unique())
            
            for factor in current_factors:
                if factor not in df.columns:
                    continue
                
                if factor not in factor_results:
                    factor_results[factor] = {}
                
                returns = df[factor].dropna()
                if len(returns) == 0:
                    continue
                
                nav = generate_nav_from_returns(returns)
                
                years = sorted(nav.index.year.unique())
                
                for i, year in enumerate(years):
                    if i == 0:
                        year_nav = nav[nav.index.year == year]
                    else:
                        prev_last_date = nav[nav.index.year == years[i-1]].index[-1]
                        curr_last_date = nav[nav.index.year == year].index[-1]
                        year_nav = nav[(nav.index >= prev_last_date) & (nav.index <= curr_last_date)]
                    
                    year_key = f'year_{year}'
                    if year_key not in factor_results[factor]:
                        factor_results[factor][year_key] = {}
                    metrics = calculate_metrics(year_nav)
                    if metrics:
                        factor_results[factor][year_key][fund_name] = metrics
                
                if 'full_period' not in factor_results[factor]:
                    factor_results[factor]['full_period'] = {}
                metrics = calculate_metrics(nav)
                if metrics:
                    factor_results[factor]['full_period'][fund_name] = metrics
        
        except Exception as e:
            logger.error(f"处理文件 {file_path} 失败: {e}")
    
    all_years = sorted(all_years)
    
    for factor in factor_results:
        has_data = False
        for period in ['full_period'] + [f'year_{y}' for y in all_years]:
            if factor_results[factor].get(period):
                has_data = True
                break
        
        if not has_data:
            continue
        
        output_file = f"{desdir}/中证500指增产品各收益回测/中证500指增_{factor}_backtest.xlsx"
        with pd.ExcelWriter(output_file) as writer:
            for period in ['full_period'] + [f'year_{y}' for y in all_years]:
                sheet_name = '成立以来' if period == 'full_period' else period.replace('year_', '')
                
                if factor_results[factor].get(period):
                    df = pd.DataFrame(factor_results[factor][period]).T
                    df.index = pd.to_numeric(df.index, errors='ignore')
                    df.index.name = 'fund'
                    df.to_excel(writer, sheet_name=sheet_name)
        
        logger.info(f"已保存因子 {factor} 的结果到 {output_file}")
    
    logger.info("处理完成")

if __name__ == "__main__":
    main()