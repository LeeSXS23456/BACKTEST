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
expdir = "E:/SJTU/实习/国泰海通/barra因子/result/管理人暴露/excess_exposure"
desdir = "E:/SJTU/实习/国泰海通/业绩回测/result"

FACTOR_COLUMNS = [
    'fund_ret', 'pure_alpha', 'style_ret', 'beta', 'book_to_price', 'comovement', 
    'earnings_yield', 'growth', 'leverage', 'liquidity', 
    'momentum', 'non_linear_size', 'residual_volatility', 'size'
]

STYLE_FACTORS = [
    'beta', 'book_to_price', 'comovement', 'earnings_yield', 'growth', 
    'leverage', 'liquidity', 'momentum', 'non_linear_size', 'residual_volatility', 'size'
]

RF = 0.015

def load_data(filepath):
    try:
        df = pd.read_excel(filepath, index_col=0)
        df.index = pd.to_datetime(df.index)
        logger.info(f"成功加载数据，形状: {df.shape}")
        logger.info(f"因子列表: {FACTOR_COLUMNS}")
        logger.info(f"基金数量: {df['fund'].nunique()}")
        return df
    except Exception as e:
        logger.error(f"加载数据失败: {e}")
        raise

def calculate_drawdown(nav):
    max_value = nav.cummax()
    drawdown = (nav - max_value) / max_value
    return drawdown

def calculate_metrics(nav_series, benchmark_returns=None):
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
    
    result = {
        '年化收益': annualized_return,
        '年化波动率': volatility,
        '夏普比率': sharpe_ratio,
        '最大回撤': max_drawdown,
        '卡玛比率': calmar_ratio
    }
    
    if benchmark_returns is not None:
        aligned_returns = returns.dropna()
        aligned_benchmark = benchmark_returns.dropna()
        
        common_dates = aligned_returns.index.intersection(aligned_benchmark.index)
        if len(common_dates) >= 2:
            excess_returns = aligned_returns.loc[common_dates] - aligned_benchmark.loc[common_dates]
            tracking_error = excess_returns.std() * np.sqrt(52)
            
            excess_return = (1 + excess_returns).prod() - 1
            excess_annualized = (1 + excess_return) ** (52 / len(common_dates)) - 1
            
            information_ratio = excess_annualized / tracking_error if tracking_error != 0 else np.nan
            
            result['跟踪误差'] = tracking_error
            result['信息比率'] = information_ratio
    
    return result

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

def process_exposure_data():
    logger.info("开始处理暴露数据")
    
    import glob
    
    exposure_files = glob.glob(f"{expdir}/*_relative_exposure.xlsx")
    logger.info(f"发现 {len(exposure_files)} 个暴露文件")
    
    if len(exposure_files) == 0:
        logger.warning("未找到任何暴露文件")
        return
    
    all_results = {'full_period': {}}
    all_years = set()
    
    for file_path in exposure_files:
        product_name = os.path.basename(file_path).replace('_relative_exposure.xlsx', '')
        logger.info(f"处理产品: {product_name}")
        
        try:
            df = pd.read_excel(file_path, index_col=0)
            df.index = pd.to_datetime(df.index)
            
            available_factors = [f for f in STYLE_FACTORS if f in df.columns]
            if not available_factors:
                logger.warning(f"产品 {product_name} 中没有找到风格因子列")
                continue
            
            abs_exposure = df[available_factors].abs()
            
            all_results['full_period'][product_name] = {}
            for factor in available_factors:
                all_results['full_period'][product_name][factor] = abs_exposure[factor].mean()
            all_results['full_period'][product_name]['总和'] = abs_exposure[available_factors].mean().sum()
            
            years = df.index.year.unique()
            all_years.update(years)
            
            for year in years:
                year_key = f'year_{year}'
                if year_key not in all_results:
                    all_results[year_key] = {}
                
                year_data = df[df.index.year == year]
                year_abs = year_data[available_factors].abs()
                
                all_results[year_key][product_name] = {}
                for factor in available_factors:
                    all_results[year_key][product_name][factor] = year_abs[factor].mean()
                all_results[year_key][product_name]['总和'] = year_abs[available_factors].mean().sum()
                
        except Exception as e:
            logger.error(f"处理文件 {file_path} 失败: {e}")
    
    all_years = sorted(all_years)
    
    output_file = f"{desdir}/exposure_summary.xlsx"
    
    with pd.ExcelWriter(output_file) as writer:
        for period in ['full_period'] + [f'year_{y}' for y in all_years]:
            if period == 'full_period':
                sheet_name = '成立以来'
            else:
                sheet_name = period.replace('year_', '')
            
            if all_results.get(period):
                df = pd.DataFrame(all_results[period]).T
                df.index.name = 'product'
                df = df[STYLE_FACTORS + ['总和']]
                df.to_excel(writer, sheet_name=sheet_name)
    
    logger.info(f"暴露数据汇总已保存到 {output_file}")

def main():
    logger.info("开始处理 Barra Alpha 回测")
    
    nav_file = f"{srcdir}/barra_alpha_ret.xlsx"
    nav_df = load_data(nav_file)
    
    all_funds = nav_df['fund'].unique()
    logger.info(f"共 {len(all_funds)} 个基金")
    
    all_years = sorted(nav_df.index.year.unique())
    logger.info(f"数据覆盖年份: {all_years}")
    
    benchmark_returns = None
    if '中证500' in all_funds and 'fund_ret' in nav_df.columns:
        benchmark_data = nav_df[nav_df['fund'] == '中证500'].copy()
        benchmark_data = benchmark_data.sort_index()
        benchmark_returns = benchmark_data['fund_ret']
        logger.info("已获取中证500作为基准")
    
    factor_results = {factor: {} for factor in FACTOR_COLUMNS}
    for factor in FACTOR_COLUMNS:
        for year in all_years:
            factor_results[factor][f'year_{year}'] = {}
        factor_results[factor]['full_period'] = {}
    
    for i, fund in enumerate(all_funds, 1):
        logger.info(f"处理基金 {i}/{len(all_funds)}: {fund}")
        
        fund_data = nav_df[nav_df['fund'] == fund].copy()
        fund_data = fund_data.sort_index()
        
        fund_results = process_fund(fund_data, benchmark_returns)
        
        for factor in FACTOR_COLUMNS:
            if factor not in fund_results:
                continue
            
            for period in ['full_period'] + [f'year_{y}' for y in all_years]:
                if period not in fund_results[factor]:
                    continue
                
                metrics = fund_results[factor][period]
                if metrics is None:
                    continue
                
                factor_results[factor][period][fund] = metrics
    
    # for factor in FACTOR_COLUMNS:
    #     has_data = False
    #     for period in ['full_period'] + [f'year_{y}' for y in all_years]:
    #         if factor_results[factor][period]:
    #             has_data = True
    #             break
        
    #     if not has_data:
    #         logger.warning(f"因子 {factor} 没有有效数据，跳过")
    #         continue
        
    #     output_file = f"{desdir}/中证500指增_{factor}_backtest.xlsx"
    #     with pd.ExcelWriter(output_file) as writer:
    #         for period in ['full_period'] + [f'year_{y}' for y in all_years]:
    #             if period == 'full_period':
    #                 sheet_name = '成立以来'
    #             else:
    #                 sheet_name = period.replace('year_', '')
                
    #             if factor_results[factor][period]:
    #                 df = pd.DataFrame(factor_results[factor][period]).T
    #                 df.index.name = 'fund'
    #                 df.to_excel(writer, sheet_name=sheet_name)
        
    #     logger.info(f"已保存因子 {factor} 的结果到 {output_file}")
    
    # logger.info("处理完成")
    
    process_exposure_data()

if __name__ == "__main__":
    main()