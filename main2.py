"""
QuantConnect OHLC Correlation Analysis Framework  
Purpose: Multi-asset correlation analysis using OHLC data with Rogers-Zhou estimators
ENHANCED: Parseable log output for data extraction without Object Store
"""

from AlgorithmImports import *
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from scipy import stats
from collections import defaultdict
import json

class OHLCCorrelationAnalyzer:
    """Correlation analyzer using OHLC data with Rogers-Zhou estimators"""
    
    def __init__(self, algorithm):
        self.algorithm = algorithm
        self.symbols = ['TSLA', 'SPY', 'QQQ', 'VIX']
        self.data = pd.DataFrame()
        self.correlation_results = {}
        self.win_loss_results = {}
        self.summary_dataframes = {}
        self.weekly_summaries = []
        self.all_time_stats = defaultdict(list)
        
    def create_summary_dataframes(self, ohlc_data, returns_data, correlation_results, stats_results):
        """Create summary DataFrames for structured output display"""
        
        # OHLC Summary DataFrame
        ohlc_summary_data = {
            'ticker': [], 'date': [], 'open': [], 'high': [], 'low': [], 'close': [],
            'volume': [], 'daily_diff': [], 'daily_return_pct': [], 'intraday_range_pct': []
        }
        
        if not ohlc_data.empty:
            latest_date = ohlc_data.index[-1]
            formatted_date = int(latest_date.strftime('%Y%m%d'))
            
            for symbol in self.symbols:
                if f'{symbol}_close' in ohlc_data.columns:
                    try:
                        open_price = ohlc_data[f'{symbol}_open'].iloc[-1]
                        high_price = ohlc_data[f'{symbol}_high'].iloc[-1]
                        low_price = ohlc_data[f'{symbol}_low'].iloc[-1]
                        close_price = ohlc_data[f'{symbol}_close'].iloc[-1]
                        volume = ohlc_data.get(f'{symbol}_volume', pd.Series([0])).iloc[-1]
                        
                        daily_diff = close_price - open_price
                        daily_return_pct = (daily_diff / open_price) * 100 if open_price > 0 else 0
                        intraday_range_pct = ((high_price - low_price) / close_price) * 100 if close_price > 0 else 0
                        
                        ohlc_summary_data['ticker'].append(symbol)
                        ohlc_summary_data['date'].append(formatted_date)
                        ohlc_summary_data['open'].append(round(open_price, 4))
                        ohlc_summary_data['high'].append(round(high_price, 4))
                        ohlc_summary_data['low'].append(round(low_price, 4))
                        ohlc_summary_data['close'].append(round(close_price, 4))
                        ohlc_summary_data['volume'].append(int(volume))
                        ohlc_summary_data['daily_diff'].append(round(daily_diff, 4))
                        ohlc_summary_data['daily_return_pct'].append(round(daily_return_pct, 4))
                        ohlc_summary_data['intraday_range_pct'].append(round(intraday_range_pct, 4))
                    except (IndexError, KeyError):
                        continue
        
        self.summary_dataframes['ohlc_summary'] = pd.DataFrame(ohlc_summary_data)
        
        # Correlation Matrix DataFrame
        correlation_matrix_data = {
            'asset_pair': [], 'pearson_returns': [], 'spearman_returns': [],
            'rogers_zhou_basic': [], 'rogers_zhou_weighted': [], 'garman_klass': [], 'correlation_spread': []
        }
        
        for pair, results in correlation_results.items():
            correlation_matrix_data['asset_pair'].append(pair)
            
            pearson = results.get('pearson_returns', np.nan)
            spearman = results.get('spearman_returns', np.nan)
            rz_basic = results.get('rz_basic', np.nan)
            rz_weighted = results.get('rz_weighted', np.nan)
            garman_klass = results.get('garman_klass', np.nan)
            
            correlation_matrix_data['pearson_returns'].append(round(pearson, 4) if not np.isnan(pearson) else None)
            correlation_matrix_data['spearman_returns'].append(round(spearman, 4) if not np.isnan(spearman) else None)
            correlation_matrix_data['rogers_zhou_basic'].append(round(rz_basic, 4) if not np.isnan(rz_basic) else None)
            correlation_matrix_data['rogers_zhou_weighted'].append(round(rz_weighted, 4) if not np.isnan(rz_weighted) else None)
            correlation_matrix_data['garman_klass'].append(round(garman_klass, 4) if not np.isnan(garman_klass) else None)
            
            if not (np.isnan(pearson) or np.isnan(rz_basic)):
                spread = rz_basic - pearson
                correlation_matrix_data['correlation_spread'].append(round(spread, 4))
            else:
                correlation_matrix_data['correlation_spread'].append(None)
        
        self.summary_dataframes['correlation_matrix'] = pd.DataFrame(correlation_matrix_data)
        
        # Statistical Summary DataFrame
        stats_summary_data = {
            'ticker': [], 'mean_daily_return': [], 'volatility_daily': [], 'sharpe_ratio': [],
            'skewness': [], 'kurtosis': [], 'var_95_pct': [], 'var_99_pct': [], 'max_return': [], 'min_return': []
        }
        
        for symbol, stats_data in stats_results.items():
            stats_summary_data['ticker'].append(symbol)
            stats_summary_data['mean_daily_return'].append(round(stats_data['mean'], 6))
            stats_summary_data['volatility_daily'].append(round(stats_data['std'], 6))
            stats_summary_data['sharpe_ratio'].append(round(stats_data['sharpe_ratio'], 4))
            stats_summary_data['skewness'].append(round(stats_data['skewness'], 4))
            stats_summary_data['kurtosis'].append(round(stats_data['kurtosis'], 4))
            stats_summary_data['var_95_pct'].append(round(stats_data['var_95'], 6))
            stats_summary_data['var_99_pct'].append(round(stats_data['var_99'], 6))
            stats_summary_data['max_return'].append(round(stats_data['max'], 6))
            stats_summary_data['min_return'].append(round(stats_data['min'], 6))
        
        self.summary_dataframes['stats_summary'] = pd.DataFrame(stats_summary_data)
        
        # Win/Loss Analysis DataFrame
        win_loss_summary_data = {
            'ticker': [], 'total_observations': [], 'wins_vix_up': [], 'losses_vix_up': [],
            'wins_vix_down': [], 'losses_vix_down': [], 'win_rate_vix_up': [], 'win_rate_vix_down': [], 'overall_win_rate': []
        }
        
        win_loss_df = self.classify_win_loss_days(returns_data)
        
        if win_loss_df is not None and not win_loss_df.empty:
            for asset in ['TSLA', 'SPY', 'QQQ']:
                win_vix_up_col = f'{asset}_win_vix_up'
                loss_vix_up_col = f'{asset}_loss_vix_up'
                win_vix_down_col = f'{asset}_win_vix_down'
                loss_vix_down_col = f'{asset}_loss_vix_down'
                
                if all(col in win_loss_df.columns for col in [win_vix_up_col, loss_vix_up_col, win_vix_down_col, loss_vix_down_col]):
                    wins_vix_up = int(win_loss_df[win_vix_up_col].sum())
                    losses_vix_up = int(win_loss_df[loss_vix_up_col].sum())
                    wins_vix_down = int(win_loss_df[win_vix_down_col].sum())
                    losses_vix_down = int(win_loss_df[loss_vix_down_col].sum())
                    
                    total_obs = wins_vix_up + losses_vix_up + wins_vix_down + losses_vix_down
                    total_wins = wins_vix_up + wins_vix_down
                    
                    win_rate_vix_up = wins_vix_up / (wins_vix_up + losses_vix_up) if (wins_vix_up + losses_vix_up) > 0 else 0
                    win_rate_vix_down = wins_vix_down / (wins_vix_down + losses_vix_down) if (wins_vix_down + losses_vix_down) > 0 else 0
                    overall_win_rate = total_wins / total_obs if total_obs > 0 else 0
                    
                    win_loss_summary_data['ticker'].append(asset)
                    win_loss_summary_data['total_observations'].append(total_obs)
                    win_loss_summary_data['wins_vix_up'].append(wins_vix_up)
                    win_loss_summary_data['losses_vix_up'].append(losses_vix_up)
                    win_loss_summary_data['wins_vix_down'].append(wins_vix_down)
                    win_loss_summary_data['losses_vix_down'].append(losses_vix_down)
                    win_loss_summary_data['win_rate_vix_up'].append(round(win_rate_vix_up, 4))
                    win_loss_summary_data['win_rate_vix_down'].append(round(win_rate_vix_down, 4))
                    win_loss_summary_data['overall_win_rate'].append(round(overall_win_rate, 4))
        
        self.summary_dataframes['win_loss_summary'] = pd.DataFrame(win_loss_summary_data)
        self.win_loss_results = win_loss_df
    
    def log_parseable_weekly_summary(self, week_year_str):
        """Log weekly summary in parseable CSV/JSON format"""
        
        self.algorithm.log("\n" + "="*80)
        self.algorithm.log(f"===WEEKLY_SUMMARY_START_{week_year_str}===")
        
        # OHLC Summary as CSV
        if 'ohlc_summary' in self.summary_dataframes and not self.summary_dataframes['ohlc_summary'].empty:
            self.algorithm.log("CSV_OHLC_SUMMARY:")
            df = self.summary_dataframes['ohlc_summary']
            self.algorithm.log(','.join(df.columns.tolist()))
            for _, row in df.iterrows():
                row_str = ','.join([str(val) if val is not None else '' for val in row.values])
                self.algorithm.log(row_str)
        
        # Correlation Matrix as CSV
        if 'correlation_matrix' in self.summary_dataframes and not self.summary_dataframes['correlation_matrix'].empty:
            self.algorithm.log("CSV_CORRELATION_MATRIX:")
            df = self.summary_dataframes['correlation_matrix']
            self.algorithm.log(','.join(df.columns.tolist()))
            for _, row in df.iterrows():
                row_str = ','.join([str(val) if val is not None else '' for val in row.values])
                self.algorithm.log(row_str)
        
        # Statistics Summary as CSV
        if 'stats_summary' in self.summary_dataframes and not self.summary_dataframes['stats_summary'].empty:
            self.algorithm.log("CSV_STATS_SUMMARY:")
            df = self.summary_dataframes['stats_summary']
            self.algorithm.log(','.join(df.columns.tolist()))
            for _, row in df.iterrows():
                row_str = ','.join([str(val) if val is not None else '' for val in row.values])
                self.algorithm.log(row_str)
        
        # Win/Loss Summary as CSV
        if 'win_loss_summary' in self.summary_dataframes and not self.summary_dataframes['win_loss_summary'].empty:
            self.algorithm.log("CSV_WIN_LOSS_SUMMARY:")
            df = self.summary_dataframes['win_loss_summary']
            self.algorithm.log(','.join(df.columns.tolist()))
            for _, row in df.iterrows():
                row_str = ','.join([str(val) if val is not None else '' for val in row.values])
                self.algorithm.log(row_str)
        
        # Week Summary JSON
        week_summary_json = {
            'week': week_year_str,
            'data_points': len(self.summary_dataframes.get('ohlc_summary', pd.DataFrame())),
            'avg_returns': {}, 'correlations': {}, 'win_rates': {}
        }
        
        if 'stats_summary' in self.summary_dataframes:
            for _, row in self.summary_dataframes['stats_summary'].iterrows():
                week_summary_json['avg_returns'][row['ticker']] = row['mean_daily_return']
        
        if 'correlation_matrix' in self.summary_dataframes:
            for _, row in self.summary_dataframes['correlation_matrix'].iterrows():
                if row['pearson_returns'] is not None:
                    week_summary_json['correlations'][row['asset_pair']] = row['pearson_returns']
        
        if 'win_loss_summary' in self.summary_dataframes:
            for _, row in self.summary_dataframes['win_loss_summary'].iterrows():
                week_summary_json['win_rates'][row['ticker']] = row['overall_win_rate']
        
        self.algorithm.log("JSON_WEEKLY_SUMMARY:")
        self.algorithm.log(json.dumps(week_summary_json))
        
        self.weekly_summaries.append(week_summary_json)
        
        if 'stats_summary' in self.summary_dataframes:
            for _, row in self.summary_dataframes['stats_summary'].iterrows():
                ticker = row['ticker']
                self.all_time_stats[ticker].append({
                    'week': week_year_str,
                    'mean_return': row['mean_daily_return'],
                    'volatility': row['volatility_daily'],
                    'sharpe': row['sharpe_ratio']
                })
        
        self.algorithm.log(f"===WEEKLY_SUMMARY_END_{week_year_str}===")
        self.algorithm.log("="*80 + "\n")
    
    def log_comprehensive_final_summary(self):
        """Log final comprehensive summary across all weeks in parseable format"""
        
        self.algorithm.log("\n" + "="*100)
        self.algorithm.log("===FINAL_COMPREHENSIVE_SUMMARY_START===")
        
        # Overall Performance Summary CSV
        self.algorithm.log("CSV_OVERALL_PERFORMANCE:")
        self.algorithm.log("ticker,total_weeks,avg_weekly_return,avg_weekly_volatility,avg_weekly_sharpe,min_weekly_return,max_weekly_return,return_consistency")
        
        for ticker, stats_list in self.all_time_stats.items():
            if stats_list:
                returns = [s['mean_return'] for s in stats_list]
                volatilities = [s['volatility'] for s in stats_list]
                sharpes = [s['sharpe'] for s in stats_list]
                
                avg_return = np.mean(returns)
                avg_vol = np.mean(volatilities)
                avg_sharpe = np.mean(sharpes)
                min_return = np.min(returns)
                max_return = np.max(returns)
                return_std = np.std(returns)
                consistency = 1 / (1 + return_std) if return_std > 0 else 1
                
                row_data = [ticker, len(stats_list), round(avg_return, 6), round(avg_vol, 6),
                           round(avg_sharpe, 4), round(min_return, 6), round(max_return, 6), round(consistency, 4)]
                self.algorithm.log(','.join([str(val) for val in row_data]))
        
        # Weekly Progression CSV
        self.algorithm.log("CSV_WEEKLY_PROGRESSION:")
        self.algorithm.log("week,data_points,total_correlations,avg_correlation,max_correlation,min_correlation")
        
        for week_data in self.weekly_summaries:
            correlations = list(week_data.get('correlations', {}).values())
            if correlations:
                avg_corr = np.mean(correlations)
                max_corr = np.max(correlations)
                min_corr = np.min(correlations)
            else:
                avg_corr = max_corr = min_corr = 0
            
            row_data = [week_data['week'], week_data.get('data_points', 0), len(correlations),
                       round(avg_corr, 4), round(max_corr, 4), round(min_corr, 4)]
            self.algorithm.log(','.join([str(val) for val in row_data]))
        
        # Final Statistics JSON
        final_stats = {
            'analysis_period': {
                'total_weeks': len(self.weekly_summaries),
                'start_week': self.weekly_summaries[0]['week'] if self.weekly_summaries else None,
                'end_week': self.weekly_summaries[-1]['week'] if self.weekly_summaries else None
            },
            'best_performers': {}, 'correlation_insights': {}, 'trading_signals': {}
        }
        
        # Find best performers
        for ticker, stats_list in self.all_time_stats.items():
            if stats_list:
                avg_sharpe = np.mean([s['sharpe'] for s in stats_list])
                final_stats['best_performers'][ticker] = {
                    'avg_sharpe': round(avg_sharpe, 4),
                    'weeks_analyzed': len(stats_list)
                }
        
        # Correlation insights
        if self.weekly_summaries:
            all_correlations = {}
            for week in self.weekly_summaries:
                for pair, corr in week.get('correlations', {}).items():
                    if pair not in all_correlations:
                        all_correlations[pair] = []
                    all_correlations[pair].append(corr)
            
            for pair, corrs in all_correlations.items():
                final_stats['correlation_insights'][pair] = {
                    'avg_correlation': round(np.mean(corrs), 4),
                    'correlation_stability': round(1 - np.std(corrs), 4),
                    'trend': 'increasing' if corrs[-1] > corrs[0] else 'decreasing'
                }
        
        self.algorithm.log("JSON_FINAL_STATISTICS:")
        self.algorithm.log(json.dumps(final_stats, indent=2))
        
        # Trading Signal Recommendations
        self.algorithm.log("TRADING_SIGNAL_RECOMMENDATIONS:")
        
        signals = []
        
        # Correlation persistence signals
        if 'correlation_insights' in final_stats:
            for pair, data in final_stats['correlation_insights'].items():
                if data['correlation_stability'] > 0.7 and abs(data['avg_correlation']) > 0.3:
                    signal_strength = min(data['correlation_stability'] * abs(data['avg_correlation']), 1.0)
                    signals.append({
                        'type': 'correlation_persistence', 'pair': pair, 'strength': round(signal_strength, 4),
                        'direction': 'positive' if data['avg_correlation'] > 0 else 'negative'
                    })
        
        # Volatility-based signals
        for ticker, perf_data in final_stats.get('best_performers', {}).items():
            if perf_data['avg_sharpe'] > 0.5:
                signals.append({
                    'type': 'momentum', 'ticker': ticker, 'strength': min(perf_data['avg_sharpe'] / 2.0, 1.0), 'direction': 'bullish'
                })
        
        # Output signals as CSV
        self.algorithm.log("CSV_TRADING_SIGNALS:")
        self.algorithm.log("signal_type,asset,strength,direction,recommendation")
        for signal in signals:
            if signal['type'] == 'correlation_persistence':
                recommendation = f"Monitor {signal['pair']} for correlation breaks"
                self.algorithm.log(f"{signal['type']},{signal['pair']},{signal['strength']},{signal['direction']},{recommendation}")
            else:
                recommendation = f"Consider {signal['direction']} position in {signal['ticker']}"
                asset = signal.get('ticker', signal.get('pair', 'unknown'))
                self.algorithm.log(f"{signal['type']},{asset},{signal['strength']},{signal['direction']},{recommendation}")
        
        self.algorithm.log("===FINAL_COMPREHENSIVE_SUMMARY_END===")
        self.algorithm.log("="*100 + "\n")
        
        self.algorithm.log("EXTRACTION SUMMARY:")
        self.algorithm.log(f"Total weekly summaries: {len(self.weekly_summaries)}")
        self.algorithm.log(f"Assets analyzed: {len(self.all_time_stats)}")
        self.algorithm.log(f"Trading signals generated: {len(signals)}")
        self.algorithm.log("All data is in parseable CSV/JSON format above")
        self.algorithm.log("Look for ===WEEKLY_SUMMARY_START_*=== and ===FINAL_COMPREHENSIVE_SUMMARY_START=== markers")
    
    def rogers_zhou_estimator(self, df1, df2, method='rz_basic'):
        """Implement Rogers-Zhou OHLC-based correlation estimators"""
        
        try:
            if method == 'rz_basic':
                rv1 = np.log(df1['high'] / df1['close']) + np.log(df1['low'] / df1['close'])
                rv2 = np.log(df2['high'] / df2['close']) + np.log(df2['low'] / df2['close'])
                
                overnight1 = np.log(df1['open'] / df1['close'].shift(1))
                overnight2 = np.log(df2['open'] / df2['close'].shift(1))
                
                combined1 = overnight1 + rv1
                combined2 = overnight2 + rv2
                
                valid_idx = ~(np.isnan(combined1) | np.isnan(combined2))
                if valid_idx.sum() < 2:
                    return np.nan
                
                return np.corrcoef(combined1[valid_idx], combined2[valid_idx])[0, 1]
                
            elif method == 'rz_weighted':
                gk_vol1 = self._garman_klass_volatility(df1)
                gk_vol2 = self._garman_klass_volatility(df2)
                
                weights1 = 1 / (gk_vol1 + 1e-8)
                weights2 = 1 / (gk_vol2 + 1e-8)
                
                rv1 = np.log(df1['high'] / df1['close']) + np.log(df1['low'] / df1['close'])
                rv2 = np.log(df2['high'] / df2['close']) + np.log(df2['low'] / df2['close'])
                
                return self._weighted_correlation(rv1, rv2, weights1 * weights2)
                
            elif method == 'garman_klass':
                gk1 = self._garman_klass_volatility(df1)
                gk2 = self._garman_klass_volatility(df2)
                
                valid_idx = ~(np.isnan(gk1) | np.isnan(gk2))
                if valid_idx.sum() < 2:
                    return np.nan
                
                return np.corrcoef(gk1[valid_idx], gk2[valid_idx])[0, 1]
        
        except Exception as e:
            return np.nan
    
    def _garman_klass_volatility(self, df):
        """Calculate Garman-Klass volatility estimator"""
        return (np.log(df['high'] / df['low']) ** 2 / 2 - 
                (2 * np.log(2) - 1) * np.log(df['close'] / df['open']) ** 2)
    
    def _weighted_correlation(self, x, y, weights):
        """Calculate weighted correlation coefficient"""
        valid_idx = ~(np.isnan(x) | np.isnan(y) | np.isnan(weights))
        x_clean = x[valid_idx]
        y_clean = y[valid_idx]
        w_clean = weights[valid_idx]
        
        if len(x_clean) < 2:
            return np.nan
        
        w_sum = w_clean.sum()
        x_mean = (x_clean * w_clean).sum() / w_sum
        y_mean = (y_clean * w_clean).sum() / w_sum
        
        cov = ((x_clean - x_mean) * (y_clean - y_mean) * w_clean).sum() / w_sum
        var_x = ((x_clean - x_mean) ** 2 * w_clean).sum() / w_sum
        var_y = ((y_clean - y_mean) ** 2 * w_clean).sum() / w_sum
        
        return cov / np.sqrt(var_x * var_y)
    
    def calculate_returns(self, ohlc_data):
        """Calculate various return measures"""
        returns_dict = {}
        
        for symbol in self.symbols:
            symbol_cols = [col for col in ohlc_data.columns if col.startswith(symbol)]
            if len(symbol_cols) >= 4:
                
                opens = ohlc_data[f'{symbol}_open']
                highs = ohlc_data[f'{symbol}_high'] 
                lows = ohlc_data[f'{symbol}_low']
                closes = ohlc_data[f'{symbol}_close']
                
                returns_dict[f'{symbol}_log_return'] = np.log(closes / closes.shift(1))
                returns_dict[f'{symbol}_overnight'] = np.log(opens / closes.shift(1))
                returns_dict[f'{symbol}_intraday'] = np.log(closes / opens)
                returns_dict[f'{symbol}_range'] = (highs - lows) / closes
                
                prev_close = closes.shift(1)
                true_ranges = []
                for i in range(len(highs)):
                    if i == 0:
                        true_ranges.append((highs.iloc[i] - lows.iloc[i]) / closes.iloc[i])
                    else:
                        tr = max(
                            highs.iloc[i] - lows.iloc[i],
                            abs(highs.iloc[i] - prev_close.iloc[i]),
                            abs(lows.iloc[i] - prev_close.iloc[i])
                        ) / closes.iloc[i]
                        true_ranges.append(tr)
                
                returns_dict[f'{symbol}_true_range'] = pd.Series(true_ranges, index=closes.index)
        
        return pd.DataFrame(returns_dict, index=ohlc_data.index)
    
    def compute_correlations(self, ohlc_data, returns_data):
        """Compute multiple correlation estimates"""
        correlation_results = {}
        
        pairs = [('TSLA', 'SPY'), ('TSLA', 'QQQ'), ('TSLA', 'VIX'), ('QQQ', 'VIX')]
        
        for asset1, asset2 in pairs:
            pair_name = f"{asset1}_{asset2}"
            correlation_results[pair_name] = {}
            
            asset1_cols = [col for col in ohlc_data.columns if col.startswith(asset1)]
            asset2_cols = [col for col in ohlc_data.columns if col.startswith(asset2)]
            
            if len(asset1_cols) >= 4 and len(asset2_cols) >= 4:
                
                asset1_ohlc = pd.DataFrame({
                    'open': ohlc_data[f'{asset1}_open'], 'high': ohlc_data[f'{asset1}_high'],
                    'low': ohlc_data[f'{asset1}_low'], 'close': ohlc_data[f'{asset1}_close']
                })
                
                asset2_ohlc = pd.DataFrame({
                    'open': ohlc_data[f'{asset2}_open'], 'high': ohlc_data[f'{asset2}_high'],
                    'low': ohlc_data[f'{asset2}_low'], 'close': ohlc_data[f'{asset2}_close']
                })
                
                ret1_col = f'{asset1}_log_return'
                ret2_col = f'{asset2}_log_return'
                
                if ret1_col in returns_data.columns and ret2_col in returns_data.columns:
                    ret1 = returns_data[ret1_col].dropna()
                    ret2 = returns_data[ret2_col].dropna()
                    
                    common_idx = ret1.index.intersection(ret2.index)
                    ret1_aligned = ret1[common_idx]
                    ret2_aligned = ret2[common_idx]
                    
                    if len(ret1_aligned) > 10:
                        correlation_results[pair_name]['pearson_returns'] = ret1_aligned.corr(ret2_aligned)
                        correlation_results[pair_name]['spearman_returns'] = ret1_aligned.corr(ret2_aligned, method='spearman')
                
                correlation_results[pair_name]['rz_basic'] = self.rogers_zhou_estimator(asset1_ohlc, asset2_ohlc, 'rz_basic')
                correlation_results[pair_name]['rz_weighted'] = self.rogers_zhou_estimator(asset1_ohlc, asset2_ohlc, 'rz_weighted')
                correlation_results[pair_name]['garman_klass'] = self.rogers_zhou_estimator(asset1_ohlc, asset2_ohlc, 'garman_klass')
                
                if ret1_col in returns_data.columns and ret2_col in returns_data.columns:
                    ret1 = returns_data[ret1_col]
                    ret2 = returns_data[ret2_col]
                    correlation_results[pair_name]['rolling_30d'] = ret1.rolling(30).corr(ret2)
                    correlation_results[pair_name]['rolling_90d'] = ret1.rolling(90).corr(ret2)
        
        return correlation_results
    
    def classify_win_loss_days(self, returns_data):
        """Classify win/loss days based on VIX conditions"""
        win_loss_dict = {}
        
        for col in returns_data.columns:
            win_loss_dict[col] = returns_data[col]
        
        if 'VIX_log_return' in returns_data.columns:
            vix_returns = returns_data['VIX_log_return']
            vix_up = vix_returns > 0
            vix_down = vix_returns <= 0
            
            for asset in ['TSLA', 'SPY', 'QQQ']:
                asset_ret_col = f'{asset}_log_return'
                if asset_ret_col in returns_data.columns:
                    asset_up = returns_data[asset_ret_col] > 0
                    
                    win_loss_dict[f'{asset}_win_vix_up'] = vix_up & asset_up
                    win_loss_dict[f'{asset}_loss_vix_up'] = vix_up & ~asset_up
                    win_loss_dict[f'{asset}_win_vix_down'] = vix_down & asset_up
                    win_loss_dict[f'{asset}_loss_vix_down'] = vix_down & ~asset_up
                    
                    win_series = pd.Series(win_loss_dict[f'{asset}_win_vix_up'])
                    loss_series = pd.Series(win_loss_dict[f'{asset}_loss_vix_up'])
                    
                    win_loss_dict[f'{asset}_cumulative_wins'] = win_series.cumsum()
                    win_loss_dict[f'{asset}_cumulative_losses'] = loss_series.cumsum()
        
        return pd.DataFrame(win_loss_dict, index=returns_data.index)
    
    def calculate_statistics(self, returns_data):
        """Calculating descriptive statistics"""
        stats_dict = {}
        
        for symbol in self.symbols:
            returns_col = f'{symbol}_log_return'
            if returns_col in returns_data.columns:
                returns_series = returns_data[returns_col].dropna()
                
                if len(returns_series) > 5:
                    stats_dict[symbol] = {
                        'mean': returns_series.mean(),
                        'median': returns_series.median(), 
                        'std': returns_series.std(),
                        'skewness': returns_series.skew(),
                        'kurtosis': returns_series.kurtosis(),
                        'sharpe_ratio': returns_series.mean() / returns_series.std() * np.sqrt(252) if returns_series.std() > 0 else 0,
                        'min': returns_series.min(),
                        'max': returns_series.max(),
                        'var_95': returns_series.quantile(0.05),
                        'var_99': returns_series.quantile(0.01)
                    }
        
        return stats_dict


class QCOHLCCorrelationAlgorithm(QCAlgorithm):
    """QuantConnect Algorithm for OHLC Correlation Analysis"""
    
    def Initialize(self):
        
        self.set_start_date(2025, 1, 1)
        self.set_end_date(2025, 8, 26)
        self.set_cash(100000)
        
        self.symbols_map = {}
        
        tsla = self.add_equity('TSLA', Resolution.DAILY)
        tsla.set_data_normalization_mode(DataNormalizationMode.ADJUSTED)
        self.symbols_map['TSLA'] = tsla.symbol
        
        spy = self.add_equity('SPY', Resolution.DAILY) 
        spy.set_data_normalization_mode(DataNormalizationMode.ADJUSTED)
        self.symbols_map['SPY'] = spy.symbol
        
        qqq = self.add_equity('QQQ', Resolution.DAILY)
        qqq.set_data_normalization_mode(DataNormalizationMode.ADJUSTED)
        self.symbols_map['QQQ'] = qqq.symbol
        
        vix = self.add_data(VIXData, 'VIX', Resolution.DAILY)
        self.symbols_map['VIX'] = vix.symbol
        
        self.analyzer = OHLCCorrelationAnalyzer(self)
        
        self.ohlc_history = defaultdict(dict)
        self.collection_complete = False
        
        self.last_analysis_week = 0
        self.min_data_points = 50
        
        self.symbols = ['TSLA', 'SPY', 'QQQ', 'VIX']
        
    def on_data(self, data):
        """Collect OHLC data"""
        
        current_date = self.time.date()
        
        for name, symbol in self.symbols_map.items():
            if symbol in data and data[symbol] is not None:
                bar = data[symbol]
                
                self.ohlc_history[current_date].update({
                    f'{name}_open': float(bar.open),
                    f'{name}_high': float(bar.high),
                    f'{name}_low': float(bar.low),
                    f'{name}_close': float(bar.close),
                    f'{name}_volume': float(bar.volume) if hasattr(bar, 'volume') else 0
                })
        
        current_week = self.time.isocalendar()[1]
        if (current_week != self.last_analysis_week and 
            len(self.ohlc_history) >= self.min_data_points):
            self.run_analysis()
            self.last_analysis_week = current_week
    
    def run_analysis(self):
        """Run correlation analysis with parseable output"""
        
        if len(self.ohlc_history) < self.min_data_points:
            return
        
        current_year = self.time.year
        current_week = self.time.isocalendar()[1]
        week_year_str = f"{current_year}_W{current_week:02d}"
        
        self.log(f"Running OHLC Correlation Analysis for {week_year_str} with {len(self.ohlc_history)} observations")
        
        try:
            ohlc_df = pd.DataFrame.from_dict(dict(self.ohlc_history), orient='index')
            ohlc_df.index = pd.to_datetime(ohlc_df.index)
            ohlc_df = ohlc_df.sort_index()
            
            min_cols_per_symbol = 4
            valid_rows = []
            for idx, row in ohlc_df.iterrows():
                symbol_counts = {}
                for col in row.index:
                    symbol = col.split('_')[0]
                    if symbol not in symbol_counts:
                        symbol_counts[symbol] = 0
                    if not pd.isna(row[col]):
                        symbol_counts[symbol] += 1
                
                if all(count >= min_cols_per_symbol for count in symbol_counts.values()):
                    valid_rows.append(idx)
            
            if len(valid_rows) < self.min_data_points:
                self.log(f"Insufficient clean data: {len(valid_rows)} valid rows")
                return
            
            ohlc_df = ohlc_df.loc[valid_rows]
            
            returns_df = self.analyzer.calculate_returns(ohlc_df)
            
            for symbol in self.symbols:
                symbol_cols = [col for col in ohlc_df.columns if col.startswith(symbol)]
                if len(symbol_cols) >= 4:
                    close_col = f'{symbol}_close'
                    if close_col in ohlc_df.columns:
                        valid_data_points = ohlc_df[close_col].count()
                        self.log(f"{symbol}: {valid_data_points} valid data points")
            
            correlation_results = self.analyzer.compute_correlations(ohlc_df, returns_df)
            
            win_loss_df = self.analyzer.classify_win_loss_days(returns_df)
            
            stats_results = self.analyzer.calculate_statistics(returns_df)
            
            self.analyzer.create_summary_dataframes(ohlc_df, returns_df, correlation_results, stats_results)
            
            self.log_brief_results(correlation_results, stats_results, win_loss_df)
            
            self.analyzer.log_parseable_weekly_summary(week_year_str)
            
            self.analyzer.data = ohlc_df
            self.analyzer.correlation_results = correlation_results
            
            self.collection_complete = True
            
        except Exception as e:
            self.log(f"Analysis error: {str(e)}")
    
    def log_brief_results(self, correlation_results, stats_results, win_loss_df):
        """Log brief traditional results"""
        
        self.log("--- BRIEF ANALYSIS SUMMARY ---")
        
        self.log("Key Correlations:")
        for pair, results in correlation_results.items():
            pearson = results.get('pearson_returns', np.nan)
            rz_basic = results.get('rz_basic', np.nan)
            if not (np.isnan(pearson) or np.isnan(rz_basic)):
                self.log(f"  {pair}: Traditional={pearson:.3f}, OHLC={rz_basic:.3f}")
        
        self.log("Key Statistics:")
        for symbol, stats_data in stats_results.items():
            self.log(f"  {symbol}: Return={stats_data['mean']:.4f}, Sharpe={stats_data['sharpe_ratio']:.2f}")
    
    def on_end_of_algorithm(self):
        """Final analysis and comprehensive summary"""
        
        if self.collection_complete:
            self.log("")
            self.log("="*80)
            self.log("FINAL ANALYSIS COMPLETE - GENERATING COMPREHENSIVE SUMMARY")
            self.log("="*80)
            
            self.analyzer.log_comprehensive_final_summary()
            
            self.log("")
            self.log("="*80)
            self.log("DATA EXTRACTION GUIDE")
            self.log("="*80)
            self.log("1. WEEKLY SUMMARIES: Look for ===WEEKLY_SUMMARY_START_YYYY_WXX=== markers")
            self.log("2. FINAL SUMMARY: Look for ===FINAL_COMPREHENSIVE_SUMMARY_START=== marker")
            self.log("3. CSV DATA: Copy text after 'CSV_' headers and paste into Excel/Python")
            self.log("4. JSON DATA: Copy text after 'JSON_' headers for structured data")
            self.log("5. All data is formatted for easy copy/paste extraction")
            self.log("")
            self.log("Framework analysis complete - Ready for signal development!")


class VIXData(PythonData):
    """Custom data class for VIX data in QuantConnect"""
    
    def get_source(self, config, date, is_live_mode):
        return SubscriptionDataSource(
            "https://cdn.cboe.com/api/global/us_indices/daily_prices/VIX_History.csv",
            SubscriptionTransportMedium.RemoteFile
        )
    
    def reader(self, config, line, date, is_live_mode):
        
        if not line or line[0] == 'D':
            return None
        
        try:
            data = line.split(',')
            
            if len(data) < 5:
                return None
                
            vix = VIXData()
            vix.symbol = config.symbol
            vix.time = datetime.strptime(data[0].strip(), "%m/%d/%Y")
            
            vix_open = float(data[1].strip())
            vix_high = float(data[2].strip()) 
            vix_low = float(data[3].strip())
            vix_close = float(data[4].strip())
            
            vix.value = vix_close
            vix.open = vix_open
            vix.high = vix_high
            vix.low = vix_low
            vix.close = vix_close
            
            return vix
            
        except (ValueError, IndexError):
            return None


# PARSEABLE OUTPUT UTILITY FUNCTIONS
def parse_log_to_dataframes(log_text):
    """Utility function to parse QuantConnect logs and extract DataFrames"""
    
    weekly_summaries = {}
    final_summary = {}
    
    lines = log_text.split('\n')
    current_section = None
    current_week = None
    csv_data = []
    
    for line in lines:
        line = line.strip()
        
        if line.startswith('===WEEKLY_SUMMARY_START_'):
            current_week = line.replace('===WEEKLY_SUMMARY_START_', '').replace('===', '')
            weekly_summaries[current_week] = {}
            current_section = 'weekly'
            continue
        
        if line.startswith('===WEEKLY_SUMMARY_END_'):
            current_section = None
            current_week = None
            csv_data = []
            continue
        
        if line.startswith('===FINAL_COMPREHENSIVE_SUMMARY_START==='):
            current_section = 'final'
            continue
        
        if line.startswith('===FINAL_COMPREHENSIVE_SUMMARY_END==='):
            current_section = None
            csv_data = []
            continue
        
        if current_section and line.startswith('CSV_'):
            csv_type = line.replace('CSV_', '').replace(':', '')
            csv_data = []
            parsing_csv = True
            continue
        
        if current_section and line.startswith('JSON_'):
            json_type = line.replace('JSON_', '').replace(':', '')
            continue
        
        if current_section and csv_data is not None and line and not line.startswith('==='):
            if ',' in line:
                csv_data.append(line)
                
                if current_section == 'weekly' and current_week:
                    if csv_type not in weekly_summaries[current_week]:
                        weekly_summaries[current_week][csv_type] = []
                    weekly_summaries[current_week][csv_type] = csv_data.copy()
                elif current_section == 'final':
                    if csv_type not in final_summary:
                        final_summary[csv_type] = []
                    final_summary[csv_type] = csv_data.copy()
    
    return weekly_summaries, final_summary

def csv_lines_to_dataframe(csv_lines):
    """Convert list of CSV lines to pandas DataFrame"""
    if not csv_lines:
        return pd.DataFrame()
    
    header = csv_lines[0].split(',')
    
    data_rows = []
    for line in csv_lines[1:]:
        row_data = line.split(',')
        converted_row = []
        for val in row_data:
            val = val.strip()
            if val == '' or val.lower() == 'none':
                converted_row.append(None)
            else:
                try:
                    if '.' in val:
                        converted_row.append(float(val))
                    else:
                        converted_row.append(int(val))
                except ValueError:
                    converted_row.append(val)
        data_rows.append(converted_row)
    
    return pd.DataFrame(data_rows, columns=header)

def extract_all_dataframes_from_log(log_text):
    """Main function to extract all DataFrames from QuantConnect log"""
    
    weekly_data, final_data = parse_log_to_dataframes(log_text)
    
    extracted_data = {
        'weekly_summaries': {},
        'final_summary': {}
    }
    
    for week, week_data in weekly_data.items():
        extracted_data['weekly_summaries'][week] = {}
        for csv_type, csv_lines in week_data.items():
            df = csv_lines_to_dataframe(csv_lines)
            if not df.empty:
                extracted_data['weekly_summaries'][week][csv_type] = df
    
    for csv_type, csv_lines in final_data.items():
        df = csv_lines_to_dataframe(csv_lines)
        if not df.empty:
            extracted_data['final_summary'][csv_type] = df
    
    return extracted_data

def create_master_summary_csv(extracted_data, output_path="./"):
    """Create master CSV files from extracted log data"""
    
    created_files = []
    
    all_weekly_ohlc = []
    for week, week_data in extracted_data['weekly_summaries'].items():
        if 'OHLC_SUMMARY' in week_data:
            df = week_data['OHLC_SUMMARY'].copy()
            df['week'] = week
            all_weekly_ohlc.append(df)
    
    if all_weekly_ohlc:
        master_ohlc = pd.concat(all_weekly_ohlc, ignore_index=True)
        ohlc_file = f"{output_path}master_weekly_ohlc.csv"
        master_ohlc.to_csv(ohlc_file, index=False)
        created_files.append(ohlc_file)
    
    all_weekly_corr = []
    for week, week_data in extracted_data['weekly_summaries'].items():
        if 'CORRELATION_MATRIX' in week_data:
            df = week_data['CORRELATION_MATRIX'].copy()
            df['week'] = week
            all_weekly_corr.append(df)
    
    if all_weekly_corr:
        master_corr = pd.concat(all_weekly_corr, ignore_index=True)
        corr_file = f"{output_path}master_weekly_correlations.csv"
        master_corr.to_csv(corr_file, index=False)
        created_files.append(corr_file)
    
    if 'final_summary' in extracted_data and extracted_data['final_summary']:
        for summary_type, df in extracted_data['final_summary'].items():
            if not df.empty:
                summary_file = f"{output_path}final_{summary_type.lower()}.csv"
                df.to_csv(summary_file, index=False)
                created_files.append(summary_file)
    
    return created_files

def main_extract_and_process(log_file_path):
    """Main function to process QuantConnect log file"""
    
    with open(log_file_path, 'r') as f:
        log_text = f.read()
    
    print("Extracting DataFrames from log...")
    extracted_data = extract_all_dataframes_from_log(log_text)
    
    print("Creating master CSV files...")
    csv_files = create_master_summary_csv(extracted_data)
    
    print(f"\nProcessing Complete!")
    print(f"Created {len(csv_files)} CSV files:")
    for file_path in csv_files:
        print(f"  - {file_path}")
    
    return {
        'extracted_data': extracted_data,
        'csv_files': csv_files
    }

# USAGE INSTRUCTIONS
USAGE_GUIDE = """
STEP 1: Run this algorithm in QuantConnect
STEP 2: Copy the entire log output to a text file
STEP 3: Look for markers: ===WEEKLY_SUMMARY_START_*=== and ===FINAL_COMPREHENSIVE_SUMMARY_START===
STEP 4: Use main_extract_and_process('your_log_file.txt') to get CSV files automatically
STEP 5: Manual extraction: copy CSV data after 'CSV_' headers into Excel/Python
"""

if __name__ == "__main__":
    print("Enhanced QuantConnect OHLC Correlation Analysis Framework")
    print("Ready for deployment - No Object Store required!")
    print(USAGE_GUIDE)
