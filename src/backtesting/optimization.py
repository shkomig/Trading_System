"""
אופטימיזציה של פרמטרי אסטרטגיות

כולל Grid Search, Random Search ו-Walk-Forward Optimization.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Callable, Optional, Any
from itertools import product
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

logger = logging.getLogger(__name__)


class StrategyOptimizer:
    """
    מחלקה לאופטימיזציה של פרמטרי אסטרטגיות
    
    תומכת במספר שיטות אופטימיזציה:
    - Grid Search
    - Random Search
    - Walk-Forward Optimization
    """
    
    def __init__(self, strategy_class, backtest_function):
        """
        אתחול
        
        Args:
            strategy_class: מחלקת האסטרטגיה
            backtest_function: פונקציה להרצת backtest
        """
        self.strategy_class = strategy_class
        self.backtest_function = backtest_function
        self.results = []
        
        logger.info(f"StrategyOptimizer initialized for {strategy_class.__name__}")
    
    def grid_search(
        self,
        data: pd.DataFrame,
        param_grid: Dict[str, List],
        metric: str = 'sharpe_ratio',
        n_jobs: int = 1
    ) -> Tuple[Dict, pd.DataFrame]:
        """
        Grid Search על כל הקומבינציות של פרמטרים
        
        Args:
            data: נתוני שוק
            param_grid: מילון של פרמטרים ורשימות ערכים
            metric: מטריקה לאופטימיזציה
            n_jobs: מספר תהליכים מקבילים
            
        Returns:
            Tuple של (best_params, results_df)
        """
        logger.info(f"Starting Grid Search with {len(param_grid)} parameters")
        
        # יצירת כל הקומבינציות
        keys = list(param_grid.keys())
        values = list(param_grid.values())
        combinations = list(product(*values))
        
        logger.info(f"Testing {len(combinations)} parameter combinations")
        
        results = []
        
        if n_jobs == 1:
            # הרצה סדרתית
            for combo in tqdm(combinations, desc="Grid Search"):
                params = dict(zip(keys, combo))
                result = self._evaluate_params(data, params, metric)
                results.append(result)
        else:
            # הרצה מקבילית
            with ProcessPoolExecutor(max_workers=n_jobs) as executor:
                futures = []
                for combo in combinations:
                    params = dict(zip(keys, combo))
                    future = executor.submit(self._evaluate_params, data, params, metric)
                    futures.append(future)
                
                for future in tqdm(as_completed(futures), total=len(futures), desc="Grid Search"):
                    result = future.result()
                    results.append(result)
        
        self.results = results
        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values(metric, ascending=False)
        
        best_params = results_df.iloc[0]['params']
        
        logger.info(f"Grid Search completed. Best {metric}: {results_df.iloc[0][metric]:.4f}")
        logger.info(f"Best params: {best_params}")
        
        return best_params, results_df
    
    def random_search(
        self,
        data: pd.DataFrame,
        param_distributions: Dict[str, Callable],
        n_iter: int = 100,
        metric: str = 'sharpe_ratio',
        n_jobs: int = 1,
        random_state: Optional[int] = None
    ) -> Tuple[Dict, pd.DataFrame]:
        """
        Random Search - דגימה אקראית של פרמטרים
        
        Args:
            data: נתוני שוק
            param_distributions: מילון של פרמטרים ופונקציות דגימה
            n_iter: מספר איטרציות
            metric: מטריקה לאופטימיזציה
            n_jobs: מספר תהליכים מקבילים
            random_state: seed אקראי
            
        Returns:
            Tuple של (best_params, results_df)
        """
        if random_state is not None:
            np.random.seed(random_state)
        
        logger.info(f"Starting Random Search with {n_iter} iterations")
        
        results = []
        
        # יצירת פרמטרים אקראיים
        param_samples = []
        for _ in range(n_iter):
            params = {}
            for key, dist_func in param_distributions.items():
                params[key] = dist_func()
            param_samples.append(params)
        
        if n_jobs == 1:
            for params in tqdm(param_samples, desc="Random Search"):
                result = self._evaluate_params(data, params, metric)
                results.append(result)
        else:
            with ProcessPoolExecutor(max_workers=n_jobs) as executor:
                futures = []
                for params in param_samples:
                    future = executor.submit(self._evaluate_params, data, params, metric)
                    futures.append(future)
                
                for future in tqdm(as_completed(futures), total=len(futures), desc="Random Search"):
                    result = future.result()
                    results.append(result)
        
        self.results = results
        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values(metric, ascending=False)
        
        best_params = results_df.iloc[0]['params']
        
        logger.info(f"Random Search completed. Best {metric}: {results_df.iloc[0][metric]:.4f}")
        logger.info(f"Best params: {best_params}")
        
        return best_params, results_df
    
    def walk_forward_optimization(
        self,
        data: pd.DataFrame,
        param_grid: Dict[str, List],
        train_period: int = 252,
        test_period: int = 63,
        metric: str = 'sharpe_ratio'
    ) -> Tuple[List[Dict], pd.DataFrame]:
        """
        Walk-Forward Optimization
        
        אופטימיזציה על חלון נע - אימון על תקופה, בדיקה על התקופה הבאה.
        
        Args:
            data: נתוני שוק
            param_grid: פרמטרים לאופטימיזציה
            train_period: גודל חלון אימון (ימים)
            test_period: גודל חלון בדיקה (ימים)
            metric: מטריקה לאופטימיזציה
            
        Returns:
            Tuple של (walk_forward_results, summary_df)
        """
        logger.info("Starting Walk-Forward Optimization")
        
        walk_forward_results = []
        total_length = len(data)
        
        # חישוב מספר חלונות
        n_windows = (total_length - train_period) // test_period
        
        logger.info(f"Will test {n_windows} windows")
        
        for i in range(n_windows):
            # הגדרת חלונות
            train_start = i * test_period
            train_end = train_start + train_period
            test_start = train_end
            test_end = test_start + test_period
            
            if test_end > total_length:
                break
            
            train_data = data.iloc[train_start:train_end]
            test_data = data.iloc[test_start:test_end]
            
            logger.info(f"Window {i+1}/{n_windows}: Train {train_start}-{train_end}, Test {test_start}-{test_end}")
            
            # אופטימיזציה על train
            best_params, _ = self.grid_search(train_data, param_grid, metric, n_jobs=1)
            
            # בדיקה על test
            test_result = self._evaluate_params(test_data, best_params, metric)
            
            walk_forward_results.append({
                'window': i + 1,
                'train_period': f"{train_start}-{train_end}",
                'test_period': f"{test_start}-{test_end}",
                'best_params': best_params,
                'test_performance': test_result
            })
        
        # סיכום
        summary_df = pd.DataFrame([
            {
                'window': r['window'],
                'test_sharpe': r['test_performance']['sharpe_ratio'],
                'test_return': r['test_performance']['total_return'],
                **r['best_params']
            }
            for r in walk_forward_results
        ])
        
        logger.info(f"Walk-Forward completed. Avg test Sharpe: {summary_df['test_sharpe'].mean():.4f}")
        
        return walk_forward_results, summary_df
    
    def _evaluate_params(
        self,
        data: pd.DataFrame,
        params: Dict,
        metric: str
    ) -> Dict:
        """
        הערכת קומבינציה של פרמטרים
        
        Args:
            data: נתוני שוק
            params: פרמטרים
            metric: מטריקה לאופטימיזציה
            
        Returns:
            מילון עם תוצאות
        """
        try:
            # יצירת אסטרטגיה עם הפרמטרים
            strategy = self.strategy_class(**params)
            
            # הרצת backtest
            results = self.backtest_function(strategy, data)
            
            return {
                'params': params,
                **results
            }
        except Exception as e:
            logger.error(f"Error evaluating params {params}: {e}")
            return {
                'params': params,
                metric: -np.inf,
                'error': str(e)
            }
    
    def plot_optimization_results(
        self,
        results_df: pd.DataFrame,
        param1: str,
        param2: Optional[str] = None,
        metric: str = 'sharpe_ratio'
    ):
        """
        ציור תוצאות אופטימיזציה
        
        Args:
            results_df: DataFrame עם תוצאות
            param1: פרמטר ראשון
            param2: פרמטר שני (אופציונלי)
            metric: מטריקה להצגה
        """
        import matplotlib.pyplot as plt
        
        if param2 is None:
            # ציור 1D
            params_values = [r['params'][param1] for r in results_df['params']]
            metric_values = results_df[metric].values
            
            plt.figure(figsize=(10, 6))
            plt.plot(params_values, metric_values, 'o-')
            plt.xlabel(param1)
            plt.ylabel(metric)
            plt.title(f'{metric} vs {param1}')
            plt.grid(True, alpha=0.3)
            plt.show()
        else:
            # ציור 2D (heatmap)
            param1_values = sorted(set(r['params'][param1] for r in results_df['params']))
            param2_values = sorted(set(r['params'][param2] for r in results_df['params']))
            
            # יצירת grid
            grid = np.zeros((len(param2_values), len(param1_values)))
            
            for _, row in results_df.iterrows():
                p1 = row['params'][param1]
                p2 = row['params'][param2]
                i = param2_values.index(p2)
                j = param1_values.index(p1)
                grid[i, j] = row[metric]
            
            plt.figure(figsize=(12, 8))
            plt.imshow(grid, cmap='viridis', aspect='auto', origin='lower')
            plt.colorbar(label=metric)
            plt.xticks(range(len(param1_values)), param1_values)
            plt.yticks(range(len(param2_values)), param2_values)
            plt.xlabel(param1)
            plt.ylabel(param2)
            plt.title(f'{metric} Heatmap')
            plt.tight_layout()
            plt.show()


def create_param_distribution(param_type: str, **kwargs) -> Callable:
    """
    יצירת פונקציית דגימה לפרמטר
    
    Args:
        param_type: 'uniform', 'int', 'choice', 'loguniform'
        **kwargs: פרמטרים ספציפיים
        
    Returns:
        פונקציה שמחזירה ערך אקראי
    """
    if param_type == 'uniform':
        low = kwargs['low']
        high = kwargs['high']
        return lambda: np.random.uniform(low, high)
    
    elif param_type == 'int':
        low = kwargs['low']
        high = kwargs['high']
        return lambda: np.random.randint(low, high + 1)
    
    elif param_type == 'choice':
        choices = kwargs['choices']
        return lambda: np.random.choice(choices)
    
    elif param_type == 'loguniform':
        low = kwargs['low']
        high = kwargs['high']
        return lambda: np.exp(np.random.uniform(np.log(low), np.log(high)))
    
    else:
        raise ValueError(f"Unknown param_type: {param_type}")

