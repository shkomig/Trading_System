"""
Strategy Registry - ניהול מרוכז של כל האסטרטגיות

מאפשר רישום, טעינה ויצירה של אסטרטגיות בצורה דינמית.
"""

from typing import Dict, Type, List, Optional
from .base_strategy import BaseStrategy
import logging

logger = logging.getLogger(__name__)


class StrategyRegistry:
    """
    מרשם מרוכז של אסטרטגיות
    
    מאפשר לרשום אסטרטגיות חדשות וליצור instances שלהן בקלות.
    
    Example:
        >>> registry = StrategyRegistry()
        >>> registry.register('ma', MovingAverageCrossover)
        >>> strategy = registry.create('ma', short_window=50, long_window=200)
    """
    
    _instance = None
    _strategies: Dict[str, Type[BaseStrategy]] = {}
    
    def __new__(cls):
        """Singleton pattern"""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def register(self, name: str, strategy_class: Type[BaseStrategy]):
        """
        רישום אסטרטגיה חדשה
        
        Args:
            name: שם האסטרטגיה (מזהה ייחודי)
            strategy_class: מחלקת האסטרטגיה
        """
        if not issubclass(strategy_class, BaseStrategy):
            raise ValueError(f"{strategy_class} must inherit from BaseStrategy")
        
        self._strategies[name] = strategy_class
        logger.info(f"Registered strategy: {name} ({strategy_class.__name__})")
    
    def unregister(self, name: str):
        """הסרת אסטרטגיה מהמרשם"""
        if name in self._strategies:
            del self._strategies[name]
            logger.info(f"Unregistered strategy: {name}")
    
    def create(self, name: str, **kwargs) -> BaseStrategy:
        """
        יצירת instance של אסטרטגיה
        
        Args:
            name: שם האסטרטגיה
            **kwargs: פרמטרים לאסטרטגיה
            
        Returns:
            Instance של האסטרטגיה
        """
        if name not in self._strategies:
            raise ValueError(f"Strategy '{name}' not found in registry")
        
        strategy_class = self._strategies[name]
        return strategy_class(**kwargs)
    
    def list_strategies(self) -> List[str]:
        """
        קבלת רשימת כל האסטרטגיות הרשומות
        
        Returns:
            רשימת שמות אסטרטגיות
        """
        return list(self._strategies.keys())
    
    def get_strategy_class(self, name: str) -> Type[BaseStrategy]:
        """
        קבלת מחלקת אסטרטגיה
        
        Args:
            name: שם האסטרטגיה
            
        Returns:
            מחלקת האסטרטגיה
        """
        if name not in self._strategies:
            raise ValueError(f"Strategy '{name}' not found in registry")
        
        return self._strategies[name]
    
    def get_strategy_info(self, name: str) -> Dict:
        """
        קבלת מידע על אסטרטגיה
        
        Args:
            name: שם האסטרטגיה
            
        Returns:
            מילון עם מידע
        """
        if name not in self._strategies:
            raise ValueError(f"Strategy '{name}' not found in registry")
        
        strategy_class = self._strategies[name]
        
        return {
            'name': name,
            'class_name': strategy_class.__name__,
            'module': strategy_class.__module__,
            'docstring': strategy_class.__doc__
        }
    
    def clear(self):
        """ניקוי כל המרשם"""
        self._strategies.clear()
        logger.info("Registry cleared")


# רישום אוטומטי של אסטרטגיות מובנות
def register_builtin_strategies():
    """רישום כל האסטרטגיות המובנות"""
    from .technical.moving_average import MovingAverageCrossover, TripleMovingAverage
    from .technical.rsi_macd import RSI_MACD_Strategy, RSI_Divergence_Strategy
    from .technical.momentum import (
        MomentumStrategy, 
        DualMomentumStrategy, 
        TrendFollowingStrategy,
        MeanReversionStrategy
    )
    
    registry = StrategyRegistry()
    
    # Technical strategies
    registry.register('ma_crossover', MovingAverageCrossover)
    registry.register('triple_ma', TripleMovingAverage)
    registry.register('rsi_macd', RSI_MACD_Strategy)
    registry.register('rsi_divergence', RSI_Divergence_Strategy)
    registry.register('momentum', MomentumStrategy)
    registry.register('dual_momentum', DualMomentumStrategy)
    registry.register('trend_following', TrendFollowingStrategy)
    registry.register('mean_reversion', MeanReversionStrategy)
    
    logger.info(f"Registered {len(registry.list_strategies())} built-in strategies")


# רישום אוטומטי בטעינה
register_builtin_strategies()


# פונקציות עזר
def get_registry() -> StrategyRegistry:
    """קבלת ה-registry"""
    return StrategyRegistry()


def create_strategy(name: str, **kwargs) -> BaseStrategy:
    """
    יצירת אסטרטגיה
    
    Args:
        name: שם האסטרטגיה
        **kwargs: פרמטרים
        
    Returns:
        Instance של אסטרטגיה
    """
    registry = get_registry()
    return registry.create(name, **kwargs)


def list_available_strategies() -> List[str]:
    """רשימת אסטרטגיות זמינות"""
    registry = get_registry()
    return registry.list_strategies()


def get_strategy(name: str, **kwargs) -> Optional[BaseStrategy]:
    """
    קבלת instance של אסטרטגיה
    
    Args:
        name: שם האסטרטגיה
        **kwargs: פרמטרים לאסטרטגיה
        
    Returns:
        Instance של אסטרטגיה או None אם לא נמצאה
    """
    try:
        registry = get_registry()
        return registry.create(name, **kwargs)
    except ValueError:
        logger.warning(f"Strategy '{name}' not found")
        return None

