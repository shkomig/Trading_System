"""
System Monitor
Real-time monitoring of trading system performance and health
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass, field
import psutil
import time

from .alert_manager import AlertManager, AlertLevel

logger = logging.getLogger(__name__)


@dataclass
class SystemMetrics:
    """System performance metrics"""
    timestamp: datetime = field(default_factory=datetime.now)
    cpu_percent: float = 0.0
    memory_percent: float = 0.0
    memory_used_mb: float = 0.0
    disk_usage_percent: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'timestamp': self.timestamp.isoformat(),
            'cpu_percent': self.cpu_percent,
            'memory_percent': self.memory_percent,
            'memory_used_mb': self.memory_used_mb,
            'disk_usage_percent': self.disk_usage_percent
        }


@dataclass
class TradingMetrics:
    """Trading performance metrics"""
    timestamp: datetime = field(default_factory=datetime.now)
    portfolio_value: float = 0.0
    daily_pnl: float = 0.0
    daily_pnl_percent: float = 0.0
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    open_positions: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'timestamp': self.timestamp.isoformat(),
            'portfolio_value': self.portfolio_value,
            'daily_pnl': self.daily_pnl,
            'daily_pnl_percent': self.daily_pnl_percent,
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'losing_trades': self.losing_trades,
            'open_positions': self.open_positions,
            'win_rate': (self.winning_trades / self.total_trades * 100) if self.total_trades > 0 else 0
        }


class SystemMonitor:
    """
    Monitors trading system health and performance
    
    Tracks:
    - System resources (CPU, memory, disk)
    - Trading performance (P&L, trades, positions)
    - Connection status
    - Error rates
    
    Sends alerts when thresholds are exceeded
    
    Example:
        >>> monitor = SystemMonitor(alert_manager)
        >>> monitor.start_monitoring()
        >>> 
        >>> # Update trading metrics
        >>> monitor.update_trading_metrics(
        ...     portfolio_value=105000,
        ...     daily_pnl=500,
        ...     total_trades=10
        ... )
        >>> 
        >>> # Get current status
        >>> status = monitor.get_status()
    """
    
    def __init__(
        self,
        alert_manager: Optional[AlertManager] = None,
        check_interval: int = 60,
        cpu_threshold: float = 80.0,
        memory_threshold: float = 80.0,
        max_daily_loss_pct: float = 5.0
    ):
        """
        Initialize System Monitor
        
        Args:
            alert_manager: AlertManager for sending alerts
            check_interval: Monitoring check interval in seconds
            cpu_threshold: CPU usage alert threshold (%)
            memory_threshold: Memory usage alert threshold (%)
            max_daily_loss_pct: Maximum daily loss percentage
        """
        self.alert_manager = alert_manager or AlertManager()
        self.check_interval = check_interval
        self.cpu_threshold = cpu_threshold
        self.memory_threshold = memory_threshold
        self.max_daily_loss_pct = max_daily_loss_pct
        
        self.is_monitoring = False
        self.start_time = None
        
        # Metrics history
        self.system_metrics_history: List[SystemMetrics] = []
        self.trading_metrics_history: List[TradingMetrics] = []
        
        # Current metrics
        self.current_system_metrics = SystemMetrics()
        self.current_trading_metrics = TradingMetrics()
        
        # Connection status
        self.is_connected_to_broker = False
        self.last_broker_check = None
        
        # Error tracking
        self.error_count = 0
        self.last_error_time = None
        
        logger.info("SystemMonitor initialized")
    
    def start_monitoring(self):
        """Start monitoring system"""
        self.is_monitoring = True
        self.start_time = datetime.now()
        logger.info("System monitoring started")
    
    def stop_monitoring(self):
        """Stop monitoring system"""
        self.is_monitoring = False
        logger.info("System monitoring stopped")
    
    def collect_system_metrics(self) -> SystemMetrics:
        """Collect current system metrics"""
        try:
            metrics = SystemMetrics(
                cpu_percent=psutil.cpu_percent(interval=1),
                memory_percent=psutil.virtual_memory().percent,
                memory_used_mb=psutil.virtual_memory().used / (1024 * 1024),
                disk_usage_percent=psutil.disk_usage('/').percent
            )
            
            self.current_system_metrics = metrics
            self.system_metrics_history.append(metrics)
            
            # Check thresholds
            self._check_system_thresholds(metrics)
            
            return metrics
            
        except Exception as e:
            logger.error(f"Failed to collect system metrics: {e}")
            return SystemMetrics()
    
    def _check_system_thresholds(self, metrics: SystemMetrics):
        """Check if system metrics exceed thresholds"""
        # CPU threshold
        if metrics.cpu_percent > self.cpu_threshold:
            self.alert_manager.add_alert(
                AlertLevel.WARNING,
                "High CPU Usage",
                f"CPU usage at {metrics.cpu_percent:.1f}% (threshold: {self.cpu_threshold}%)",
                metadata={'cpu_percent': metrics.cpu_percent}
            )
        
        # Memory threshold
        if metrics.memory_percent > self.memory_threshold:
            self.alert_manager.add_alert(
                AlertLevel.WARNING,
                "High Memory Usage",
                f"Memory usage at {metrics.memory_percent:.1f}% (threshold: {self.memory_threshold}%)",
                metadata={'memory_percent': metrics.memory_percent}
            )
    
    def update_trading_metrics(
        self,
        portfolio_value: float,
        daily_pnl: float,
        daily_pnl_percent: float = 0.0,
        total_trades: int = 0,
        winning_trades: int = 0,
        losing_trades: int = 0,
        open_positions: int = 0
    ):
        """
        Update trading metrics
        
        Args:
            portfolio_value: Current portfolio value
            daily_pnl: Daily profit/loss
            daily_pnl_percent: Daily P&L percentage
            total_trades: Total number of trades
            winning_trades: Number of winning trades
            losing_trades: Number of losing trades
            open_positions: Number of open positions
        """
        metrics = TradingMetrics(
            portfolio_value=portfolio_value,
            daily_pnl=daily_pnl,
            daily_pnl_percent=daily_pnl_percent,
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            open_positions=open_positions
        )
        
        self.current_trading_metrics = metrics
        self.trading_metrics_history.append(metrics)
        
        # Check trading thresholds
        self._check_trading_thresholds(metrics)
    
    def _check_trading_thresholds(self, metrics: TradingMetrics):
        """Check if trading metrics exceed thresholds"""
        # Daily loss limit
        if metrics.daily_pnl_percent < -self.max_daily_loss_pct:
            self.alert_manager.add_alert(
                AlertLevel.CRITICAL,
                "Daily Loss Limit Exceeded",
                f"Daily loss at {metrics.daily_pnl_percent:.2f}% "
                f"(limit: {self.max_daily_loss_pct}%)\n"
                f"Consider stopping trading for today.",
                metadata={
                    'daily_pnl': metrics.daily_pnl,
                    'daily_pnl_percent': metrics.daily_pnl_percent
                }
            )
    
    def update_connection_status(self, is_connected: bool):
        """
        Update broker connection status
        
        Args:
            is_connected: Whether connected to broker
        """
        was_connected = self.is_connected_to_broker
        self.is_connected_to_broker = is_connected
        self.last_broker_check = datetime.now()
        
        # Alert on connection change
        if was_connected and not is_connected:
            self.alert_manager.add_alert(
                AlertLevel.ERROR,
                "Broker Connection Lost",
                "Lost connection to broker. Attempting to reconnect..."
            )
        elif not was_connected and is_connected:
            self.alert_manager.add_alert(
                AlertLevel.INFO,
                "Broker Connected",
                "Successfully connected to broker"
            )
    
    def log_error(self, error_msg: str, details: Optional[Dict[str, Any]] = None):
        """
        Log error and track error rate
        
        Args:
            error_msg: Error message
            details: Additional error details
        """
        self.error_count += 1
        self.last_error_time = datetime.now()
        
        self.alert_manager.add_alert(
            AlertLevel.ERROR,
            "System Error",
            error_msg,
            metadata=details or {}
        )
        
        # Check error rate
        if self.error_count > 10:
            time_window = timedelta(minutes=10)
            if self.last_error_time and datetime.now() - self.last_error_time < time_window:
                self.alert_manager.add_alert(
                    AlertLevel.CRITICAL,
                    "High Error Rate",
                    f"Detected {self.error_count} errors in last {time_window.seconds // 60} minutes"
                )
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get current system status
        
        Returns:
            Dictionary with current status
        """
        uptime = None
        if self.start_time:
            uptime = (datetime.now() - self.start_time).total_seconds()
        
        return {
            'is_monitoring': self.is_monitoring,
            'uptime_seconds': uptime,
            'system_metrics': self.current_system_metrics.to_dict(),
            'trading_metrics': self.current_trading_metrics.to_dict(),
            'is_connected_to_broker': self.is_connected_to_broker,
            'last_broker_check': self.last_broker_check.isoformat() if self.last_broker_check else None,
            'error_count': self.error_count,
            'last_error_time': self.last_error_time.isoformat() if self.last_error_time else None
        }
    
    def get_metrics_summary(
        self,
        time_window: Optional[timedelta] = None
    ) -> Dict[str, Any]:
        """
        Get summary of metrics over time window
        
        Args:
            time_window: Time window for summary (default: last hour)
            
        Returns:
            Dictionary with metrics summary
        """
        if time_window is None:
            time_window = timedelta(hours=1)
        
        cutoff_time = datetime.now() - time_window
        
        # Filter metrics
        recent_system = [m for m in self.system_metrics_history if m.timestamp > cutoff_time]
        recent_trading = [m for m in self.trading_metrics_history if m.timestamp > cutoff_time]
        
        summary = {
            'time_window_hours': time_window.total_seconds() / 3600,
            'system': {},
            'trading': {}
        }
        
        # System metrics summary
        if recent_system:
            import numpy as np
            summary['system'] = {
                'avg_cpu': np.mean([m.cpu_percent for m in recent_system]),
                'max_cpu': np.max([m.cpu_percent for m in recent_system]),
                'avg_memory': np.mean([m.memory_percent for m in recent_system]),
                'max_memory': np.max([m.memory_percent for m in recent_system])
            }
        
        # Trading metrics summary
        if recent_trading:
            latest = recent_trading[-1]
            summary['trading'] = {
                'current_portfolio_value': latest.portfolio_value,
                'daily_pnl': latest.daily_pnl,
                'daily_pnl_percent': latest.daily_pnl_percent,
                'total_trades': latest.total_trades,
                'win_rate': (latest.winning_trades / latest.total_trades * 100) if latest.total_trades > 0 else 0,
                'open_positions': latest.open_positions
            }
        
        return summary
    
    def print_status(self):
        """Print formatted status to console"""
        status = self.get_status()
        
        print("\n" + "="*70)
        print("SYSTEM MONITOR STATUS")
        print("="*70)
        
        print(f"\nMonitoring: {'ACTIVE' if status['is_monitoring'] else 'INACTIVE'}")
        if status['uptime_seconds']:
            hours = int(status['uptime_seconds'] // 3600)
            minutes = int((status['uptime_seconds'] % 3600) // 60)
            print(f"Uptime: {hours}h {minutes}m")
        
        print("\nSystem Metrics:")
        sys_metrics = status['system_metrics']
        print(f"  CPU: {sys_metrics['cpu_percent']:.1f}%")
        print(f"  Memory: {sys_metrics['memory_percent']:.1f}% ({sys_metrics['memory_used_mb']:.0f} MB)")
        print(f"  Disk: {sys_metrics['disk_usage_percent']:.1f}%")
        
        print("\nTrading Metrics:")
        trade_metrics = status['trading_metrics']
        print(f"  Portfolio Value: ${trade_metrics['portfolio_value']:,.2f}")
        print(f"  Daily P&L: ${trade_metrics['daily_pnl']:,.2f} ({trade_metrics['daily_pnl_percent']:+.2f}%)")
        print(f"  Total Trades: {trade_metrics['total_trades']}")
        print(f"  Win Rate: {trade_metrics['win_rate']:.1f}%")
        print(f"  Open Positions: {trade_metrics['open_positions']}")
        
        print("\nConnection:")
        print(f"  Broker: {'CONNECTED' if status['is_connected_to_broker'] else 'DISCONNECTED'}")
        
        print("\nErrors:")
        print(f"  Total: {status['error_count']}")
        
        print("="*70 + "\n")

