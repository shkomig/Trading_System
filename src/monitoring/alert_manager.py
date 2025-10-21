"""
Alert Manager
System for generating and sending alerts
"""

import logging
from enum import Enum
from typing import Optional, List, Dict, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import os

logger = logging.getLogger(__name__)


class AlertLevel(Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class Alert:
    """
    Alert data class
    
    Attributes:
        level: Severity level of the alert
        title: Short alert title
        message: Detailed alert message
        timestamp: When the alert was created
        metadata: Additional information
    """
    level: AlertLevel
    title: str
    message: str
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert alert to dictionary"""
        return {
            'level': self.level.value,
            'title': self.title,
            'message': self.message,
            'timestamp': self.timestamp.isoformat(),
            'metadata': self.metadata
        }
    
    def __str__(self) -> str:
        """String representation"""
        return f"[{self.level.value.upper()}] {self.title}: {self.message}"


class AlertManager:
    """
    Manages alerts and notifications
    
    Supports multiple alert channels:
    - Logging
    - Email
    - Telegram (if configured)
    - Custom callbacks
    
    Example:
        >>> manager = AlertManager()
        >>> manager.add_alert(
        ...     AlertLevel.WARNING,
        ...     "High Volatility",
        ...     "Market volatility exceeded 30%"
        ... )
    """
    
    def __init__(
        self,
        enable_email: bool = False,
        email_config: Optional[Dict[str, Any]] = None,
        enable_telegram: bool = False,
        telegram_config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize Alert Manager
        
        Args:
            enable_email: Whether to send email alerts
            email_config: Email configuration (smtp_server, port, username, password, recipients)
            enable_telegram: Whether to send Telegram alerts
            telegram_config: Telegram configuration (bot_token, chat_id)
        """
        self.enable_email = enable_email
        self.email_config = email_config or {}
        self.enable_telegram = enable_telegram
        self.telegram_config = telegram_config or {}
        
        self.alerts_history: List[Alert] = []
        self.custom_callbacks: List[Callable] = []
        
        # Alert rules
        self.alert_rules: List[Dict[str, Any]] = []
        
        logger.info("AlertManager initialized")
    
    def add_alert(
        self,
        level: AlertLevel,
        title: str,
        message: str,
        metadata: Optional[Dict[str, Any]] = None,
        send_notifications: bool = True
    ) -> Alert:
        """
        Create and process a new alert
        
        Args:
            level: Alert severity level
            title: Short title
            message: Detailed message
            metadata: Additional information
            send_notifications: Whether to send notifications
            
        Returns:
            Created Alert object
        """
        alert = Alert(
            level=level,
            title=title,
            message=message,
            metadata=metadata or {}
        )
        
        self.alerts_history.append(alert)
        
        # Log alert
        self._log_alert(alert)
        
        # Send notifications if enabled
        if send_notifications:
            self._send_notifications(alert)
        
        return alert
    
    def _log_alert(self, alert: Alert):
        """Log alert to logging system"""
        log_msg = f"{alert.title}: {alert.message}"
        
        if alert.level == AlertLevel.INFO:
            logger.info(log_msg)
        elif alert.level == AlertLevel.WARNING:
            logger.warning(log_msg)
        elif alert.level == AlertLevel.ERROR:
            logger.error(log_msg)
        elif alert.level == AlertLevel.CRITICAL:
            logger.critical(log_msg)
    
    def _send_notifications(self, alert: Alert):
        """Send alert through all enabled channels"""
        # Email
        if self.enable_email and self.email_config:
            self._send_email(alert)
        
        # Telegram
        if self.enable_telegram and self.telegram_config:
            self._send_telegram(alert)
        
        # Custom callbacks
        for callback in self.custom_callbacks:
            try:
                callback(alert)
            except Exception as e:
                logger.error(f"Custom callback failed: {e}")
    
    def _send_email(self, alert: Alert):
        """Send alert via email"""
        try:
            smtp_server = self.email_config.get('smtp_server', 'smtp.gmail.com')
            port = self.email_config.get('port', 587)
            username = self.email_config.get('username')
            password = self.email_config.get('password')
            recipients = self.email_config.get('recipients', [])
            
            if not username or not password or not recipients:
                logger.warning("Email config incomplete. Skipping email alert.")
                return
            
            # Create message
            msg = MIMEMultipart()
            msg['From'] = username
            msg['To'] = ', '.join(recipients)
            msg['Subject'] = f"[{alert.level.value.upper()}] {alert.title}"
            
            body = f"""
Trading System Alert

Level: {alert.level.value.upper()}
Time: {alert.timestamp.strftime('%Y-%m-%d %H:%M:%S')}

{alert.message}

Metadata:
{alert.metadata}
            """
            
            msg.attach(MIMEText(body, 'plain'))
            
            # Send email
            with smtplib.SMTP(smtp_server, port) as server:
                server.starttls()
                server.login(username, password)
                server.send_message(msg)
            
            logger.info(f"Email alert sent to {recipients}")
            
        except Exception as e:
            logger.error(f"Failed to send email alert: {e}")
    
    def _send_telegram(self, alert: Alert):
        """Send alert via Telegram"""
        try:
            import requests
            
            bot_token = self.telegram_config.get('bot_token')
            chat_id = self.telegram_config.get('chat_id')
            
            if not bot_token or not chat_id:
                logger.warning("Telegram config incomplete. Skipping Telegram alert.")
                return
            
            # Format message
            emoji = {
                AlertLevel.INFO: 'ℹ️',
                AlertLevel.WARNING: '⚠️',
                AlertLevel.ERROR: '❌',
                AlertLevel.CRITICAL: '🚨'
            }
            
            text = f"""
{emoji.get(alert.level, '')} *{alert.level.value.upper()}*

*{alert.title}*

{alert.message}

_{alert.timestamp.strftime('%Y-%m-%d %H:%M:%S')}_
            """
            
            # Send message
            url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
            data = {
                'chat_id': chat_id,
                'text': text,
                'parse_mode': 'Markdown'
            }
            
            response = requests.post(url, data=data)
            response.raise_for_status()
            
            logger.info(f"Telegram alert sent")
            
        except Exception as e:
            logger.error(f"Failed to send Telegram alert: {e}")
    
    def add_callback(self, callback: Callable[[Alert], None]):
        """
        Add custom alert callback
        
        Args:
            callback: Function that takes Alert as argument
        """
        self.custom_callbacks.append(callback)
        logger.info(f"Added custom alert callback: {callback.__name__}")
    
    def add_rule(
        self,
        name: str,
        condition: Callable[[], bool],
        level: AlertLevel,
        title: str,
        message: str
    ):
        """
        Add alert rule that triggers when condition is met
        
        Args:
            name: Rule name
            condition: Function that returns True when alert should trigger
            level: Alert level
            title: Alert title
            message: Alert message
        """
        rule = {
            'name': name,
            'condition': condition,
            'level': level,
            'title': title,
            'message': message,
            'last_triggered': None
        }
        
        self.alert_rules.append(rule)
        logger.info(f"Added alert rule: {name}")
    
    def check_rules(self):
        """Check all alert rules and trigger if conditions are met"""
        for rule in self.alert_rules:
            try:
                if rule['condition']():
                    self.add_alert(
                        level=rule['level'],
                        title=rule['title'],
                        message=rule['message'],
                        metadata={'rule': rule['name']}
                    )
                    rule['last_triggered'] = datetime.now()
            except Exception as e:
                logger.error(f"Error checking rule '{rule['name']}': {e}")
    
    def get_recent_alerts(
        self,
        count: int = 10,
        level: Optional[AlertLevel] = None
    ) -> List[Alert]:
        """
        Get recent alerts
        
        Args:
            count: Number of alerts to return
            level: Filter by alert level
            
        Returns:
            List of recent alerts
        """
        alerts = self.alerts_history
        
        if level:
            alerts = [a for a in alerts if a.level == level]
        
        return alerts[-count:]
    
    def get_alerts_by_level(self) -> Dict[AlertLevel, int]:
        """Get count of alerts by level"""
        counts = {level: 0 for level in AlertLevel}
        
        for alert in self.alerts_history:
            counts[alert.level] += 1
        
        return counts
    
    def clear_alerts(self):
        """Clear alerts history"""
        self.alerts_history.clear()
        logger.info("Alerts history cleared")


# Pre-configured alert templates
def create_trade_alert(
    alert_manager: AlertManager,
    symbol: str,
    action: str,
    quantity: int,
    price: float,
    success: bool = True
):
    """Create alert for trade execution"""
    level = AlertLevel.INFO if success else AlertLevel.ERROR
    title = f"Trade {'Executed' if success else 'Failed'}: {action} {symbol}"
    message = f"{action} {quantity} shares of {symbol} at ${price:.2f}"
    
    alert_manager.add_alert(
        level=level,
        title=title,
        message=message,
        metadata={
            'symbol': symbol,
            'action': action,
            'quantity': quantity,
            'price': price,
            'success': success
        }
    )


def create_risk_alert(
    alert_manager: AlertManager,
    reason: str,
    details: str,
    current_value: float,
    threshold: float
):
    """Create alert for risk management trigger"""
    alert_manager.add_alert(
        level=AlertLevel.WARNING,
        title=f"Risk Alert: {reason}",
        message=f"{details}\nCurrent: {current_value:.2f}, Threshold: {threshold:.2f}",
        metadata={
            'reason': reason,
            'current_value': current_value,
            'threshold': threshold
        }
    )


def create_system_alert(
    alert_manager: AlertManager,
    level: AlertLevel,
    component: str,
    message: str
):
    """Create alert for system events"""
    alert_manager.add_alert(
        level=level,
        title=f"System: {component}",
        message=message,
        metadata={'component': component}
    )

