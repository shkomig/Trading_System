"""
Ollama LLM Analyzer
Uses local LLM (Ollama) for trading analysis, recommendations, and explanations
"""

import logging
from typing import Dict, Any, List, Optional
import requests
import json

logger = logging.getLogger(__name__)


class OllamaAnalyzer:
    """
    Local LLM analyzer using Ollama
    
    Provides:
    - Sentiment analysis from news/text
    - Trading strategy recommendations
    - Trade decision explanations
    - Market analysis insights
    
    Example:
        >>> analyzer = OllamaAnalyzer(model="llama2")
        >>> sentiment = analyzer.analyze_sentiment("Market showing strong bullish signals")
        >>> recommendation = analyzer.get_strategy_recommendation(market_data)
    """
    
    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "llama2",
        timeout: int = 30
    ):
        """
        Initialize Ollama Analyzer
        
        Args:
            base_url: Ollama API base URL
            model: Model name to use (llama2, mistral, etc.)
            timeout: Request timeout in seconds
        """
        self.base_url = base_url
        self.model = model
        self.timeout = timeout
        
        self.is_available = self._check_availability()
        
        if self.is_available:
            logger.info(f"Ollama analyzer initialized with model: {model}")
        else:
            logger.warning("Ollama not available. LLM features will be disabled.")
    
    def _check_availability(self) -> bool:
        """Check if Ollama is running and accessible"""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            return response.status_code == 200
        except Exception:
            return False
    
    def _generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7
    ) -> str:
        """
        Generate response from Ollama
        
        Args:
            prompt: User prompt
            system_prompt: System prompt for context
            temperature: Generation temperature
            
        Returns:
            Generated text response
        """
        if not self.is_available:
            return "Ollama not available. Please start Ollama service."
        
        try:
            url = f"{self.base_url}/api/generate"
            
            data = {
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": temperature
                }
            }
            
            if system_prompt:
                data["system"] = system_prompt
            
            response = requests.post(
                url,
                json=data,
                timeout=self.timeout
            )
            
            response.raise_for_status()
            result = response.json()
            
            return result.get('response', '').strip()
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return f"Error: {str(e)}"
    
    def analyze_sentiment(
        self,
        text: str
    ) -> Dict[str, Any]:
        """
        Analyze sentiment from text
        
        Args:
            text: Text to analyze (news, social media, etc.)
            
        Returns:
            Dictionary with sentiment analysis results
        """
        system_prompt = """You are a financial sentiment analyzer. 
Analyze the given text and provide:
1. Overall sentiment (bullish/bearish/neutral)
2. Confidence score (0-1)
3. Key phrases that influenced the sentiment
4. Brief explanation

Respond in JSON format only."""
        
        prompt = f"""Analyze the sentiment of this financial text:

"{text}"

Provide your analysis in this JSON format:
{{
    "sentiment": "bullish|bearish|neutral",
    "confidence": 0.85,
    "key_phrases": ["phrase1", "phrase2"],
    "explanation": "Brief explanation"
}}"""
        
        response = self._generate(prompt, system_prompt, temperature=0.3)
        
        try:
            # Try to parse JSON response
            result = json.loads(response)
            return result
        except json.JSONDecodeError:
            # Fallback if not valid JSON
            return {
                "sentiment": "neutral",
                "confidence": 0.5,
                "key_phrases": [],
                "explanation": response[:200]
            }
    
    def get_strategy_recommendation(
        self,
        market_data: Dict[str, Any],
        current_positions: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """
        Get trading strategy recommendation based on market data
        
        Args:
            market_data: Current market data and indicators
            current_positions: List of current open positions
            
        Returns:
            Strategy recommendation
        """
        system_prompt = """You are an expert algorithmic trading advisor.
Analyze market conditions and provide strategic recommendations.
Consider technical indicators, risk management, and market context."""
        
        positions_info = ""
        if current_positions:
            positions_info = f"\n\nCurrent Positions: {json.dumps(current_positions, indent=2)}"
        
        prompt = f"""Based on the following market data, provide a trading strategy recommendation:

Market Data:
{json.dumps(market_data, indent=2)}
{positions_info}

Provide your recommendation including:
1. Suggested action (buy/sell/hold)
2. Reasoning
3. Risk assessment
4. Entry/exit points if applicable
5. Position sizing suggestion"""
        
        response = self._generate(prompt, system_prompt, temperature=0.5)
        
        return {
            "recommendation": response,
            "timestamp": "now",
            "market_data": market_data
        }
    
    def explain_trade(
        self,
        trade_details: Dict[str, Any],
        strategy_name: str
    ) -> str:
        """
        Generate explanation for a trade decision
        
        Args:
            trade_details: Details of the trade
            strategy_name: Name of strategy that generated the trade
            
        Returns:
            Human-readable trade explanation
        """
        system_prompt = """You are a trading educator. 
Explain trading decisions in clear, simple language that helps traders understand the reasoning."""
        
        prompt = f"""Explain this trade decision:

Strategy: {strategy_name}
Trade Details:
{json.dumps(trade_details, indent=2)}

Provide a clear explanation of:
1. Why this trade was made
2. What signals triggered it
3. What the expected outcome is
4. What risks are involved"""
        
        response = self._generate(prompt, system_prompt, temperature=0.6)
        return response
    
    def analyze_market_conditions(
        self,
        market_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Comprehensive market conditions analysis
        
        Args:
            market_data: Market data including price, volume, indicators
            
        Returns:
            Market analysis report
        """
        system_prompt = """You are a market analyst.
Provide comprehensive analysis of current market conditions.
Focus on actionable insights and risk factors."""
        
        prompt = f"""Analyze current market conditions:

Market Data:
{json.dumps(market_data, indent=2)}

Provide analysis covering:
1. Market trend (bullish/bearish/sideways)
2. Volatility assessment
3. Support and resistance levels
4. Key risk factors
5. Trading opportunities"""
        
        response = self._generate(prompt, system_prompt, temperature=0.5)
        
        return {
            "analysis": response,
            "timestamp": "now",
            "market_data": market_data
        }
    
    def get_risk_assessment(
        self,
        portfolio: Dict[str, Any],
        proposed_trade: Dict[str, Any]
    ) -> str:
        """
        Assess risk of a proposed trade
        
        Args:
            portfolio: Current portfolio state
            proposed_trade: Details of proposed trade
            
        Returns:
            Risk assessment text
        """
        system_prompt = """You are a risk management expert.
Evaluate trading risk objectively and provide clear risk assessments."""
        
        prompt = f"""Assess the risk of this proposed trade:

Current Portfolio:
{json.dumps(portfolio, indent=2)}

Proposed Trade:
{json.dumps(proposed_trade, indent=2)}

Provide risk assessment including:
1. Risk level (low/medium/high)
2. Potential downside
3. Position sizing recommendation
4. Risk mitigation suggestions
5. Overall recommendation"""
        
        response = self._generate(prompt, system_prompt, temperature=0.3)
        return response
    
    def compare_strategies(
        self,
        strategies_performance: Dict[str, Dict[str, Any]]
    ) -> str:
        """
        Compare performance of multiple strategies
        
        Args:
            strategies_performance: Dictionary mapping strategy names to their performance metrics
            
        Returns:
            Comparison analysis
        """
        system_prompt = """You are a trading strategy analyst.
Compare strategies objectively based on their performance metrics."""
        
        prompt = f"""Compare these trading strategies:

{json.dumps(strategies_performance, indent=2)}

Provide comparison including:
1. Best performing strategy
2. Risk-adjusted performance ranking
3. Strengths and weaknesses of each
4. Recommendation for different market conditions
5. Overall conclusion"""
        
        response = self._generate(prompt, system_prompt, temperature=0.5)
        return response
    
    def generate_trading_insights(
        self,
        historical_trades: List[Dict[str, Any]]
    ) -> str:
        """
        Generate insights from historical trading data
        
        Args:
            historical_trades: List of past trades with their outcomes
            
        Returns:
            Trading insights and lessons learned
        """
        system_prompt = """You are a trading coach.
Analyze historical trades to provide actionable insights and lessons."""
        
        # Sample recent trades if too many
        recent_trades = historical_trades[-20:] if len(historical_trades) > 20 else historical_trades
        
        prompt = f"""Analyze these recent trades and provide insights:

{json.dumps(recent_trades, indent=2)}

Provide insights on:
1. Common patterns in winning trades
2. Common mistakes in losing trades
3. Timing and execution quality
4. Suggested improvements
5. Key lessons learned"""
        
        response = self._generate(prompt, system_prompt, temperature=0.6)
        return response


# Helper functions for easy access

def quick_sentiment_check(text: str, model: str = "llama2") -> str:
    """
    Quick sentiment check
    
    Args:
        text: Text to analyze
        model: Ollama model to use
        
    Returns:
        Sentiment string
    """
    analyzer = OllamaAnalyzer(model=model)
    result = analyzer.analyze_sentiment(text)
    return result.get('sentiment', 'neutral')


def quick_recommendation(market_data: Dict[str, Any], model: str = "llama2") -> str:
    """
    Quick trading recommendation
    
    Args:
        market_data: Current market data
        model: Ollama model to use
        
    Returns:
        Recommendation text
    """
    analyzer = OllamaAnalyzer(model=model)
    result = analyzer.get_strategy_recommendation(market_data)
    return result.get('recommendation', 'No recommendation available')

