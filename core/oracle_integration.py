"""Oracle Prediction Engine Integration for Jarvis.

Provides prediction capabilities across multiple domains:
- Economics (Dalio cycles, Solow growth, Taylor rule, IS-LM)
- Finance (Monte Carlo, Black-Sitterman, options, portfolio)
- Physics (coupled oscillators, Langevin dynamics, Hurst exponent)
- Geopolitics, Climate, Sentiment, Contagion, Demographics, ESG
- Technical analysis (RSI, MACD, Bollinger, SMA/EMA, etc.)

The OracleEngine is loaded lazily and cached as a singleton.
Prediction calls are delegated through this module.
"""
from core.helpers import safe_error_message
import logging
import threading
import time
from datetime import date, timedelta
from typing import Any, Optional

logger = logging.getLogger(__name__)

_oracle_engine = None
_oracle_lock = threading.Lock()


def get_oracle_engine():
    """Get or create the singleton Oracle prediction engine."""
    global _oracle_engine
    if _oracle_engine is None:
        with _oracle_lock:
            if _oracle_engine is None:
                try:
                    import sys
                    from pathlib import Path
                    oracle_path = Path.home() / "oracle"
                    if oracle_path.exists() and str(oracle_path / "src") not in sys.path:
                        sys.path.insert(0, str(oracle_path / "src"))
                    from oracle.prediction.engine import OracleEngine
                    _oracle_engine = OracleEngine()
                    logger.info("Oracle prediction engine initialized")
                except ImportError as e:
                    logger.warning(f"Oracle not available: {e}")
                    return None
                except Exception as e:
                    logger.error(f"Failed to initialize Oracle: {e}")
                    return None
    return _oracle_engine


def oracle_predict(
    domain: str = "all",
    target_date: Optional[str] = None,
    confidence: str = "high",
    context: Optional[dict] = None,
) -> dict[str, Any]:
    """Run Oracle prediction for a given domain and target date.

    Args:
        domain: Prediction domain - one of:
            'economic', 'geopolitical', 'climate', 'sentiment',
            'contagion', 'demographics', 'all'
        target_date: ISO date string (default: 1 year from today)
        confidence: 'low', 'medium', 'high', 'very_high'
        context: Optional context dict with domain-specific parameters

    Returns:
        Dict with predictions, confidence intervals, and domain analysis.
    """
    engine = get_oracle_engine()
    if engine is None:
        return {"error": "Oracle engine not available", "predictions": []}

    VALID_DOMAINS = {"economic", "economy", "economics", "geopolitical", "geopolitics", 
                     "climate", "environment", "sentiment", "market_sentiment", 
                     "contagion", "financial_contagion", "demographics", "population", "all"}
    if domain.lower() not in VALID_DOMAINS:
        return {"error": f"Invalid domain: {domain}. Use one of: economic, geopolitical, climate, sentiment, contagion, demographics, all", "predictions": []}

    if target_date is None:
        target = date.today() + timedelta(days=365)
    else:
        try:
            target = date.fromisoformat(target_date)
        except (ValueError, TypeError):
            return {"error": f"Invalid date format: {target_date}. Use YYYY-MM-DD", "predictions": []}

    from oracle.core.types import ConfidenceLevel
    confidence_map = {
        "low": ConfidenceLevel.LOW,
        "medium": ConfidenceLevel.MEDIUM,
        "high": ConfidenceLevel.HIGH,
        "very_high": ConfidenceLevel.VERY_HIGH,
    }
    conf = confidence_map.get(confidence.lower(), ConfidenceLevel.HIGH)

    context = context or {}
    domain_lower = domain.lower()

    try:
        if domain_lower in ("economic", "economy", "economics"):
            _configure_economic(engine, context)
        elif domain_lower in ("geopolitical", "geopolitics"):
            _configure_geopolitical(engine, context)
        elif domain_lower in ("climate", "environment"):
            _configure_climate(engine, context)
        elif domain_lower in ("sentiment", "market_sentiment"):
            _configure_sentiment(engine, context)
        elif domain_lower in ("contagion", "financial_contagion"):
            _configure_contagion(engine, context)
        elif domain_lower in ("demographics", "population"):
            _configure_demographics(engine, context)

        forecast = engine.predict(target, conf)

        return {
            "domain": domain,
            "target_date": target.isoformat(),
            "confidence": confidence,
            "predictions": [
                {
                    "variable": p.variable,
                    "value": round(p.value, 4) if isinstance(p.value, (int, float)) else p.value,
                    "lower_bound": round(p.lower_bound, 4) if isinstance(p.lower_bound, (int, float)) else None,
                    "upper_bound": round(p.upper_bound, 4) if isinstance(p.upper_bound, (int, float)) else None,
                    "model": p.model_name,
                    "domain": p.domain.value if hasattr(p.domain, 'value') else str(p.domain),
                }
                for p in forecast.predictions
            ],
            "combined_forecast": {
                "value": round(forecast.combined_value, 4) if hasattr(forecast, 'combined_value') and isinstance(forecast.combined_value, (int, float)) else None,
                "confidence_score": round(forecast.confidence_score, 4) if hasattr(forecast, 'confidence_score') else None,
            },
            "num_models": len(forecast.predictions),
        }
    except Exception as e:
        logger.error(f"Oracle prediction error: {e}")
        return {"error": safe_error_message(e), "predictions": []}


def oracle_diagnose(
    domain: str = "all",
    context: Optional[dict] = None,
) -> dict[str, Any]:
    """Run cross-domain interaction diagnostics.

    Identifies how different domains affect each other (6 interaction types).
    """
    engine = get_oracle_engine()
    if engine is None:
        return {"error": "Oracle engine not available", "interactions": []}

    context = context or {}
    try:
        if context:
            _apply_context(engine, context)
        result = engine.diagnose()
        return {
            "domain": domain,
            "interactions": [
                {
                    "from_domain": i.from_domain.value if hasattr(i.from_domain, 'value') else str(i.from_domain),
                    "to_domain": i.to_domain.value if hasattr(i.to_domain, 'value') else str(i.to_domain),
                    "interaction_type": i.interaction_type if hasattr(i, 'interaction_type') else "unknown",
                    "strength": round(i.strength, 4) if hasattr(i, 'strength') and isinstance(i.strength, (int, float)) else None,
                    "description": i.description if hasattr(i, 'description') else "",
                }
                for i in (result.interactions if hasattr(result, 'interactions') else [])
            ],
            "num_interactions": len(result.interactions) if hasattr(result, 'interactions') else 0,
        }
    except Exception as e:
        logger.error(f"Oracle diagnose error: {e}")
        return {"error": safe_error_message(e), "interactions": []}


def oracle_timeslider(
    target_date: Optional[str] = None,
    scenario: str = "baseline",
) -> dict[str, Any]:
    """Quick prediction scan across multiple time horizons."""
    engine = get_oracle_engine()
    if engine is None:
        return {"error": "Oracle engine not available", "forecasts": []}

    if target_date is None:
        target = date.today() + timedelta(days=365)
    else:
        try:
            target = date.fromisoformat(target_date)
        except (ValueError, TypeError):
            return {"error": f"Invalid date format: {target_date}. Use YYYY-MM-DD", "forecasts": []}

    try:
        from oracle.prediction.timeslider import TimeSlider
        slider = TimeSlider(engine)
        result = slider.quick_scan(target)

        return {
            "target_date": target.isoformat(),
            "scenario": scenario,
            "forecasts": {
                horizon: {
                    "value": round(data.get("value", 0), 4) if isinstance(data.get("value"), (int, float)) else data.get("value"),
                    "confidence": round(data.get("confidence", 0), 4) if isinstance(data.get("confidence"), (int, float)) else data.get("confidence"),
                }
                for horizon, data in (result if isinstance(result, dict) else {}).items()
            } if isinstance(result, dict) else {"raw": str(result)[:500]},
        }
    except Exception as e:
        logger.error(f"Oracle timeslider error: {e}")
        return {"error": safe_error_message(e), "forecasts": []}


def _apply_context(engine, context: dict) -> None:
    """Apply context parameters to the engine based on provided keys."""
    if any(k in context for k in ("gdp_growth_rate", "debt_to_gdp", "interest_rate",
                                     "inflation_rate", "unemployment_rate")):
        _configure_economic(engine, context)


def _configure_economic(engine, context: dict) -> None:
    """Configure economic domain predictions."""
    engine.set_economic_state(
        gdp_growth_rate=context.get("gdp_growth_rate", 0.02),
        debt_to_gdp=context.get("debt_to_gdp", 1.2),
        interest_rate=context.get("interest_rate", 0.045),
        money_supply_growth=context.get("money_supply_growth", 0.05),
        credit_growth=context.get("credit_growth", 0.04),
        productivity_growth=context.get("productivity_growth", 0.015),
        inflation_rate=context.get("inflation_rate", 0.03),
        unemployment_rate=context.get("unemployment_rate", 0.04),
    )


def _configure_geopolitical(engine, context: dict) -> None:
    """Configure geopolitical domain predictions."""
    engine.set_geopolitical_state(**{k: v for k, v in context.items()
                                      if k in ("military_tension", "trade_conflict", "diplomatic_instability",
                                               "alliance_strength", "resource_competition")})


def _configure_climate(engine, context: dict) -> None:
    """Configure climate domain predictions."""
    engine.set_climate_state(**{k: v for k, v in context.items()
                                  if k in ("temperature_anomaly", "co2_concentration", "sea_level_rise",
                                           "extreme_weather_index", "carbon_price")})


def _configure_sentiment(engine, context: dict) -> None:
    """Configure sentiment domain predictions."""
    engine.set_sentiment_data(**{k: v for k, v in context.items()
                                   if k in ("fear_greed_index", "social_volume", "news_sentiment",
                                            "institutional_flow", "retail_flow")})


def _configure_contagion(engine, context: dict) -> None:
    """Configure contagion domain predictions."""
    engine.set_contagion_state(**{k: v for k, v in context.items()
                                    if k in ("systemic_risk", "interbank_connectivity", "sovereign_risk",
                                             "corporate_defaults", "liquidity_stress")})


def oracle_portfolio(
    action: str = "optimize",
    assets: list[dict] | None = None,
    method: str = "mean_variance",
    target_return: float | None = None,
    risk_free_rate: float = 0.04,
) -> dict[str, Any]:
    """Portfolio optimization and analysis.

    Args:
        action: 'optimize', 'efficient_frontier', 'risk_parity', 'black_litterman'
        assets: List of dicts with 'symbol', 'weight', 'expected_return', 'risk' keys
        method: Optimization method
        target_return: Target portfolio return (for mean-variance)
        risk_free_rate: Risk-free rate for Sharpe calculations
    """
    if risk_free_rate < -0.5 or risk_free_rate > 1.0:
        return {"error": "risk_free_rate must be between -0.5 and 1.0"}
    if action not in ("optimize", "efficient_frontier", "risk_parity", "black_litterman"):
        return {"error": f"Unknown portfolio action: {action}. Use: optimize, efficient_frontier, risk_parity, black_litterman"}
    if assets and len(assets) > 50:
        return {"error": "Maximum 50 assets allowed"}
    if target_return is not None and not (-1.0 <= target_return <= 10.0):
        return {"error": "target_return must be between -1.0 and 10.0"}
    engine = get_oracle_engine()
    if engine is None:
        return {"error": "Oracle engine not available"}

    try:
        from oracle.optimization.portfolio import PortfolioOptimizer
        optimizer = PortfolioOptimizer()

        if action == "optimize":
            result = optimizer.optimize_portfolio(
                assets=assets or [],
                method=method,
                target_return=target_return,
                risk_free_rate=risk_free_rate,
            )
        elif action == "efficient_frontier":
            result = optimizer.efficient_frontier(
                n_points=50,
                assets=assets or [],
            )
        elif action == "risk_parity":
            result = optimizer.risk_parity(assets=assets or [])
        elif action == "black_litterman":
            result = optimizer.black_litterman(assets=assets or [])
        else:
            return {"error": f"Unknown portfolio action: {action}"}

        if hasattr(result, '__dict__'):
            return {k: v for k, v in result.__dict__.items()
                    if not k.startswith('_') and not callable(v)}
        return {"result": str(result)[:2000]}

    except ImportError as e:
        return {"error": f"Portfolio optimizer not available: {e}"}
    except Exception as e:
        logger.error(f"Oracle portfolio error: {e}")
        return {"error": "Portfolio optimization failed"}


def oracle_options(
    action: str = "price",
    spot: float = 100.0,
    strike: float = 100.0,
    time_to_expiry: float = 1.0,
    risk_free_rate: float = 0.04,
    volatility: float = 0.25,
    option_type: str = "call",
    dividend_yield: float = 0.0,
) -> dict[str, Any]:
    """Options pricing and Greeks calculation using Black-Scholes.

    Args:
        action: 'price', 'greeks', 'implied_volatility', 'chain', 'strategy'
        spot: Current underlying price
        strike: Strike price
        time_to_expiry: Time to expiry in years
        risk_free_rate: Risk-free interest rate
        volatility: Implied volatility
        option_type: 'call' or 'put'
        dividend_yield: Continuously compounded dividend yield
    """
    if action not in ("price", "greeks", "implied_volatility", "chain", "strategy"):
        return {"error": f"Unknown options action: {action}. Use: price, greeks, implied_volatility, chain"}
    if spot <= 0 or strike <= 0:
        return {"error": "spot and strike must be positive"}
    if time_to_expiry < 0:
        return {"error": "time_to_expiry must be non-negative"}
    if volatility < 0:
        return {"error": "volatility must be non-negative"}
    if option_type not in ("call", "put"):
        return {"error": "option_type must be 'call' or 'put'"}
    if dividend_yield < -0.5 or dividend_yield > 1.0:
        return {"error": "dividend_yield must be between -0.5 and 1.0"}
    if risk_free_rate < -0.5 or risk_free_rate > 1.0:
        return {"error": "risk_free_rate must be between -0.5 and 1.0"}
    try:
        from oracle.optimization.options import OptionsPricer
        pricer = OptionsPricer()

        if action == "price":
            price = pricer.black_scholes_price(
                spot=spot, strike=strike, time_to_expiry=time_to_expiry,
                risk_free_rate=risk_free_rate, volatility=volatility,
                option_type=option_type, dividend_yield=dividend_yield,
            )
            return {"action": "price", "option_type": option_type, "price": round(price, 4),
                    "spot": spot, "strike": strike, "time_to_expiry": time_to_expiry,
                    "volatility": volatility, "risk_free_rate": risk_free_rate}

        elif action == "greeks":
            greeks = pricer.calculate_greeks(
                spot=spot, strike=strike, time_to_expiry=time_to_expiry,
                risk_free_rate=risk_free_rate, volatility=volatility,
                option_type=option_type, dividend_yield=dividend_yield,
            )
            if hasattr(greeks, '__dict__'):
                return {k: round(v, 6) if isinstance(v, (int, float)) else v
                        for k, v in greeks.__dict__.items() if not k.startswith('_')}
            return {"greeks": str(greeks)[:1000]}

        elif action == "implied_volatility":
            iv = pricer.implied_volatility(
                spot=spot, strike=strike, time_to_expiry=time_to_expiry,
                risk_free_rate=risk_free_rate, market_price=spot,
                option_type=option_type, dividend_yield=dividend_yield,
            )
            return {"implied_volatility": round(iv, 6) if isinstance(iv, (int, float)) else str(iv)}

        elif action == "chain":
            chain = pricer.option_chain(
                spot=spot, time_to_expiry=time_to_expiry,
                risk_free_rate=risk_free_rate, volatility=volatility,
                dividend_yield=dividend_yield,
            )
            if isinstance(chain, list):
                return {"chain": chain[:50]}
            return {"chain": str(chain)[:2000]}

        else:
            return {"error": f"Unknown options action: {action}. Use: price, greeks, implied_volatility, chain"}

    except ImportError as e:
        return {"error": f"Options pricer not available: {e}"}
    except Exception as e:
        logger.error(f"Oracle options error: {e}")
        return {"error": "Options calculation failed"}


def oracle_technical_analysis(
    symbol: str | None = None,
    data: list[float] | None = None,
    indicators: list[str] | None = None,
) -> dict[str, Any]:
    """Technical analysis calculation.

    Args:
        symbol: Ticker symbol (uses real data if available)
        data: Price data array (if no symbol)
        indicators: List of indicators to calculate (rsi, macd, bollinger, sma, ema, stochastic, atr, adx)
    """
    try:
        import numpy as np
        from oracle.patterns.technical_analysis import TechnicalAnalysis
        ta = TechnicalAnalysis()
        prices = np.array(data) if data else np.array([100 + i * 0.5 for i in range(100)])
        indicators = indicators or ["rsi", "macd", "bollinger"]

        results = {}
        if "rsi" in indicators:
            rsi = ta.rsi(prices)
            results["rsi"] = rsi if isinstance(rsi, (int, float)) else float(rsi[-1]) if hasattr(rsi, '__len__') else None
        if "macd" in indicators:
            macd_result = ta.macd(prices)
            if isinstance(macd_result, dict):
                results["macd"] = {k: float(v[-1]) if hasattr(v, '__len__') and len(v) > 0 else v
                                    for k, v in macd_result.items() if not k.startswith('_')}
            elif hasattr(macd_result, '__dict__'):
                results["macd"] = {k: round(float(v[-1]) if hasattr(v, '__len__') and len(v) > 0 else float(v), 4)
                                    for k, v in macd_result.__dict__.items() if not k.startswith('_') and not callable(v)}
        if "bollinger" in indicators:
            bb = ta.bollinger_bands(prices)
            if isinstance(bb, dict):
                results["bollinger"] = {k: round(float(v[-1]) if hasattr(v, '__len__') and len(v) > 0 else float(v), 4)
                                         for k, v in bb.items() if not k.startswith('_') and not callable(v)}
            elif hasattr(bb, '__dict__'):
                results["bollinger"] = {k: round(float(v[-1]) if hasattr(v, '__len__') and len(v) > 0 else float(v), 4)
                                         for k, v in bb.__dict__.items() if not k.startswith('_') and not callable(v)}
        if "sma" in indicators:
            sma = ta.sma(prices, period=20)
            results["sma_20"] = round(float(sma[-1]), 4) if hasattr(sma, '__len__') and len(sma) > 0 else None
        if "ema" in indicators:
            ema = ta.ema(prices, period=20)
            results["ema_20"] = round(float(ema[-1]), 4) if hasattr(ema, '__len__') and len(ema) > 0 else None

        return {"symbol": symbol, "indicators": results, "data_points": len(prices)}

    except ImportError as e:
        return {"error": f"Technical analysis not available: {e}"}
    except Exception as e:
        logger.error(f"Oracle TA error: {e}")
        return {"error": "Technical analysis failed"}


def oracle_market_data(
    symbols: list[str] | None = None,
    period: str = "1mo",
) -> dict[str, Any]:
    """Fetch real market data using yfinance.

    Args:
        symbols: List of ticker symbols (default: ['SPY', 'QQQ', 'IWM', 'GLD', 'TLT'])
        period: Data period ('1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y')
    """
    try:
        from oracle.ingestion.real_data import RealDataFetcher
        fetcher = RealDataFetcher()
        symbols = symbols or ["SPY", "QQQ", "IWM", "GLD", "TLT"]
        results = {}
        for symbol in symbols:
            try:
                data = fetcher.fetch(symbol, period=period)
                if data is not None and len(data) > 0:
                    latest = data.iloc[-1] if hasattr(data, 'iloc') else data[-1]
                    results[symbol] = {
                        "close": round(float(latest['Close']), 2) if 'Close' in latest else None,
                        "volume": int(latest['Volume']) if 'Volume' in latest else None,
                        "data_points": len(data),
                    }
            except Exception:
                results[symbol] = {"error": "Data fetch failed"}
        return {"symbols": results, "period": period}

    except ImportError as e:
        return {"error": f"Market data not available: {e}"}
    except Exception as e:
        logger.error(f"Oracle market data error: {e}")
        return {"error": "Market data fetch failed"}


def _configure_demographics(engine, context: dict) -> None:
    """Configure demographics domain predictions."""
    engine.set_demographics_state(**{k: v for k, v in context.items()
                                      if k in ("population_growth", "aging_ratio", "urbanization_rate",
                                               "dependency_ratio", "migration_rate")})


def oracle_scenario(
    domain: str = "all",
    scenario_type: str = "baseline",
    target_date: Optional[str] = None,
    context: Optional[dict] = None,
) -> dict[str, Any]:
    """Run scenario analysis for different futures.

    Args:
        domain: Prediction domain
        scenario_type: One of: baseline, bull, bear, crash, recovery, high_vol, low_vol
        target_date: ISO date string
        context: Optional context parameters

    Returns:
        Dict with scenario predictions and comparisons.
    """
    VALID_SCENARIOS = {"baseline", "bull", "bear", "crash", "recovery", "high_vol", "low_vol"}
    if scenario_type not in VALID_SCENARIOS:
        return {"error": f"Invalid scenario_type: {scenario_type}. Use: {', '.join(sorted(VALID_SCENARIOS))}"}
    
    engine = get_oracle_engine()
    if engine is None:
        return {"error": "Oracle engine not available"}

    context = context or {}
    try:
        from oracle.prediction.timeslider import TimeSlider
        slider = TimeSlider(engine)
        if target_date is None:
            from datetime import date, timedelta
            target = date.today() + timedelta(days=365)
        else:
            target = date.fromisoformat(target_date)

        result = slider.run_scenario(str(target), scenario_type)
        return {
            "scenario_type": scenario_type,
            "target_date": str(target),
            "domain": domain,
            "predictions": result.get("predictions", []),
            "comparison": result.get("comparison", {}),
        }
    except Exception as e:
        logger.error(f"Oracle scenario error: {e}")
        return {"error": safe_error_message(e)}


def oracle_backtest(
    domain: str = "all",
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    method: str = "walk_forward",
) -> dict[str, Any]:
    """Backtest Oracle predictions against historical data.

    Args:
        domain: Prediction domain
        start_date: ISO date string (default: 2 years ago)
        end_date: ISO date string (default: today)
        method: Backtest method - 'walk_forward', 'model_comparison', 'ga_optimize'

    Returns:
        Dict with backtest results including accuracy metrics.
    """
    VALID_METHODS = {"walk_forward", "model_comparison", "ga_optimize"}
    if method not in VALID_METHODS:
        return {"error": f"Invalid backtest method: {method}. Use: {', '.join(sorted(VALID_METHODS))}"}
    
    if start_date is not None:
        try:
            from datetime import date as _date
            _date.fromisoformat(start_date)
        except (ValueError, TypeError):
            return {"error": f"Invalid start_date format: {start_date}. Use YYYY-MM-DD"}
    
    if end_date is not None:
        try:
            from datetime import date as _date
            _date.fromisoformat(end_date)
        except (ValueError, TypeError):
            return {"error": f"Invalid end_date format: {end_date}. Use YYYY-MM-DD"}
    
    engine = get_oracle_engine()
    if engine is None:
        return {"error": "Oracle engine not available"}

    try:
        from datetime import date, timedelta
        from oracle.backtesting.walk_forward import WalkForward
        from oracle.backtesting.model_comparison import ModelComparison
        from oracle.backtesting.ga_optimizer import WeightOptimizer

        if start_date is None:
            start = date.today() - timedelta(days=730)
        else:
            start = date.fromisoformat(start_date)

        if end_date is None:
            end = date.today()
        else:
            end = date.fromisoformat(end_date)

        if method == "walk_forward":
            wf = WalkForward(engine)
            results = wf.run(str(start), str(end))
        elif method == "model_comparison":
            mc = ModelComparison(engine)
            results = mc.compare(str(start), str(end))
        elif method == "ga_optimize":
            wo = WeightOptimizer(engine)
            results = wo.optimize(str(start), str(end))
        else:
            return {"error": f"Unknown backtest method: {method}"}

        return {
            "method": method,
            "start_date": str(start),
            "end_date": str(end),
            "results": results,
        }
    except Exception as e:
        logger.error(f"Oracle backtest error: {e}")
        return {"error": safe_error_message(e)}


def oracle_esg(
    symbol: Optional[str] = None,
    sector: Optional[str] = None,
    scenario: str = "baseline",
) -> dict[str, Any]:
    """ESG risk assessment and sector benchmarking.

    Args:
        symbol: Company/ETF symbol (optional)
        sector: Sector name for benchmarking (optional)
        scenario: Scenario type - 'baseline', 'improved', 'degraded'

    Returns:
        Dict with ESG scores, risk ratings, and recommendations.
    """
    VALID_SCENARIOS = {"baseline", "improved", "degraded"}
    if scenario not in VALID_SCENARIOS:
        return {"error": f"Invalid ESG scenario: {scenario}. Use: {', '.join(sorted(VALID_SCENARIOS))}"}
    if symbol is not None and (not isinstance(symbol, str) or len(symbol) > 10):
        return {"error": "symbol must be a string of at most 10 characters"}
    if sector is not None and (not isinstance(sector, str) or len(sector) > 50):
        return {"error": "sector must be a string of at most 50 characters"}
    
    engine = get_oracle_engine()
    if engine is None:
        return {"error": "Oracle engine not available"}

    try:
        from oracle.domains.esg import ESGForecaster
        esg = ESGForecaster()

        if symbol:
            result = esg.assess(symbol, scenario=scenario)
        elif sector:
            result = esg.benchmark_sector(sector, scenario=scenario)
        else:
            result = esg.get_market_overview(scenario=scenario)

        return {
            "symbol": symbol,
            "sector": sector,
            "scenario": scenario,
            **result,
        }
    except Exception as e:
        logger.error(f"Oracle ESG error: {e}")
        return {"error": safe_error_message(e)}


def oracle_yield_curve(
    method: str = "nelson_siegel",
    scenario: str = "baseline",
) -> dict[str, Any]:
    """Yield curve analysis and scenario testing.

    Args:
        method: Fitting method - 'nelson_siegel', 'nelson_siegel_svensson', 'fit'
        scenario: Scenario type - 'baseline', 'steepen', 'flatten', 'invert'

    Returns:
        Dict with yield curve parameters, spot rates, and forward rates.
    """
    VALID_METHODS = {"nelson_siegel", "nelson_siegel_svensson", "fit"}
    VALID_SCENARIOS = {"baseline", "steepen", "flatten", "invert"}
    if method not in VALID_METHODS:
        return {"error": f"Invalid yield curve method: {method}. Use: {', '.join(sorted(VALID_METHODS))}"}
    if scenario not in VALID_SCENARIOS:
        return {"error": f"Invalid scenario: {scenario}. Use: {', '.join(sorted(VALID_SCENARIOS))}"}
    
    engine = get_oracle_engine()
    if engine is None:
        return {"error": "Oracle engine not available"}

    try:
        from oracle.domains.yield_curve import YieldCurveAnalyzer
        yc = YieldCurveAnalyzer()

        if method == "nelson_siegel":
            result = yc.nelson_siegel()
        elif method == "nelson_siegel_svensson":
            result = yc.nelson_siegel_svensson()
        elif method == "fit":
            result = yc.fit_to_observed()
        else:
            return {"error": f"Unknown yield curve method: {method}"}

        if scenario != "baseline":
            scenario_result = yc.scenario_analysis(scenario)
            result["scenario"] = scenario_result

        return {
            "method": method,
            "scenario": scenario,
            **result,
        }
    except Exception as e:
        logger.error(f"Oracle yield curve error: {e}")
        return {"error": safe_error_message(e)}


def oracle_risk_decomposition(
    portfolio: Optional[list] = None,
    method: str = "component_var",
) -> dict[str, Any]:
    """Risk decomposition and factor attribution.

    Args:
        portfolio: List of asset symbols (default: standard portfolio)
        method: Decomposition method - 'component_var', 'factor_attribution', 'risk_budget', 'tracking_error', 'stress_decomposition'

    Returns:
        Dict with risk metrics, factor exposures, and stress test results.
    """
    VALID_METHODS = {"component_var", "factor_attribution", "risk_budget", "tracking_error", "stress_decomposition"}
    if method not in VALID_METHODS:
        return {"error": f"Invalid risk decomposition method: {method}. Use: {', '.join(sorted(VALID_METHODS))}"}
    
    if portfolio is not None:
        if not isinstance(portfolio, list):
            return {"error": "portfolio must be a list of ticker symbols"}
        if len(portfolio) > 50:
            return {"error": "portfolio must have at most 50 symbols"}
        for sym in portfolio:
            if not isinstance(sym, str) or len(sym) > 10:
                return {"error": f"Invalid portfolio symbol: {sym}"}
    
    engine = get_oracle_engine()
    if engine is None:
        return {"error": "Oracle engine not available"}

    try:
        from oracle.domains.risk_decomposition import RiskDecomposer
        rd = RiskDecomposer()

        if portfolio is None:
            portfolio = ["SPY", "QQQ", "IWM", "EFA", "AGG", "GLD"]

        if method == "component_var":
            result = rd.component_var(portfolio)
        elif method == "factor_attribution":
            result = rd.factor_attribution(portfolio)
        elif method == "risk_budget":
            result = rd.risk_budget(portfolio)
        elif method == "tracking_error":
            result = rd.tracking_error(portfolio)
        elif method == "stress_decomposition":
            result = rd.stress_decomposition(portfolio)
        else:
            return {"error": f"Unknown risk decomposition method: {method}"}

        return {
            "portfolio": portfolio,
            "method": method,
            **result,
        }
    except Exception as e:
        logger.error(f"Oracle risk decomposition error: {e}")
        return {"error": safe_error_message(e)}


def oracle_valuation(
    symbol: str,
    method: str = "dcf",
    scenario: str = "baseline",
) -> dict[str, Any]:
    """Company/asset valuation using multiple methods.

    Args:
        symbol: Company/asset symbol
        method: Valuation method - 'ddm', 'dcf', 'residual_income', 'eva', 'sensitivity'
        scenario: Scenario type - 'baseline', 'bull', 'bear'

    Returns:
        Dict with valuation metrics, fair value estimates, and sensitivity analysis.
    """
    VALID_METHODS = {"ddm", "dcf", "residual_income", "eva", "sensitivity"}
    VALID_SCENARIOS = {"baseline", "bull", "bear"}
    if method not in VALID_METHODS:
        return {"error": f"Invalid valuation method: {method}. Use: {', '.join(sorted(VALID_METHODS))}"}
    if scenario not in VALID_SCENARIOS:
        return {"error": f"Invalid scenario: {scenario}. Use: {', '.join(sorted(VALID_SCENARIOS))}"}
    if not symbol or not isinstance(symbol, str) or len(symbol) > 10:
        return {"error": "symbol must be a non-empty string of at most 10 characters"}
    
    engine = get_oracle_engine()
    if engine is None:
        return {"error": "Oracle engine not available"}

    try:
        from oracle.domains.valuation import ValuationEngine
        ve = ValuationEngine()

        if method == "ddm":
            result = ve.ddm(symbol, scenario=scenario)
        elif method == "dcf":
            result = ve.dcf(symbol, scenario=scenario)
        elif method == "residual_income":
            result = ve.residual_income(symbol, scenario=scenario)
        elif method == "eva":
            result = ve.eva(symbol, scenario=scenario)
        elif method == "sensitivity":
            result = ve.sensitivity_analysis(symbol)
        else:
            return {"error": f"Unknown valuation method: {method}"}

        return {
            "symbol": symbol,
            "method": method,
            "scenario": scenario,
            **result,
        }
    except Exception as e:
        logger.error(f"Oracle valuation error: {e}")
        return {"error": safe_error_message(e)}


def oracle_predict_to_knowledge_graph(
    domain: str = "all",
    target_date: Optional[str] = None,
    confidence: str = "medium",
    context: Optional[dict] = None,
) -> dict[str, Any]:
    """Run Oracle prediction and store insights in Jarvis knowledge graph.

    Bridges Oracle's multi-domain predictions into the knowledge graph,
    creating INSI GHT, CONCEPT, and PREDICTION nodes with cross-domain edges.

    Returns:
        Dict with prediction results and knowledge graph integration status.
    """
    result = oracle_predict(
        domain=domain,
        target_date=target_date,
        confidence=confidence,
        context=context,
    )

    if "error" in result and "predictions" not in result:
        return {**result, "kg_integrated": False}

    try:
        from core.knowledge_graph import get_knowledge_graph
        kg = get_knowledge_graph()

        integrated_nodes = []
        predictions = result.get("predictions", [])

        for pred in predictions[:10]:
            variable = pred.get("variable", "unknown")
            value = pred.get("value")
            model = pred.get("model", "unknown")
            pred_domain = pred.get("domain", domain)

            insight_content = (
                f"Oracle prediction ({pred_domain}): {variable} = {value} "
                f"[model: {model}, confidence: {confidence}]"
            )

            node_id = kg.add_insight(
                insight_id=f"oracle_{domain}_{variable}_{model}",
                content=insight_content,
                category=f"oracle_{pred_domain}",
                strategy=model,
            )
            integrated_nodes.append(node_id)

            concept_node = kg.add_node(variable, "CONCEPT", {"source": "oracle", "domain": pred_domain})
            kg.add_edge(node_id, concept_node, "REFERENCES_CONCEPT", 0.6)

        if len(predictions) > 1:
            for i in range(min(len(integrated_nodes) - 1, 5)):
                kg.add_edge(integrated_nodes[i], integrated_nodes[i + 1], "RELATED_TO", 0.4)

        cross_domain_concepts = set()
        for pred in predictions:
            d = pred.get("domain", domain)
            if d != domain:
                cross_domain_concepts.add(d)

        for cd in cross_domain_concepts:
            cd_node = kg.add_node(f"cross_domain_{cd}", "CONCEPT", {"source": "oracle_cross_domain"})
            for nid in integrated_nodes[:3]:
                kg.add_edge(nid, cd_node, "INFLUENCES", 0.3)

        result["kg_integrated"] = True
        result["kg_nodes_created"] = len(integrated_nodes)
        return result

    except Exception as e:
        logger.warning(f"Oracle-KG integration failed: {e}")
        result["kg_integrated"] = False
        result["kg_error"] = safe_error_message(e)
        return result


def oracle_diagnose_to_knowledge_graph(
    domain: str = "all",
    context: Optional[dict] = None,
) -> dict[str, Any]:
    """Run Oracle cross-domain diagnostics and store in knowledge graph.

    Creates INTERACTION nodes representing cross-domain influences
    discovered by the Oracle engine.

    Returns:
        Dict with diagnostic results and knowledge graph integration status.
    """
    result = oracle_diagnose(domain=domain, context=context)

    if "error" in result and "interactions" not in result:
        return {**result, "kg_integrated": False}

    try:
        from core.knowledge_graph import get_knowledge_graph
        kg = get_knowledge_graph()

        interactions = result.get("interactions", [])
        integrated_edges = 0

        for interaction in interactions[:15]:
            from_domain = interaction.get("from_domain", "unknown")
            to_domain = interaction.get("to_domain", "unknown")
            itype = interaction.get("interaction_type", "unknown")
            strength = interaction.get("strength", 0.5)
            desc = interaction.get("description", "")

            from_node = kg.add_node(
                f"oracle_domain_{from_domain}",
                "CONCEPT",
                {"source": "oracle", "domain": from_domain},
            )
            to_node = kg.add_node(
                f"oracle_domain_{to_domain}",
                "CONCEPT",
                {"source": "oracle", "domain": to_domain},
            )

            edge_type = f"CROSS_DOMAIN_{itype}".upper()[:30]
            kg.add_edge(from_node, to_node, edge_type, min(strength, 1.0))
            integrated_edges += 1

            if desc:
                kg.add_insight(
                    insight_id=f"oracle_interaction_{from_domain}_{to_domain}",
                    content=f"Cross-domain interaction: {from_domain} → {to_domain} ({itype}): {desc}",
                    category=f"oracle_interaction_{itype}",
                )

        result["kg_integrated"] = True
        result["kg_edges_created"] = integrated_edges
        return result

    except Exception as e:
        logger.warning(f"Oracle-KG diagnose integration failed: {e}")
        result["kg_integrated"] = False
        result["kg_error"] = safe_error_message(e)
        return result


_ORACLE_INSIGHT_CACHE: dict[str, tuple[list[dict], float]] = {}
_ORACLE_INSIGHT_CACHE_TTL: float = 3600.0  # 1 hour


def oracle_auto_insights(
    domains: list[str] | None = None,
    force_refresh: bool = False,
) -> dict[str, Any]:
    """Automatically extract and store key Oracle insights across domains.

    Runs cross-domain predictions and diagnostics, extracting the most impactful
    insights and storing them in the knowledge graph. Uses a 1-hour cache
    to avoid excessive Oracle queries.

    Args:
        domains: List of domains to analyze (default: all key domains)
        force_refresh: Force refresh even if cache is still valid

    Returns:
        Dict with extracted insights and KG integration status.
    """
    import time

    if domains is None:
        domains = ["economic", "geopolitical", "climate"]

    now = time.time()
    cache_key = ",".join(sorted(domains))

    if not force_refresh and cache_key in _ORACLE_INSIGHT_CACHE:
        cached_insights, cached_time = _ORACLE_INSIGHT_CACHE[cache_key]
        if now - cached_time < _ORACLE_INSIGHT_CACHE_TTL:
            return {
                "insights": cached_insights,
                "domains": domains,
                "cached": True,
                "cache_age_seconds": round(now - cached_time),
            }

    all_insights = []

    for domain in domains:
        try:
            result = oracle_predict(
                domain=domain,
                confidence="low",
            )
            if "predictions" not in result or "error" in result:
                continue

            predictions = result.get("predictions", [])[:3]
            for pred in predictions:
                var = pred.get("variable", "")
                val = pred.get("value")
                lower = pred.get("lower_bound")
                upper = pred.get("upper_bound")
                confidence = pred.get("confidence_score") if pred.get("confidence_score") else None

                if var and val is not None:
                    insight = {
                        "variable": var,
                        "value": val,
                        "domain": domain,
                        "range": f"[{lower}, {upper}]" if lower is not None and upper is not None else None,
                        "confidence": confidence,
                        "description": f"{domain.title()} prediction: {var} = {val}"
                                       + (f" (range: [{lower}, {upper}])" if lower is not None and upper is not None else ""),
                    }
                    all_insights.append(insight)
        except Exception as e:
            logger.debug(f"Oracle auto-insight for {domain} failed: {e}")
            continue

    try:
        result = oracle_diagnose(domain="all")
        if "interactions" in result and result.get("interactions"):
            interactions = result["interactions"][:5]
            for interaction in interactions:
                from_d = interaction.get("from_domain", "")
                to_d = interaction.get("to_domain", "")
                strength = interaction.get("strength", 0)
                itype = interaction.get("interaction_type", "unknown")
                desc = interaction.get("description", "")

                if from_d and to_d:
                    all_insights.append({
                        "variable": f"{from_d}→{to_d}",
                        "value": strength,
                        "domain": "cross_domain",
                        "range": None,
                        "confidence": None,
                        "description": f"Cross-domain: {desc}" if desc else f"{from_d} affects {to_d} ({itype}, strength={strength:.2f})",
                    })
    except Exception as e:
        logger.debug(f"Oracle auto-insight diagnose failed: {e}")

    _ORACLE_INSIGHT_CACHE[cache_key] = (all_insights, now)

    return {
        "insights": all_insights,
        "domains": domains,
        "cached": False,
        "total_insights": len(all_insights),
    }


class OracleCircuitBreaker:
    """Circuit breaker for Oracle prediction calls.

    Prevents cascading failures when Oracle is slow or unavailable.
    States: CLOSED (normal) → OPEN (failing) → HALF_OPEN (testing recovery).
    """

    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        half_open_max: int = 2,
    ):
        self._failure_threshold = failure_threshold
        self._recovery_timeout = recovery_timeout
        self._half_open_max = half_open_max
        self._state = self.CLOSED
        self._failure_count = 0
        self._success_count = 0
        # Monotonic: wall-clock steps (NTP/manual) must not pin the breaker
        # open or trip recovery early.
        self._last_failure_time: float = 0.0
        self._half_open_trials = 0
        self._lock = threading.Lock()
        self._total_calls = 0
        self._total_failures = 0

    @property
    def state(self) -> str:
        with self._lock:
            if self._state == self.OPEN:
                if time.monotonic() - self._last_failure_time >= self._recovery_timeout:
                    self._state = self.HALF_OPEN
                    self._half_open_trials = 0
            return self._state

    def allow_call(self) -> bool:
        with self._lock:
            if self._state == self.CLOSED:
                return True
            if self._state == self.HALF_OPEN:
                if self._half_open_trials < self._half_open_max:
                    self._half_open_trials += 1
                    return True
                return False
            if self._state == self.OPEN:
                if time.monotonic() - self._last_failure_time >= self._recovery_timeout:
                    self._state = self.HALF_OPEN
                    self._half_open_trials = 1
                    return True
            return False

    def record_success(self) -> None:
        with self._lock:
            self._total_calls += 1
            if self._state == self.HALF_OPEN:
                self._success_count += 1
                if self._success_count >= self._half_open_max:
                    self._state = self.CLOSED
                    self._failure_count = 0
                    self._success_count = 0
            else:
                self._failure_count = max(0, self._failure_count - 1)

    def record_failure(self) -> None:
        with self._lock:
            self._total_calls += 1
            self._total_failures += 1
            self._failure_count += 1
            self._last_failure_time = time.monotonic()
            if self._state == self.HALF_OPEN:
                self._state = self.OPEN
            elif self._failure_count >= self._failure_threshold:
                self._state = self.OPEN

    def reset(self) -> None:
        with self._lock:
            self._state = self.CLOSED
            self._failure_count = 0
            self._success_count = 0
            self._half_open_trials = 0

    def get_status(self) -> dict[str, Any]:
        return {
            "state": self.state,
            "failure_count": self._failure_count,
            "success_count": self._success_count,
            "threshold": self._failure_threshold,
            "recovery_timeout": self._recovery_timeout,
            "total_calls": self._total_calls,
            "total_failures": self._total_failures,
        }


_oracle_circuit_breaker: OracleCircuitBreaker | None = None
_cb_lock = threading.Lock()


def get_oracle_circuit_breaker() -> OracleCircuitBreaker:
    global _oracle_circuit_breaker
    if _oracle_circuit_breaker is None:
        with _cb_lock:
            if _oracle_circuit_breaker is None:
                _oracle_circuit_breaker = OracleCircuitBreaker()
    return _oracle_circuit_breaker


class OracleCallCache:
    """TTL-based cache for Oracle prediction results.

    Caches prediction results by domain + params hash with configurable TTL.
    Thread-safe with automatic expiry and size bounds.
    """

    def __init__(self, ttl: float = 600.0, max_size: int = 200):
        self._ttl = ttl
        self._max_size = max_size
        self._cache: dict[str, tuple[Any, float]] = {}
        self._lock = threading.Lock()
        self._hits = 0
        self._misses = 0

    def get(self, key: str) -> Any | None:
        with self._lock:
            if key in self._cache:
                value, timestamp = self._cache[key]
                if time.time() - timestamp < self._ttl:
                    self._hits += 1
                    return value
                del self._cache[key]
            self._misses += 1
            return None

    def set(self, key: str, value: Any) -> None:
        with self._lock:
            self._cache[key] = (value, time.time())
            self._evict()

    def invalidate(self, key: str | None = None) -> None:
        with self._lock:
            if key is not None:
                self._cache.pop(key, None)
            else:
                self._cache.clear()

    def _evict(self) -> None:
        if len(self._cache) <= self._max_size:
            return
        now = time.time()
        expired = [
            k for k, (_, ts) in self._cache.items()
            if now - ts >= self._ttl
        ]
        for k in expired:
            del self._cache[k]
        if len(self._cache) > self._max_size:
            sorted_keys = sorted(
                self._cache.items(),
                key=lambda x: x[1][1],
            )
            for k, _ in sorted_keys[:len(self._cache) - self._max_size]:
                del self._cache[k]

    def stats(self) -> dict[str, Any]:
        with self._lock:
            total = self._hits + self._misses
            return {
                "size": len(self._cache),
                "max_size": self._max_size,
                "ttl": self._ttl,
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": round(self._hits / total, 4) if total > 0 else 0.0,
            }


_oracle_call_cache: OracleCallCache | None = None
_cc_lock = threading.Lock()


def get_oracle_call_cache() -> OracleCallCache:
    global _oracle_call_cache
    if _oracle_call_cache is None:
        with _cc_lock:
            if _oracle_call_cache is None:
                _oracle_call_cache = OracleCallCache()
    return _oracle_call_cache


def oracle_retry_call(
    func: callable,
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 30.0,
    backoff_factor: float = 2.0,
) -> Any:
    """Execute an Oracle call with exponential backoff and circuit breaker.

    Args:
        func: Callable to execute
        max_retries: Maximum retry attempts
        base_delay: Initial delay in seconds
        max_delay: Maximum delay between retries
        backoff_factor: Multiplier for each retry

    Returns:
        Result of func()

    Raises:
        Exception: If all retries exhausted or circuit breaker is open
    """
    cb = get_oracle_circuit_breaker()

    if not cb.allow_call():
        raise RuntimeError(f"Oracle circuit breaker is {cb.state}, call rejected")

    delay = base_delay
    last_error = None

    for attempt in range(max_retries + 1):
        try:
            result = func()
            cb.record_success()
            return result
        except Exception as e:
            last_error = e
            cb.record_failure()

            if not cb.allow_call():
                logger.warning(f"Oracle circuit breaker opened after {attempt + 1} attempts")
                break

            if attempt < max_retries:
                import random
                jitter = random.uniform(0, delay * 0.1)
                sleep_time = min(delay + jitter, max_delay)
                logger.debug(f"Oracle call failed (attempt {attempt + 1}/{max_retries + 1}), "
                           f"retrying in {sleep_time:.1f}s: {e}")
                time.sleep(sleep_time)
                delay = min(delay * backoff_factor, max_delay)

    if last_error is not None:
        raise last_error
    raise RuntimeError("Oracle call failed: circuit breaker open")


def oracle_predict_resilient(
    domain: str = "all",
    target_date: Optional[str] = None,
    confidence: str = "high",
    context: Optional[dict] = None,
) -> dict[str, Any]:
    """Oracle prediction with circuit breaker, caching, and retry.

    Wraps oracle_predict with resilience features:
    - Circuit breaker prevents cascading failures
    - TTL cache avoids redundant Oracle queries
    - Exponential backoff retry on transient failures
    """
    import hashlib
    import json

    cache = get_oracle_call_cache()
    cache_key = hashlib.sha256(
        json.dumps({
            "domain": domain,
            "target_date": target_date,
            "confidence": confidence,
            "context": context,
        }, sort_keys=True, default=str).encode()
    ).hexdigest()

    cached = cache.get(cache_key)
    if cached is not None:
        return {**cached, "from_cache": True}

    try:
        result = oracle_retry_call(
            lambda: oracle_predict(domain, target_date, confidence, context)
        )
        cache.set(cache_key, result)
        return {**result, "from_cache": False}
    except RuntimeError as e:
        return {"error": safe_error_message(e), "predictions": [], "circuit_breaker": cb_state()}
    except Exception as e:
        return {"error": safe_error_message(e), "predictions": []}


def oracle_diagnose_resilient(
    domain: str = "all",
    context: Optional[dict] = None,
) -> dict[str, Any]:
    """Oracle diagnostics with circuit breaker and retry."""
    try:
        result = oracle_retry_call(
            lambda: oracle_diagnose(domain, context)
        )
        return result
    except RuntimeError as e:
        return {"error": safe_error_message(e), "interactions": [], "circuit_breaker": cb_state()}
    except Exception as e:
        return {"error": safe_error_message(e), "interactions": []}


def cb_state() -> dict[str, Any]:
    """Get current Oracle circuit breaker state."""
    return get_oracle_circuit_breaker().get_status()


def cache_stats() -> dict[str, Any]:
    """Get Oracle call cache statistics."""
    return get_oracle_call_cache().stats()