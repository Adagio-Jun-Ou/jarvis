"""Oracle Prediction Engine Tool for Jarvis.

Exposes Oracle's multi-domain prediction capabilities as a Jarvis tool.
Supports: predict, diagnose, timeslider, scenario analysis.
"""
import asyncio
import logging
from typing import Any

from .base import Tool
from core.helpers import safe_error_message

logger = logging.getLogger(__name__)


class OraclePredictTool(Tool):
    """Predict future outcomes across multiple domains using Oracle engine."""

    name = "oracle_predict"
    description = "Predict future outcomes using the Oracle multi-domain prediction engine. Supports: predict, diagnose, timeslider, scenario, backtest, portfolio, options, technical_analysis, market_data, esg, yield_curve, risk_decomposition, valuation"
    parameters = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["predict", "diagnose", "timeslider", "scenario", "backtest", "portfolio", "options", "technical_analysis", "market_data", "esg", "yield_curve", "risk_decomposition", "valuation", "predict_to_kg", "diagnose_to_kg"],
                "description": "The prediction action to perform",
            },
            "domain": {
                "type": "string",
                "enum": ["economic", "geopolitical", "climate", "sentiment", "contagion", "demographics", "all"],
                "description": "The domain to predict for",
            },
            "target_date": {
                "type": "string",
                "description": "ISO date string for the prediction target (e.g., '2027-01-01'). Default: 1 year from today.",
            },
            "confidence": {
                "type": "string",
                "enum": ["low", "medium", "high", "very_high"],
                "description": "Confidence level for predictions. Default: high",
            },
            "context": {
                "type": "object",
                "description": "Domain-specific parameters (e.g., gdp_growth_rate, interest_rate, military_tension)",
            },
            "portfolio_params": {
                "type": "object",
                "description": "Parameters for portfolio optimization (action, assets, method, target_return)",
            },
            "options_params": {
                "type": "object",
                "description": "Parameters for options pricing (spot, strike, volatility, etc.)",
            },
            "ta_params": {
                "type": "object",
                "description": "Parameters for technical analysis (symbol, data, indicators)",
            },
            "market_params": {
                "type": "object",
                "description": "Parameters for market data (symbols, period)",
            },
            "scenario_params": {
                "type": "object",
                "description": "Parameters for scenario analysis (scenario_type, target_date)",
            },
            "backtest_params": {
                "type": "object",
                "description": "Parameters for backtesting (start_date, end_date, method)",
            },
            "esg_params": {
                "type": "object",
                "description": "Parameters for ESG analysis (symbol, sector, scenario)",
            },
            "yield_curve_params": {
                "type": "object",
                "description": "Parameters for yield curve analysis (method, scenario)",
            },
            "risk_decomp_params": {
                "type": "object",
                "description": "Parameters for risk decomposition (portfolio, method)",
            },
            "valuation_params": {
                "type": "object",
                "description": "Parameters for valuation (symbol, method, scenario)",
            },
        },
        "required": ["action"],
    }

    async def execute(self, **params) -> dict[str, Any]:
        action = params.get("action", "predict")
        domain = params.get("domain", "all")
        target_date = params.get("target_date")
        confidence = params.get("confidence", "high")
        context = params.get("context", {})

        try:
            if action in ("predict", "diagnose", "timeslider"):
                from core.oracle_integration import (
                    oracle_predict, oracle_diagnose, oracle_timeslider,
                )
                if action == "predict":
                    result = await asyncio.to_thread(
                        oracle_predict, domain, target_date, confidence, context,
                    )
                elif action == "diagnose":
                    result = await asyncio.to_thread(
                        oracle_diagnose, domain, context,
                    )
                elif action == "timeslider":
                    result = await asyncio.to_thread(
                        oracle_timeslider, target_date, "baseline",
                    )
                return result

            elif action == "portfolio":
                from core.oracle_integration import oracle_portfolio
                p = params.get("portfolio_params", {})
                return await asyncio.to_thread(
                    oracle_portfolio,
                    action=p.get("action", "optimize"),
                    assets=p.get("assets"),
                    method=p.get("method", "mean_variance"),
                    target_return=p.get("target_return"),
                    risk_free_rate=p.get("risk_free_rate", 0.04),
                )

            elif action == "options":
                from core.oracle_integration import oracle_options
                o = params.get("options_params", {})
                return await asyncio.to_thread(
                    oracle_options,
                    action=o.get("action", "price"),
                    spot=float(o.get("spot", 100.0)),
                    strike=float(o.get("strike", 100.0)),
                    time_to_expiry=float(o.get("time_to_expiry", 1.0)),
                    risk_free_rate=float(o.get("risk_free_rate", 0.04)),
                    volatility=float(o.get("volatility", 0.25)),
                    option_type=o.get("option_type", "call"),
                    dividend_yield=float(o.get("dividend_yield", 0.0)),
                )

            elif action == "technical_analysis":
                from core.oracle_integration import oracle_technical_analysis
                t = params.get("ta_params", {})
                return oracle_technical_analysis(
                    symbol=t.get("symbol"),
                    data=t.get("data"),
                    indicators=t.get("indicators"),
                )

            elif action == "market_data":
                from core.oracle_integration import oracle_market_data
                m = params.get("market_params", {})
                return oracle_market_data(
                    symbols=m.get("symbols"),
                    period=m.get("period", "1mo"),
                )

            elif action == "scenario":
                from core.oracle_integration import oracle_scenario
                s = params.get("scenario_params", {})
                return oracle_scenario(
                    domain=domain,
                    scenario_type=s.get("scenario_type", "baseline"),
                    target_date=s.get("target_date") or target_date,
                    context=context,
                )

            elif action == "backtest":
                from core.oracle_integration import oracle_backtest
                b = params.get("backtest_params", {})
                return oracle_backtest(
                    domain=domain,
                    start_date=b.get("start_date"),
                    end_date=b.get("end_date"),
                    method=b.get("method", "walk_forward"),
                )

            elif action == "esg":
                from core.oracle_integration import oracle_esg
                e = params.get("esg_params", {})
                return oracle_esg(
                    symbol=e.get("symbol"),
                    sector=e.get("sector"),
                    scenario=e.get("scenario", "baseline"),
                )

            elif action == "yield_curve":
                from core.oracle_integration import oracle_yield_curve
                y = params.get("yield_curve_params", {})
                return oracle_yield_curve(
                    method=y.get("method", "nelson_siegel"),
                    scenario=y.get("scenario", "baseline"),
                )

            elif action == "risk_decomposition":
                from core.oracle_integration import oracle_risk_decomposition
                r = params.get("risk_decomp_params", {})
                return oracle_risk_decomposition(
                    portfolio=r.get("portfolio"),
                    method=r.get("method", "component_var"),
                )

            elif action == "valuation":
                from core.oracle_integration import oracle_valuation
                v = params.get("valuation_params", {})
                return oracle_valuation(
                    symbol=v.get("symbol", ""),
                    method=v.get("method", "dcf"),
                    scenario=v.get("scenario", "baseline"),
                )

            elif action == "predict_to_kg":
                from core.oracle_integration import oracle_predict_to_knowledge_graph
                return await asyncio.to_thread(
                    oracle_predict_to_knowledge_graph,
                    domain=domain,
                    target_date=target_date,
                    confidence=confidence,
                    context=context,
                )

            elif action == "diagnose_to_kg":
                from core.oracle_integration import oracle_diagnose_to_knowledge_graph
                return await asyncio.to_thread(
                    oracle_diagnose_to_knowledge_graph,
                    domain=domain,
                    context=context,
                )

            else:
                return {"error": f"Unknown action: {action}", "predictions": []}

        except ImportError as e:
            return {"error": f"Oracle engine not available: {e}"}
        except Exception as e:
            logger.error(f"Oracle tool error: {e}")
            return {"error": "Prediction failed", "detail": safe_error_message(e)[:200]}


def register_oracle_tools(registry) -> int:
    """Register Oracle prediction tools with a tool registry."""
    registry.register(OraclePredictTool())
    return 1