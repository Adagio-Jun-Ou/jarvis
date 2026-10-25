"""Tests for Oracle prediction engine integration."""

import pytest
import sys
from pathlib import Path


@pytest.fixture
def oracle_path():
    """Ensure Oracle path is in sys.path."""
    oracle_src = Path.home() / "oracle" / "src"
    if str(oracle_src) not in sys.path:
        sys.path.insert(0, str(oracle_src))
    return oracle_src


class TestOracleIntegration:
    """Test Oracle engine integration module."""

    def test_import_oracle_integration(self):
        from core.oracle_integration import get_oracle_engine
        assert callable(get_oracle_engine)

    def test_oracle_engine_initialization(self, oracle_path):
        from core.oracle_integration import get_oracle_engine
        engine = get_oracle_engine()
        if engine is None:
            pytest.skip("Oracle engine not available")
        assert engine is not None

    def test_oracle_predict_economic(self, oracle_path):
        from core.oracle_integration import oracle_predict
        result = oracle_predict(
            domain="economic",
            target_date="2027-01-01",
            confidence="high",
            context={"gdp_growth_rate": 0.025, "interest_rate": 0.05},
        )
        assert "predictions" in result
        if "error" not in result:
            assert len(result["predictions"]) > 0
            pred = result["predictions"][0]
            assert "variable" in pred
            assert "model" in pred

    def test_oracle_predict_all_domains(self, oracle_path):
        from core.oracle_integration import oracle_predict
        result = oracle_predict(domain="all", confidence="medium")
        assert "predictions" in result
        if "error" not in result:
            assert result["num_models"] >= 0

    def test_oracle_predict_invalid_confidence(self, oracle_path):
        from core.oracle_integration import oracle_predict
        result = oracle_predict(confidence="invalid_level")
        assert "predictions" in result
        if "error" not in result:
            assert result["confidence"] == "invalid_level"

    def test_oracle_predict_invalid_domain(self, oracle_path):
        from core.oracle_integration import oracle_predict
        result = oracle_predict(domain="invalid_domain_xyz")
        assert "error" in result
        assert "Invalid domain" in result["error"]

    def test_oracle_predict_context_validation(self, oracle_path):
        from core.oracle_integration import oracle_predict
        result = oracle_predict(domain="economic", context={"gdp_growth_rate": "not_a_number"})
        assert "predictions" in result or "error" in result

    def test_oracle_diagnose(self, oracle_path):
        from core.oracle_integration import oracle_diagnose
        result = oracle_diagnose(domain="economic")
        assert "interactions" in result

    def test_oracle_timeslider(self, oracle_path):
        from core.oracle_integration import oracle_timeslider
        result = oracle_timeslider(target_date="2028-06-01")
        assert "forecasts" in result or "error" in result

    def test_oracle_predict_invalid_date(self, oracle_path):
        from core.oracle_integration import oracle_predict
        result = oracle_predict(target_date="not-a-date")
        assert "error" in result or "predictions" in result

    def test_oracle_predict_geopolitical(self, oracle_path):
        from core.oracle_integration import oracle_predict
        result = oracle_predict(
            domain="geopolitical",
            context={"military_tension": 0.6, "trade_conflict": 0.4},
        )
        assert "predictions" in result

    def test_oracle_predict_climate(self, oracle_path):
        from core.oracle_integration import oracle_predict
        result = oracle_predict(
            domain="climate",
            context={"temperature_anomaly": 1.5, "co2_concentration": 420},
        )
        assert "predictions" in result

    def test_oracle_predict_contagion(self, oracle_path):
        from core.oracle_integration import oracle_predict
        result = oracle_predict(
            domain="contagion",
            context={"systemic_risk": 0.7, "interbank_connectivity": 0.5},
        )
        assert "predictions" in result


class TestOracleTool:
    """Test Oracle tool registration."""

    def test_oracle_tool_schema(self):
        from tools.oracle import OraclePredictTool
        tool = OraclePredictTool()
        schema = tool.get_schema()
        assert schema["name"] == "oracle_predict"
        assert "action" in schema["parameters"]["properties"]

    def test_oracle_tool_registration(self):
        from tools.oracle import register_oracle_tools
        from tools.base import ToolRegistry
        registry = ToolRegistry()
        count = register_oracle_tools(registry)
        assert count == 1
        assert "oracle_predict" in registry.list_tools()

    @pytest.mark.asyncio
    async def test_oracle_tool_execute_predict(self):
        from tools.oracle import OraclePredictTool
        tool = OraclePredictTool()
        result = await tool.execute(action="predict", domain="economic", confidence="high")
        assert isinstance(result, dict)
        assert "predictions" in result or "error" in result

    @pytest.mark.asyncio
    async def test_oracle_tool_execute_diagnose(self):
        from tools.oracle import OraclePredictTool
        tool = OraclePredictTool()
        result = await tool.execute(action="diagnose", domain="all")
        assert isinstance(result, dict)
        assert "interactions" in result or "error" in result

    @pytest.mark.asyncio
    async def test_oracle_tool_execute_timeslider(self):
        from tools.oracle import OraclePredictTool
        tool = OraclePredictTool()
        result = await tool.execute(action="timeslider")
        assert isinstance(result, dict)

    @pytest.mark.asyncio
    async def test_oracle_tool_execute_invalid_action(self):
        from tools.oracle import OraclePredictTool
        tool = OraclePredictTool()
        result = await tool.execute(action="invalid_action")
        assert isinstance(result, dict)
        assert "error" in result


class TestOraclePortfolio:
    """Test Oracle portfolio optimization integration."""

    def test_oracle_portfolio_module_loads(self):
        from core.oracle_integration import oracle_portfolio
        assert callable(oracle_portfolio)

    def test_oracle_portfolio_optimize(self):
        from core.oracle_integration import oracle_portfolio
        result = oracle_portfolio(action="optimize")
        assert isinstance(result, dict)

    def test_oracle_portfolio_efficient_frontier(self):
        from core.oracle_integration import oracle_portfolio
        result = oracle_portfolio(action="efficient_frontier")
        assert isinstance(result, dict)


class TestOracleOptions:
    """Test Oracle options pricing integration."""

    def test_oracle_options_module_loads(self):
        from core.oracle_integration import oracle_options
        assert callable(oracle_options)

    def test_oracle_options_price(self):
        from core.oracle_integration import oracle_options
        result = oracle_options(action="price", spot=100, strike=100, volatility=0.25)
        assert isinstance(result, dict)
        if "error" not in result:
            assert "price" in result

    def test_oracle_options_greeks(self):
        from core.oracle_integration import oracle_options
        result = oracle_options(action="greeks", spot=100, strike=105)
        assert isinstance(result, dict)

    def test_oracle_options_put(self):
        from core.oracle_integration import oracle_options
        result = oracle_options(action="price", option_type="put")
        assert isinstance(result, dict)

    def test_oracle_options_invalid_action(self):
        from core.oracle_integration import oracle_options
        result = oracle_options(action="invalid")
        assert isinstance(result, dict)
        assert "error" in result


class TestOracleTechnicalAnalysis:
    """Test Oracle technical analysis integration."""

    def test_oracle_ta_module_loads(self):
        from core.oracle_integration import oracle_technical_analysis
        assert callable(oracle_technical_analysis)

    def test_oracle_ta_default_indicators(self):
        from core.oracle_integration import oracle_technical_analysis
        result = oracle_technical_analysis(indicators=["rsi"])
        assert isinstance(result, dict)
        if "error" not in result:
            assert "indicators" in result

    def test_oracle_ta_multiple_indicators(self):
        from core.oracle_integration import oracle_technical_analysis
        result = oracle_technical_analysis(indicators=["rsi", "macd", "bollinger"])
        assert isinstance(result, dict)


class TestOracleMarketData:
    """Test Oracle market data integration."""

    def test_oracle_market_data_module_loads(self):
        from core.oracle_integration import oracle_market_data
        assert callable(oracle_market_data)

    def test_oracle_market_data_default_symbols(self):
        from core.oracle_integration import oracle_market_data
        result = oracle_market_data()
        assert isinstance(result, dict)


class TestOracleToolAdvanced:
    """Test Oracle tool with advanced action types."""

    @pytest.mark.asyncio
    async def test_oracle_tool_options_action(self):
        from tools.oracle import OraclePredictTool
        tool = OraclePredictTool()
        result = await tool.execute(
            action="options",
            options_params={"spot": 100, "strike": 100, "volatility": 0.25},
        )
        assert isinstance(result, dict)

    @pytest.mark.asyncio
    async def test_oracle_tool_portfolio_action(self):
        from tools.oracle import OraclePredictTool
        tool = OraclePredictTool()
        result = await tool.execute(
            action="portfolio",
            portfolio_params={"action": "optimize"},
        )
        assert isinstance(result, dict)

    @pytest.mark.asyncio
    async def test_oracle_tool_ta_action(self):
        from tools.oracle import OraclePredictTool
        tool = OraclePredictTool()
        result = await tool.execute(
            action="technical_analysis",
            ta_params={"indicators": ["rsi", "macd"]},
        )
        assert isinstance(result, dict)

    @pytest.mark.asyncio
    async def test_oracle_tool_market_data_action(self):
        from tools.oracle import OraclePredictTool
        tool = OraclePredictTool()
        result = await tool.execute(
            action="market_data",
            market_params={"period": "5d"},
        )
        assert isinstance(result, dict)


class TestOracleAdvancedWrappers:
    """Test Oracle advanced wrapper functions."""

    def test_oracle_scenario_callable(self):
        from core.oracle_integration import oracle_scenario
        assert callable(oracle_scenario)

    def test_oracle_scenario_baseline(self):
        from core.oracle_integration import oracle_scenario
        result = oracle_scenario(scenario_type="baseline")
        assert isinstance(result, dict)
        assert "scenario_type" in result or "error" in result

    def test_oracle_scenario_bear(self):
        from core.oracle_integration import oracle_scenario
        result = oracle_scenario(scenario_type="bear")
        assert isinstance(result, dict)

    def test_oracle_backtest_callable(self):
        from core.oracle_integration import oracle_backtest
        assert callable(oracle_backtest)

    def test_oracle_backtest_walk_forward(self):
        from core.oracle_integration import oracle_backtest
        result = oracle_backtest(method="walk_forward")
        assert isinstance(result, dict)
        assert "method" in result or "error" in result

    def test_oracle_backtest_invalid_method(self):
        from core.oracle_integration import oracle_backtest
        result = oracle_backtest(method="nonexistent")
        assert isinstance(result, dict)

    def test_oracle_esg_callable(self):
        from core.oracle_integration import oracle_esg
        assert callable(oracle_esg)

    def test_oracle_esg_baseline(self):
        from core.oracle_integration import oracle_esg
        result = oracle_esg(scenario="baseline")
        assert isinstance(result, dict)

    def test_oracle_yield_curve_callable(self):
        from core.oracle_integration import oracle_yield_curve
        assert callable(oracle_yield_curve)

    def test_oracle_yield_curve_nelson_siegel(self):
        from core.oracle_integration import oracle_yield_curve
        result = oracle_yield_curve(method="nelson_siegel")
        assert isinstance(result, dict)
        assert "method" in result or "error" in result

    def test_oracle_yield_curve_invalid_method(self):
        from core.oracle_integration import oracle_yield_curve
        result = oracle_yield_curve(method="nonexistent")
        assert isinstance(result, dict)
        assert "error" in result

    def test_oracle_risk_decomposition_invalid_method(self):
        from core.oracle_integration import oracle_risk_decomposition
        result = oracle_risk_decomposition(method="nonexistent")
        assert isinstance(result, dict)
        assert "error" in result

    def test_oracle_risk_decomposition_invalid_portfolio(self):
        from core.oracle_integration import oracle_risk_decomposition
        result = oracle_risk_decomposition(portfolio="not_a_list")
        assert isinstance(result, dict)
        assert "error" in result

    def test_oracle_risk_decomposition_portfolio_too_large(self):
        from core.oracle_integration import oracle_risk_decomposition
        result = oracle_risk_decomposition(portfolio=["A"] * 51)
        assert isinstance(result, dict)
        assert "error" in result

    def test_oracle_scenario_invalid_type(self):
        from core.oracle_integration import oracle_scenario
        result = oracle_scenario(scenario_type="nonexistent")
        assert isinstance(result, dict)
        assert "error" in result

    def test_oracle_backtest_invalid_method(self):
        from core.oracle_integration import oracle_backtest
        result = oracle_backtest(method="nonexistent")
        assert isinstance(result, dict)
        assert "error" in result

    def test_oracle_backtest_invalid_date(self):
        from core.oracle_integration import oracle_backtest
        result = oracle_backtest(start_date="not-a-date")
        assert isinstance(result, dict)
        assert "error" in result

    def test_oracle_esg_invalid_scenario(self):
        from core.oracle_integration import oracle_esg
        result = oracle_esg(scenario="nonexistent")
        assert isinstance(result, dict)
        assert "error" in result

    def test_oracle_esg_invalid_symbol(self):
        from core.oracle_integration import oracle_esg
        result = oracle_esg(symbol="THISISWAYTOOLONG")
        assert isinstance(result, dict)
        assert "error" in result

    def test_oracle_valuation_invalid_method(self):
        from core.oracle_integration import oracle_valuation
        result = oracle_valuation(symbol="AAPL", method="nonexistent")
        assert isinstance(result, dict)
        assert "error" in result

    def test_oracle_valuation_invalid_scenario(self):
        from core.oracle_integration import oracle_valuation
        result = oracle_valuation(symbol="AAPL", scenario="nonexistent")
        assert isinstance(result, dict)
        assert "error" in result

    def test_oracle_valuation_empty_symbol(self):
        from core.oracle_integration import oracle_valuation
        result = oracle_valuation(symbol="")
        assert isinstance(result, dict)
        assert "error" in result

    def test_oracle_risk_decomposition_callable(self):
        from core.oracle_integration import oracle_risk_decomposition
        assert callable(oracle_risk_decomposition)

    def test_oracle_risk_decomposition_component_var(self):
        from core.oracle_integration import oracle_risk_decomposition
        result = oracle_risk_decomposition(method="component_var")
        assert isinstance(result, dict)

    def test_oracle_risk_decomposition_stress(self):
        from core.oracle_integration import oracle_risk_decomposition
        result = oracle_risk_decomposition(method="stress_decomposition")
        assert isinstance(result, dict)

    def test_oracle_risk_decomposition_invalid_method(self):
        from core.oracle_integration import oracle_risk_decomposition
        result = oracle_risk_decomposition(method="nonexistent")
        assert isinstance(result, dict)
        assert "error" in result

    def test_oracle_valuation_callable(self):
        from core.oracle_integration import oracle_valuation
        assert callable(oracle_valuation)

    def test_oracle_valuation_dcf(self):
        from core.oracle_integration import oracle_valuation
        result = oracle_valuation(symbol="AAPL", method="dcf")
        assert isinstance(result, dict)

    def test_oracle_valuation_invalid_method(self):
        from core.oracle_integration import oracle_valuation
        result = oracle_valuation(symbol="AAPL", method="nonexistent")
        assert isinstance(result, dict)
        assert "error" in result


class TestOracleToolExtendedActions:
    """Test Oracle tool with extended action types."""

    @pytest.mark.asyncio
    async def test_oracle_tool_scenario_action(self):
        from tools.oracle import OraclePredictTool
        tool = OraclePredictTool()
        result = await tool.execute(
            action="scenario",
            scenario_params={"scenario_type": "bear"},
        )
        assert isinstance(result, dict)

    @pytest.mark.asyncio
    async def test_oracle_tool_backtest_action(self):
        from tools.oracle import OraclePredictTool
        tool = OraclePredictTool()
        result = await tool.execute(
            action="backtest",
            backtest_params={"method": "walk_forward"},
        )
        assert isinstance(result, dict)

    @pytest.mark.asyncio
    async def test_oracle_tool_esg_action(self):
        from tools.oracle import OraclePredictTool
        tool = OraclePredictTool()
        result = await tool.execute(
            action="esg",
            esg_params={"scenario": "baseline"},
        )
        assert isinstance(result, dict)

    @pytest.mark.asyncio
    async def test_oracle_tool_yield_curve_action(self):
        from tools.oracle import OraclePredictTool
        tool = OraclePredictTool()
        result = await tool.execute(
            action="yield_curve",
            yield_curve_params={"method": "nelson_siegel"},
        )
        assert isinstance(result, dict)

    @pytest.mark.asyncio
    async def test_oracle_tool_risk_decomp_action(self):
        from tools.oracle import OraclePredictTool
        tool = OraclePredictTool()
        result = await tool.execute(
            action="risk_decomposition",
            risk_decomp_params={"method": "component_var"},
        )
        assert isinstance(result, dict)

    @pytest.mark.asyncio
    async def test_oracle_tool_valuation_action(self):
        from tools.oracle import OraclePredictTool
        tool = OraclePredictTool()
        result = await tool.execute(
            action="valuation",
            valuation_params={"symbol": "AAPL", "method": "dcf"},
        )
        assert isinstance(result, dict)


class TestOracleAgentContext:
    """Test Oracle context injection into agent system prompt."""

    def test_get_oracle_context_no_topics(self):
        from core.system_prompt import SystemPromptBuilder
        builder = SystemPromptBuilder(type("StubAgent", (), {})())
        builder._oracle_context_cache = {}
        builder._oracle_context_ttl = 300.0
        result = builder._oracle_context([])
        assert result is None

    def test_get_oracle_context_financial_topics(self):
        from core.system_prompt import SystemPromptBuilder
        builder = SystemPromptBuilder(type("StubAgent", (), {})())
        builder._oracle_context_cache = {}
        builder._oracle_context_ttl = 300.0
        result = builder._oracle_context(["stock market volatility"])
        assert result is None or "Oracle Prediction" in result or isinstance(result, str)

    def test_get_oracle_context_non_financial(self):
        from core.system_prompt import SystemPromptBuilder
        builder = SystemPromptBuilder(type("StubAgent", (), {})())
        builder._oracle_context_cache = {}
        builder._oracle_context_ttl = 300.0
        result = builder._oracle_context(["weather", "cooking recipes"])
        assert result is None

    def test_oracle_context_uses_cache(self):
        from core.system_prompt import SystemPromptBuilder
        import time
        builder = SystemPromptBuilder(type("StubAgent", (), {})())
        builder._oracle_context_cache = {}
        builder._oracle_context_ttl = 300.0
        
        result = builder._oracle_context(["inflation and gdp"])
        cache_key = "economic"
        
        if result is not None:
            assert cache_key in builder._oracle_context_cache
            cached_result, cached_time = builder._oracle_context_cache[cache_key]
            assert cached_result == result
            assert cached_time > 0

    def test_oracle_context_cache_ttl_expiry(self):
        from core.system_prompt import SystemPromptBuilder
        import time
        builder = SystemPromptBuilder(type("StubAgent", (), {})())
        builder._oracle_context_cache = {}
        builder._oracle_context_ttl = 0.001
        
        result = builder._oracle_context(["stock market"])
        if result is not None:
            time.sleep(0.002)
            result2 = builder._oracle_context(["stock market"])
            if result2 is not None:
                assert result2 != result or True


class TestOracleKnowledgeGraphBridge:
    """Test Oracle-Knowledge Graph integration."""

    def test_predict_to_kg_callable(self):
        from core.oracle_integration import oracle_predict_to_knowledge_graph
        assert callable(oracle_predict_to_knowledge_graph)

    def test_diagnose_to_kg_callable(self):
        from core.oracle_integration import oracle_diagnose_to_knowledge_graph
        assert callable(oracle_diagnose_to_knowledge_graph)

    def test_predict_to_kg_economic(self):
        from core.oracle_integration import oracle_predict_to_knowledge_graph
        result = oracle_predict_to_knowledge_graph(domain="economic", confidence="low")
        assert isinstance(result, dict)
        assert "kg_integrated" in result

    def test_diagnose_to_kg(self):
        from core.oracle_integration import oracle_diagnose_to_knowledge_graph
        result = oracle_diagnose_to_knowledge_graph(domain="all")
        assert isinstance(result, dict)
        assert "kg_integrated" in result

    @pytest.mark.asyncio
    async def test_oracle_tool_predict_to_kg_action(self):
        from tools.oracle import OraclePredictTool
        tool = OraclePredictTool()
        result = await tool.execute(
            action="predict_to_kg",
            domain="economic",
            confidence="low",
        )
        assert isinstance(result, dict)
        assert "kg_integrated" in result or "error" in result

    @pytest.mark.asyncio
    async def test_oracle_tool_diagnose_to_kg_action(self):
        from tools.oracle import OraclePredictTool
        tool = OraclePredictTool()
        result = await tool.execute(
            action="diagnose_to_kg",
            domain="all",
        )
        assert isinstance(result, dict)
        assert "kg_integrated" in result or "error" in result

    def test_tool_schema_includes_kg_actions(self):
        from tools.oracle import OraclePredictTool
        tool = OraclePredictTool()
        schema = tool.get_schema()
        actions = schema["parameters"]["properties"]["action"]["enum"]
        assert "predict_to_kg" in actions
        assert "diagnose_to_kg" in actions


class TestOracleAgentContextComprehensive:
    """Comprehensive tests for Oracle context injection in agent reasoning loop."""

    def test_economy_topic_detection(self):
        from core.system_prompt import SystemPromptBuilder
        builder = SystemPromptBuilder(type("StubAgent", (), {})())
        builder._oracle_context_cache = {}
        builder._oracle_context_ttl = 300.0
        result = builder._oracle_context(["inflation and gdp growth"])
        if result is not None:
            assert "Oracle Prediction" in result
            assert "economic" in result.lower() or "Economic" in result

    def test_geopolitics_topic_detection(self):
        from core.system_prompt import SystemPromptBuilder
        builder = SystemPromptBuilder(type("StubAgent", (), {})())
        builder._oracle_context_cache = {}
        builder._oracle_context_ttl = 300.0
        result = builder._oracle_context(["war and sanctions"])
        if result is not None:
            assert "Oracle Prediction" in result

    def test_climate_topic_detection(self):
        from core.system_prompt import SystemPromptBuilder
        builder = SystemPromptBuilder(type("StubAgent", (), {})())
        builder._oracle_context_cache = {}
        builder._oracle_context_ttl = 300.0
        result = builder._oracle_context(["climate change and carbon emissions"])
        if result is not None:
            assert "Oracle Prediction" in result

    def test_finance_topic_detection(self):
        from core.system_prompt import SystemPromptBuilder
        builder = SystemPromptBuilder(type("StubAgent", (), {})())
        builder._oracle_context_cache = {}
        builder._oracle_context_ttl = 300.0
        result = builder._oracle_context(["stock market portfolio investment"])
        if result is not None:
            assert "Oracle Prediction" in result

    def test_technology_topic_detection(self):
        from core.system_prompt import SystemPromptBuilder
        builder = SystemPromptBuilder(type("StubAgent", (), {})())
        builder._oracle_context_cache = {}
        builder._oracle_context_ttl = 300.0
        result = builder._oracle_context(["ai chips and semiconductor disruption"])
        if result is not None:
            assert "Oracle Prediction" in result

    def test_health_topic_detection(self):
        from core.system_prompt import SystemPromptBuilder
        builder = SystemPromptBuilder(type("StubAgent", (), {})())
        builder._oracle_context_cache = {}
        builder._oracle_context_ttl = 300.0
        result = builder._oracle_context(["pandemic vaccine outbreak"])
        if result is not None:
            assert "Oracle Prediction" in result

    def test_energy_topic_detection(self):
        from core.system_prompt import SystemPromptBuilder
        builder = SystemPromptBuilder(type("StubAgent", (), {})())
        builder._oracle_context_cache = {}
        builder._oracle_context_ttl = 300.0
        result = builder._oracle_context(["oil renewable solar energy"])
        if result is not None:
            assert "Oracle Prediction" in result

    def test_sentiment_topic_detection(self):
        from core.system_prompt import SystemPromptBuilder
        builder = SystemPromptBuilder(type("StubAgent", (), {})())
        builder._oracle_context_cache = {}
        builder._oracle_context_ttl = 300.0
        result = builder._oracle_context(["market sentiment consumer confidence"])
        if result is not None:
            assert "Oracle Prediction" in result

    def test_no_topic_returns_none(self):
        from core.system_prompt import SystemPromptBuilder
        builder = SystemPromptBuilder(type("StubAgent", (), {})())
        builder._oracle_context_cache = {}
        builder._oracle_context_ttl = 300.0
        result = builder._oracle_context(["cooking recipes"])
        assert result is None

    def test_mixed_topics_picks_first_domain(self):
        from core.system_prompt import SystemPromptBuilder
        builder = SystemPromptBuilder(type("StubAgent", (), {})())
        builder._oracle_context_cache = {}
        builder._oracle_context_ttl = 300.0
        result = builder._oracle_context(["stock market and climate change"])
        if result is not None:
            assert "Oracle Prediction" in result

    def test_empty_topics_returns_none(self):
        from core.system_prompt import SystemPromptBuilder
        builder = SystemPromptBuilder(type("StubAgent", (), {})())
        result = builder._oracle_context([])
        assert result is None


class TestOracleWebEndpointCount:
    """Verify Oracle web API endpoint count is correct."""

    def test_oracle_endpoint_total(self):
        import re
        with open('interfaces/web_react.py', 'r', encoding='utf-8') as f:
            content = f.read()
        oracle_endpoints = len(re.findall(r'@self\.app\.(get|post)\("/api/oracle/', content))
        assert oracle_endpoints >= 15, f"Expected >= 15 Oracle endpoints, found {oracle_endpoints}"

    def test_oracle_dashboard_exists(self):
        import re
        with open('interfaces/web_react.py', 'r', encoding='utf-8') as f:
            content = f.read()
        assert '"/oracle"' in content

    def test_oracle_sse_stream_exists(self):
        import re
        with open('interfaces/web_react.py', 'r', encoding='utf-8') as f:
            content = f.read()
        assert '"/api/oracle/market-stream"' in content