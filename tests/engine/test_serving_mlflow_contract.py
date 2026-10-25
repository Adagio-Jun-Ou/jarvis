"""The serving module against the *real* MLflow contract.

`tests/test_serving_contract.py` proves the translation logic without
MLflow installed. These tests run only when MLflow is present and hold
the emitted shapes to `mlflow.types.responses` — the schema Model
Serving actually validates. The loader wrapper accepts a plain dict
*only if it validates* (`_response_to_dict` model-validates and raises
MlflowException otherwise), so shape drift here is a serving-time
outage, not a cosmetic difference.
"""

from __future__ import annotations

import pytest

from agent import serving

mlflow_types = pytest.importorskip(
    "mlflow.types.responses", reason="mlflow not installed"
)
ResponsesAgentRequest = mlflow_types.ResponsesAgentRequest
ResponsesAgentResponse = mlflow_types.ResponsesAgentResponse
ResponsesAgentStreamEvent = mlflow_types.ResponsesAgentStreamEvent


@pytest.fixture
def env(monkeypatch):
    monkeypatch.setenv("ALEX_ENV", "dev")
    from alex import config, testing

    config._settings = None
    testing._fakes = None
    yield
    config._settings = None
    testing._fakes = None


def test_every_stream_shape_validates():
    iid = serving.new_item_id()
    events = [
        serving.text_delta(iid, "hello"),
        serving.done_item(serving.output_text_item(iid, "hello there")),
        serving.done_item(
            serving.function_call_item(
                serving.new_item_id("fc"), "search", {"q": "x"}, "call_1"
            )
        ),
        serving.done_item(
            serving.function_call_output_item(
                serving.new_item_id("fco"), "call_1", {"hits": 3}
            )
        ),
    ]
    for event in events:
        ResponsesAgentStreamEvent.model_validate(event)


def test_the_predict_response_validates():
    iid = serving.new_item_id()
    ResponsesAgentResponse.model_validate(
        {
            "output": [
                serving.function_call_item(serving.new_item_id("fc"), "s", {}, "c1"),
                serving.function_call_output_item(serving.new_item_id("o"), "c1", "r"),
                serving.output_text_item(iid, "done"),
            ]
        }
    )


def test_build_agent_subclasses_the_real_base():
    from mlflow.pyfunc import ResponsesAgent

    agent = serving.build_agent()

    assert isinstance(agent, ResponsesAgent)
    assert isinstance(agent, serving.AlexResponsesAgent)


def test_a_full_turn_through_the_real_request_type(env, monkeypatch):
    """End to end with a typed ResponsesAgentRequest, the exact object
    Model Serving hands over — including reading its pydantic messages
    and its ChatContext identity fields."""

    class FakeVM:
        def subscribe_sse(self, fn):
            self._sub = fn
            return lambda: None

        def run_turn(self, *, text, history=None, **_kw):
            assert text == "what is the budget?"
            self._sub({"type": "text", "text": "1.2M USD."})
            return "1.2M USD.", []

        def close(self):
            pass

    captured = {}

    def make(session_id, user_email, *, plan_mode=False):
        captured["session"], captured["email"] = session_id, user_email
        return FakeVM()

    monkeypatch.setattr(serving, "make_vm_for_request", make)

    request = ResponsesAgentRequest(
        input=[{"role": "user", "content": "what is the budget?"}],
        context={"conversation_id": "conv-9", "user_id": "dana@alex.dev"},
    )
    out = serving.build_agent().predict(request)

    ResponsesAgentResponse.model_validate(out)
    assert captured == {"session": "conv-9", "email": "dana@alex.dev"}
    final = out["output"][-1]
    assert final["content"][0]["text"] == "1.2M USD."
