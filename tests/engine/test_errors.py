"""The cross-boundary error taxonomy carries HTTP status + JSON shape."""
from alex.errors import (
    AlexError,
    BadRequest,
    ConstitutionalBlock,
    Denied,
    NotFound,
    UpstreamError,
)


def test_http_status_mapping():
    assert Denied().http_status == 403
    assert ConstitutionalBlock().http_status == 403
    assert NotFound().http_status == 404
    assert BadRequest().http_status == 400
    assert UpstreamError().http_status == 502
    assert AlexError().http_status == 500  # default fallback


def test_to_dict_includes_code_message_details():
    e = BadRequest("bad thing", details={"field": "x"})
    assert e.to_dict() == {
        "error": "bad_request",
        "message": "bad thing",
        "details": {"field": "x"},
    }


def test_to_dict_omits_empty_message_and_details():
    assert NotFound().to_dict() == {"error": "not_found"}
