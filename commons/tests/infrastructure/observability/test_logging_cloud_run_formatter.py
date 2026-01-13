import json
import logging

from caster_commons.observability.logging import CloudJsonSanitizer, ExtrasFormatter


def test_formatter_emits_json_payload_for_json_fields_in_cloud_run(monkeypatch) -> None:
    monkeypatch.setenv("K_SERVICE", "caster-platform")
    formatter = ExtrasFormatter("%(levelname)s %(name)s: %(message)s")

    record = logging.LogRecord(
        name="caster_commons.llm.calls",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg="llm.invoke.retry.complete",
        args=(),
        exc_info=None,
    )
    record.data = {"provider": "openai"}
    record.json_fields = {"request": {"payload": b"hello"}, "response": {"ok": True}}

    rendered = formatter.format(record)
    payload = json.loads(rendered)

    assert payload["message"] == "llm.invoke.retry.complete"
    assert payload["severity"] == "INFO"
    assert payload["logger"] == "caster_commons.llm.calls"
    assert payload["data"]["provider"] == "openai"
    assert payload["request"]["payload"] == "<bytes len=5>"
    assert payload["response"]["ok"] is True


def test_formatter_emits_json_payload_for_json_fields_in_kubernetes(monkeypatch) -> None:
    monkeypatch.delenv("K_SERVICE", raising=False)
    monkeypatch.setenv("KUBERNETES_SERVICE_HOST", "10.0.0.1")
    formatter = ExtrasFormatter("%(levelname)s %(name)s: %(message)s")

    record = logging.LogRecord(
        name="caster_commons.tools.desearch.calls",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg="desearch.request.complete",
        args=(),
        exc_info=None,
    )
    record.data = {"provider": "desearch"}
    record.json_fields = {"request": {"payload": b"hello"}, "response": {"ok": True}}

    rendered = formatter.format(record)
    payload = json.loads(rendered)

    assert payload["message"] == "desearch.request.complete"
    assert payload["severity"] == "INFO"
    assert payload["logger"] == "caster_commons.tools.desearch.calls"
    assert payload["data"]["provider"] == "desearch"
    assert payload["request"]["payload"] == "<bytes len=5>"
    assert payload["response"]["ok"] is True


def test_formatter_ignores_json_fields_outside_managed_runtimes(monkeypatch) -> None:
    monkeypatch.delenv("K_SERVICE", raising=False)
    monkeypatch.delenv("KUBERNETES_SERVICE_HOST", raising=False)
    formatter = ExtrasFormatter("%(levelname)s %(name)s: %(message)s")

    record = logging.LogRecord(
        name="caster_commons.llm.calls",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg="llm.invoke.retry.complete",
        args=(),
        exc_info=None,
    )
    record.data = {"provider": "openai"}
    record.json_fields = {"request": {"payload": "hello"}}

    rendered = formatter.format(record)

    assert rendered.startswith("INFO caster_commons.llm.calls: llm.invoke.retry.complete")
    assert "data=" in rendered
    assert "request" not in rendered


def test_cloud_json_sanitizer_injects_data_into_json_fields() -> None:
    sanitizer = CloudJsonSanitizer()
    record = logging.LogRecord(
        name="caster_commons.llm.calls",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg="llm.invoke.retry.complete",
        args=(),
        exc_info=None,
    )
    record.data = {"provider": "openai", "payload": b"hello"}
    record.json_fields = {"request": {"ok": True}}

    assert sanitizer.filter(record) is True
    assert record.json_fields["request"]["ok"] is True
    assert record.json_fields["data"]["provider"] == "openai"
    assert record.json_fields["data"]["payload"] == "<bytes len=5>"
