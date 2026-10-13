import pytest


@pytest.fixture
def memory_provider_circuits(monkeypatch):
    """Isolate LLM routing tests from the persistent provider cooldown ledger."""
    import autotube.llm as llm

    class MemoryCircuits:
        def __init__(self):
            self._open = {}
            self._failures = {}
            self.events = []

        def is_open(self, provider):
            return provider in self._open

        def blocked_until(self, provider):
            return self._open.get(provider)

        def open(self, provider, status_code, reason, cooldown_seconds, retry_after_seconds=None):
            value = f"cooldown:{provider}"
            self._open[provider] = value
            self.events.append({"provider": provider, "status_code": status_code, "action": "circuit_open"})
            return value

        def note_5xx(self, provider, status_code, threshold=3):
            count = self._failures.get(provider, 0) + 1
            self._failures[provider] = count
            if count >= threshold:
                self.open(provider, status_code, "repeated_http_5xx", 600)
                return True
            return False

        def note_success(self, provider):
            self._failures[provider] = 0

        def snapshot(self):
            return {key: {"blocked_until": value} for key, value in self._open.items()}

    monkeypatch.setattr(llm, "ProviderCircuitBreakers", MemoryCircuits)
    return MemoryCircuits
