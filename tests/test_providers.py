from flense.config import FlenseConfig
from flense.pricing import estimate_cost_saved, lookup_input_price
from flense.providers import build_adapters
from flense.providers.mistral import MistralAdapter


class TestMistralAdapter:
    def setup_method(self):
        self.adapter = MistralAdapter("mistral", "https://api.mistral.ai")

    def test_rewrite_url_strips_prefix(self):
        url = self.adapter.rewrite_url("/mistral/v1/chat/completions")
        assert url == "https://api.mistral.ai/v1/chat/completions"

    def test_ack_response_is_openai_compatible(self):
        ack = self.adapter.make_ack_response("done", "mistral-large-latest")
        assert ack["object"] == "chat.completion"
        assert ack["choices"][0]["message"]["role"] == "assistant"
        assert ack["model"] == "mistral-large-latest"

    def test_error_response_shape(self):
        err = self.adapter.make_error_response("upstream down")
        assert err["error"]["message"] == "upstream down"


class TestMistralWiring:
    def test_default_config_includes_mistral(self):
        cfg = FlenseConfig()
        assert "mistral" in cfg.providers
        assert cfg.providers["mistral"].upstream == "https://api.mistral.ai"

    def test_build_adapters_includes_mistral(self):
        adapters = build_adapters(FlenseConfig())
        assert isinstance(adapters.get("mistral"), MistralAdapter)
        assert adapters["mistral"].upstream == "https://api.mistral.ai"


class TestMistralPricing:
    def test_known_models_priced(self):
        assert lookup_input_price("mistral-large-latest", "mistral") == 2.00
        assert lookup_input_price("codestral-latest", "mistral") == 0.30
        assert lookup_input_price("ministral-8b-latest", "mistral") == 0.10

    def test_unknown_model_returns_none(self):
        assert lookup_input_price("mystery-model", "mistral") is None

    def test_cost_saved(self):
        # 1,000,000 input tokens saved on mistral-large at $2/M == $2.00
        assert estimate_cost_saved(1_000_000, "mistral-large-latest", "mistral") == 2.00
