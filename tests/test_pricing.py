from flense.pricing import estimate_cost_saved, lookup_input_price


def test_anthropic_exact_match():
    price = lookup_input_price("claude-sonnet-4", "anthropic")
    assert price == 3.00


def test_anthropic_prefix_match():
    """Model IDs with date suffixes should match the prefix."""
    price = lookup_input_price("claude-sonnet-4-20250514", "anthropic")
    assert price == 3.00


def test_anthropic_opus():
    assert lookup_input_price("claude-opus-4-20250514", "anthropic") == 15.00


def test_openai_exact():
    assert lookup_input_price("gpt-4o", "openai") == 2.50


def test_openai_mini_matches_before_base():
    """gpt-4o-mini should match its own entry, not gpt-4o."""
    assert lookup_input_price("gpt-4o-mini", "openai") == 0.15


def test_unknown_model():
    assert lookup_input_price("some-unknown-model", "anthropic") is None


def test_unknown_provider():
    assert lookup_input_price("claude-sonnet-4", "google") is None


def test_config_override():
    overrides = {"my-custom-model": 5.00}
    assert lookup_input_price("my-custom-model", "anthropic", overrides) == 5.00


def test_config_override_takes_precedence():
    overrides = {"claude-sonnet-4": 99.00}
    assert lookup_input_price("claude-sonnet-4", "anthropic", overrides) == 99.00


def test_estimate_cost_saved():
    # 1000 tokens saved at $3/M = $0.003
    cost = estimate_cost_saved(1000, "claude-sonnet-4-20250514", "anthropic")
    assert cost is not None
    assert abs(cost - 0.003) < 1e-9


def test_estimate_cost_saved_unknown_model():
    cost = estimate_cost_saved(1000, "unknown-model", "anthropic")
    assert cost is None


def test_estimate_cost_saved_zero_tokens():
    cost = estimate_cost_saved(0, "claude-sonnet-4", "anthropic")
    assert cost == 0.0
