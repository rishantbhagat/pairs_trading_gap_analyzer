"""Automated Test Runner for Pairs Gap Analyzer."""

import sys
import traceback
from pathlib import Path

# Add project root to sys.path
root_dir = Path(__file__).resolve().parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from tests.test_data import (
    test_compute_fundamental_divergence,
    test_fetch_pair_prices_rejects_invalid_pair,
    test_fetch_pair_prices_handles_multiindex_level_order,
)
from tests.test_features import (
    test_ols_and_rolling_zscore,
    test_half_life_and_hurst,
    test_gap_classification,
    test_sentiment_scoring,
    test_support_resistance_channels_use_confirmed_pivots_and_rank_strength,
    test_support_resistance_rejects_missing_ohlc,
)
from tests.test_deep_model import (
    test_deep_gap_net_train_and_predict,
    test_permutation_feature_importance,
    test_batch_normalization_backprop_matches_finite_differences,
    test_deep_dataset_does_not_backfill_features_or_targets_or_current_fundamentals,
    test_train_deep_gap_model_pipeline,
)
from tests.test_backtest import (
    test_pairs_backtester_simulation,
    test_backtester_does_not_charge_inactive_initial_position,
    test_backtester_charges_both_legs_on_entry_and_exit,
)
from tests.test_ai import (
    test_prompt_and_offline_heuristic,
    test_openrouter_request_auth_and_response_validation,
)

test_suite = [
    ("test_compute_fundamental_divergence", test_compute_fundamental_divergence),
    ("test_fetch_pair_prices_rejects_invalid_pair", test_fetch_pair_prices_rejects_invalid_pair),
    ("test_fetch_pair_prices_handles_multiindex_level_order", test_fetch_pair_prices_handles_multiindex_level_order),
    ("test_ols_and_rolling_zscore", test_ols_and_rolling_zscore),
    ("test_half_life_and_hurst", test_half_life_and_hurst),
    ("test_gap_classification", test_gap_classification),
    ("test_sentiment_scoring", test_sentiment_scoring),
    ("test_support_resistance_channels_use_confirmed_pivots_and_rank_strength", test_support_resistance_channels_use_confirmed_pivots_and_rank_strength),
    ("test_support_resistance_rejects_missing_ohlc", test_support_resistance_rejects_missing_ohlc),
    ("test_deep_gap_net_train_and_predict", test_deep_gap_net_train_and_predict),
    ("test_permutation_feature_importance", test_permutation_feature_importance),
    ("test_batch_normalization_backprop_matches_finite_differences", test_batch_normalization_backprop_matches_finite_differences),
    ("test_deep_dataset_does_not_backfill_features_or_targets_or_current_fundamentals", test_deep_dataset_does_not_backfill_features_or_targets_or_current_fundamentals),
    ("test_train_deep_gap_model_pipeline", test_train_deep_gap_model_pipeline),
    ("test_pairs_backtester_simulation", test_pairs_backtester_simulation),
    ("test_backtester_does_not_charge_inactive_initial_position", test_backtester_does_not_charge_inactive_initial_position),
    ("test_backtester_charges_both_legs_on_entry_and_exit", test_backtester_charges_both_legs_on_entry_and_exit),
    ("test_prompt_and_offline_heuristic", test_prompt_and_offline_heuristic),
    ("test_openrouter_request_auth_and_response_validation", test_openrouter_request_auth_and_response_validation),
]

def main():
    print("=" * 60)
    print("RUNNING AUTOMATED UNIT & INTEGRATION TESTS")
    print("=" * 60)
    
    passed = 0
    failed = 0
    
    for name, func in test_suite:
        try:
            func()
            print(f"[PASS] {name}")
            passed += 1
        except Exception as e:
            print(f"[FAIL] {name}: {e}")
            traceback.print_exc()
            failed += 1
            
    print("=" * 60)
    print(f"RESULTS: {passed} passed, {failed} failed")
    print("=" * 60)
    
    if failed > 0:
        sys.exit(1)
    else:
        sys.exit(0)

if __name__ == "__main__":
    main()
