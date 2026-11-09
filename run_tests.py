"""
Test Runner Script for Aircraft Engine Predictive Maintenance (AEPM).

Executes all test modules in tests/, logs progress, measures execution timing,
and outputs pass/fail statistics for project validation reporting.
"""

import importlib
import logging
import sys
import time
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("run_tests")


def run_all_tests():
    """Run all pipeline and dashboard test functions and return result metrics."""
    from tests import test_dashboard, test_pipeline

    test_modules = [
        ("Pipeline Tests", test_pipeline, [
            "test_input_schema_validation",
            "test_missing_values_validation",
            "test_numeric_types_validation",
            "test_malformed_and_empty_input_handling",
            "test_feature_preparation_and_leakage_removal",
            "test_rul_prediction_validity",
            "test_health_status_classification",
            "test_end_to_end_inference_pipeline",
            "test_dashboard_utils",
        ]),
        ("Dashboard Tests", test_dashboard, [
            "test_status_badge",
            "test_df_to_csv_bytes",
            "test_load_json_safe",
            "test_plotly_chart_builders",
            "test_page_modules_import",
        ]),
    ]

    total_count = 0
    passed_count = 0
    failed_count = 0
    failures = []

    start_time = time.time()
    logger.info("Starting AEPM automated test suite execution...")

    for suite_name, module, test_fn_names in test_modules:
        logger.info("=== Executing %s ===", suite_name)
        for fn_name in test_fn_names:
            total_count += 1
            test_fn = getattr(module, fn_name, None)
            if test_fn is None:
                logger.error("  [FAIL] Test function '%s' not found in module.", fn_name)
                failed_count += 1
                failures.append(f"{suite_name}.{fn_name} (Not Found)")
                continue

            try:
                test_fn()
                passed_count += 1
                logger.info("  [PASS] %s", fn_name)
            except Exception as exc:
                failed_count += 1
                failures.append(f"{suite_name}.{fn_name}: {exc}")
                logger.error("  [FAIL] %s - Exception: %s", fn_name, exc)

    elapsed_time = round(time.time() - start_time, 3)

    logger.info("==================================================")
    logger.info("Test Suite Execution Summary:")
    logger.info("Total Tests : %d", total_count)
    logger.info("Passed      : %d", passed_count)
    logger.info("Failed      : %d", failed_count)
    logger.info("Elapsed     : %.3f seconds", elapsed_time)
    logger.info("==================================================")

    return {
        "total_tests": total_count,
        "passed_tests": passed_count,
        "failed_tests": failed_count,
        "elapsed_seconds": elapsed_time,
        "failures": failures,
    }


if __name__ == "__main__":
    results = run_all_tests()
    if results["failed_tests"] > 0:
        logger.error("Test suite FAILED with %d failures.", results["failed_tests"])
        sys.exit(1)
    else:
        logger.info("[SUCCESS] All %d tests passed successfully!", results["passed_tests"])
        sys.exit(0)
