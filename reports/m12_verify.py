"""
Milestone 12 — Full end-to-end verification script for AEPM.
Verifies model, inference pipeline, leakage exclusion, health classification,
CSV export, and deployment artifacts.
"""
import sys
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src import config
from src.predictor import RULPredictor
from src.inference import prepare_inference_features, run_inference_pipeline
from src.health_status import get_health_summary
from app.utils import df_to_csv_bytes

results_data = {}

print("=" * 60)
print("  AEPM — Milestone 12 Full Verification")
print("=" * 60)

# ── 1. Model file ─────────────────────────────────────────────────
print("\n[1] MODEL VERIFICATION")
model_path = config.MODELS_DIR / "random_forest_v2.joblib"
model_exists = model_path.exists()
model_size_mb = round(model_path.stat().st_size / 1_000_000, 2) if model_exists else 0
print(f"  Model file exists : {model_exists}")
print(f"  Model file size   : {model_size_mb} MB")

predictor = RULPredictor(model_path=model_path)
model_features = list(predictor.model.feature_names_in_)
feature_count = len(model_features)
leakage_present = "life_percentage" in model_features
algorithm = type(predictor.model).__name__
print(f"  Algorithm         : {algorithm}")
print(f"  Feature count     : {feature_count}")
print(f"  Leakage present   : {leakage_present}")
print(f"  Feature[0]        : {model_features[0]}")
print(f"  Feature[-1]       : {model_features[-1]}")

results_data["model"] = {
    "path": str(model_path),
    "exists": model_exists,
    "size_mb": model_size_mb,
    "algorithm": algorithm,
    "feature_count": feature_count,
    "leakage_present": leakage_present,
}

# ── 2. End-to-end on test_FD001.txt ─────────────────────────────
print("\n[2] END-TO-END INFERENCE ON test_FD001.txt")
test_path = config.TEST_FD001_PATH
df_raw = pd.read_csv(
    test_path, sep=r"\s+", header=None,
    names=config.COLUMN_NAMES, engine="python"
)
engine_count = int(df_raw["unit_nr"].nunique())
total_rows = len(df_raw)
print(f"  Input rows    : {total_rows}")
print(f"  Input engines : {engine_count}")

results_df = run_inference_pipeline(df_raw, model_path=model_path)
all_finite = bool(np.isfinite(results_df["predicted_rul"]).all())
all_nonneg = bool((results_df["predicted_rul"] >= 0).all())
rul_min = round(float(results_df["predicted_rul"].min()), 2)
rul_max = round(float(results_df["predicted_rul"].max()), 2)
rul_mean = round(float(results_df["predicted_rul"].mean()), 2)
rul_median = round(float(results_df["predicted_rul"].median()), 2)

print(f"  Output rows   : {len(results_df)}")
print(f"  Output cols   : {list(results_df.columns)}")
print(f"  All finite    : {all_finite}")
print(f"  All non-neg   : {all_nonneg}")
print(f"  RUL min       : {rul_min}")
print(f"  RUL max       : {rul_max}")
print(f"  RUL mean      : {rul_mean}")
print(f"  RUL median    : {rul_median}")

# Sample predictions (first 5 engines, last cycle)
last_cycles = results_df.groupby("unit_nr").last().reset_index()
print("\n  Sample final-cycle predictions (first 5 engines):")
print(f"  {'Engine':>8} {'Pred RUL':>12} {'Health':>12}")
for _, row in last_cycles.head(5).iterrows():
    print(f"  {int(row['unit_nr']):>8} {row['predicted_rul']:>12.2f} {row['health_status']:>12}")

results_data["inference"] = {
    "input_rows": total_rows,
    "engine_count": engine_count,
    "output_rows": len(results_df),
    "all_finite": all_finite,
    "all_non_negative": all_nonneg,
    "rul_min": rul_min,
    "rul_max": rul_max,
    "rul_mean": rul_mean,
    "rul_median": rul_median,
}

# ── 3. Health classification ─────────────────────────────────────
print("\n[3] HEALTH CLASSIFICATION (final cycle per engine)")
summary = get_health_summary(last_cycles)
health_dist = {}
for _, row in summary.iterrows():
    status = str(row["health_status"])
    count = int(row["count"])
    pct = round(float(row["percentage"]), 1)
    print(f"  {status:10s} : {count:3d} engines ({pct}%)")
    health_dist[status] = {"count": count, "percentage": pct}

results_data["health_classification"] = health_dist

# ── 4. CSV export ─────────────────────────────────────────────────
print("\n[4] CSV EXPORT")
csv_bytes = df_to_csv_bytes(results_df)
csv_size = len(csv_bytes)
csv_header_ok = b"unit_nr,time_cycles,predicted_rul,health_status" in csv_bytes
print(f"  CSV size      : {csv_size:,} bytes")
print(f"  CSV header OK : {csv_header_ok}")
results_data["csv_export"] = {
    "size_bytes": csv_size,
    "header_correct": csv_header_ok,
}

# ── 5. Deployment artifacts ───────────────────────────────────────
print("\n[5] DEPLOYMENT ARTIFACTS")
artifact_list = [
    "requirements.txt",
    "requirements-dev.txt",
    "setup.py",
    "Dockerfile",
    ".dockerignore",
    ".streamlit/config.toml",
    "run_dashboard.sh",
    "run_tests.bat",
    "run_dashboard.bat",
    ".github/workflows/ci.yml",
    "CHANGELOG.md",
    "README.md",
]
artifacts_status = {}
all_present = True
for name in artifact_list:
    p = Path(name)
    exists = p.exists()
    size = p.stat().st_size if exists else 0
    status_str = "OK     " if exists else "MISSING"
    print(f"  [{status_str}] {name}")
    artifacts_status[name] = {"present": exists, "size_bytes": size}
    if not exists:
        all_present = False

results_data["deployment_artifacts"] = {
    "all_present": all_present,
    "details": artifacts_status,
}

# ── 6. Feature ordering check ─────────────────────────────────────
print("\n[6] FEATURE ORDERING CHECK")
feat_df = prepare_inference_features(df_raw)
inference_cols = [c for c in feat_df.columns if c in model_features]
ordering_match = inference_cols == model_features[:len(inference_cols)]
print(f"  Inference feature cols available : {len(inference_cols)}")
print(f"  Model expects                    : {len(model_features)}")
print(f"  Ordering consistent              : {ordering_match}")

results_data["feature_ordering"] = {
    "inference_features_available": len(inference_cols),
    "model_expected_features": len(model_features),
    "ordering_consistent": ordering_match,
}

print("\n" + "=" * 60)
print("  VERIFICATION COMPLETE")
print("=" * 60)

# Persist for collection
import json
Path("reports/m12_verify_raw.json").write_text(json.dumps(results_data, indent=2))
print("\nRaw results saved to reports/m12_verify_raw.json")
