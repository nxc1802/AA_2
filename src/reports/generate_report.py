# ==============================================================================
# IMPORTS & DEPENDENCIES
# ==============================================================================
import os
import sys
import logging
import json
import pandas as pd

# ==============================================================================
# CONFIGURABLE PARAMETERS & PATHS
# ==============================================================================
RESULT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../result"))
METRICS_DIR = os.path.join(RESULT_DIR, "metrics")
REPORTS_DIR = os.path.join(RESULT_DIR, "reports")
LOG_DIR = os.path.join(RESULT_DIR, "logs")

os.makedirs(METRICS_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# ==============================================================================
# LOGGING SETUP
# ==============================================================================
logger = logging.getLogger("GenerateReport")
logger.setLevel(logging.INFO)
formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s")

file_handler = logging.FileHandler(os.path.join(LOG_DIR, "generate_report.log"), encoding="utf-8")
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

# ==============================================================================
# PIVOT TABLES & REPORT GENERATION IMPLEMENTATION
# ==============================================================================
def generate_reports(df):
    """Generates pivot tables and exports full metrics across K values."""
    if df is None or len(df) == 0:
        logger.warning("Empty DataFrame passed to generate_reports.")
        return

    df_sparse = df[pd.to_numeric(df["K"], errors="coerce").notna()].copy()
    if len(df_sparse) > 0:
        df_sparse["K"] = pd.to_numeric(df_sparse["K"])

        idx_cols = ["Group", "Attack Method"] if "Group" in df_sparse.columns else ["Attack Method"]

        asr_pivot = df_sparse.pivot(index=idx_cols, columns="K", values="ASR (%)").reset_index()
        asr_pivot.to_csv(os.path.join(METRICS_DIR, "asr_k_pivot.csv"), index=False)
        with open(os.path.join(METRICS_DIR, "asr_k_pivot.json"), "w", encoding="utf-8") as f:
            json.dump(asr_pivot.to_dict(orient="records"), f, indent=4)

        rob_pivot = df_sparse.pivot(index=idx_cols, columns="K", values="Robust Acc (%)").reset_index()
        rob_pivot.to_csv(os.path.join(METRICS_DIR, "robust_accuracy_k_pivot.csv"), index=False)
        with open(os.path.join(METRICS_DIR, "robust_accuracy_k_pivot.json"), "w", encoding="utf-8") as f:
            json.dump(rob_pivot.to_dict(orient="records"), f, indent=4)

        iter_pivot = df_sparse.pivot(index=idx_cols, columns="K", values="Avg Iterations").reset_index()
        iter_pivot.to_csv(os.path.join(METRICS_DIR, "iterations_k_pivot.csv"), index=False)
        with open(os.path.join(METRICS_DIR, "iterations_k_pivot.json"), "w", encoding="utf-8") as f:
            json.dump(iter_pivot.to_dict(orient="records"), f, indent=4)

    cols = ["Group", "Attack Method", "K", "PSNR (dB)", "SSIM", "LPIPS", "Avg L0", "Avg L2", "Avg L_inf", "Avg Iterations"]
    cols = [c for c in cols if c in df.columns]
    img_quality = df[cols].copy()
    img_quality.to_csv(os.path.join(METRICS_DIR, "image_quality_metrics.csv"), index=False)
    with open(os.path.join(METRICS_DIR, "image_quality_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(img_quality.to_dict(orient="records"), f, indent=4)

    logger.info("=== SUMMARY PIVOT TABLES GENERATED ===")
    return compile_attack_report()

def compile_attack_report():
    json_path = os.path.join(METRICS_DIR, "full_attack_metrics.json")
    if not os.path.isfile(json_path):
        json_path = os.path.join(METRICS_DIR, "full_attack_benchmark.json")

    if not os.path.isfile(json_path):
        res_list = []
        if os.path.exists(METRICS_DIR):
            for fname in os.listdir(METRICS_DIR):
                if fname.endswith(".json") and not fname.endswith("pivot.json"):
                    fpath = os.path.join(METRICS_DIR, fname)
                    with open(fpath, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        if isinstance(data, list):
                            res_list.extend(data)
                        else:
                            res_list.append(data)
        if len(res_list) == 0:
            logger.warning(f"No benchmark JSON metrics found in '{METRICS_DIR}'.")
            return None
        df = pd.DataFrame(res_list)
    else:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        df = pd.DataFrame(data)

    md_report = "# Comprehensive Sparse Adversarial Attack Benchmark Report (Groups A, B, C)\n\n"
    md_report += f"> **Generated At:** `{pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}`\n\n"
    md_report += "## 1. Attack Performance Summary Table\n\n"
    md_report += df.to_markdown(index=False)
    md_report += "\n\n---\n"
    md_report += "## 2. Experimental Group Categorization\n"
    md_report += "- **Group A (Direct K-Sweep):** JSMA, OnePixel, CornerSearch, SAIF, PGD0, Sparse-PGD, Sparse-RS, BruSLe, IPFSA, GradientGuidance, CPA, FCSA, FMSA-budgeted, HSA-budgeted\n"
    md_report += "- **Group B (Minimal Support Optimization & ASR@K):** SparseFool, SigmaZero, Homotopy, GSE, Pixle, FMSA-minimal-support\n"
    md_report += "- **Group C (Non-pixel-K Attacks):** FGSM, BIM, PGD, SFA (Spectral Frequency Attack)\n"

    out_md = os.path.join(REPORTS_DIR, "summary_attack_report.md")
    with open(out_md, "w", encoding="utf-8") as f:
        f.write(md_report)

    logger.info(f"Generated Markdown summary report at: {out_md}")
    return out_md

if __name__ == "__main__":
    logger.info("=== Running Standalone Report Generator Test ===")
    report_file = compile_attack_report()
    logger.info(f"Report written to {report_file}")
