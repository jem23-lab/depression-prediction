import pandas as pd
import numpy as np
from scipy.stats import kruskal
import os

def main():
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    csv_path = os.path.join(root_dir, "logs", "evaluation_records.csv")

    if not os.path.exists(csv_path):
        print(f"File not found: {csv_path}")
        return

    df = pd.read_csv(csv_path)

    # We need to compute averages per method for the 4 metrics
    metrics = ["rating_clarity", "rating_correctness", "rating_helpfulness", "rating_trust"]
    methods = ["RAG", "HYBRID", "SHAP", "COUNTERFACTUAL"]

    # Filter to known methods
    df = df[df["selected_use_case_name"].isin(methods)].copy()

    print("=" * 80)
    print("KRUSKAL-WALLIS TEST ACROSS METHODS")
    print("=" * 80)

    for metric in metrics + ["rating_overall_avg"]:
        groups = [group[metric].dropna().values for name, group in df.groupby("selected_use_case_name") if len(group[metric].dropna()) > 0]
        if len(groups) > 1:
            stat, p = kruskal(*groups)
            print(f"{metric.replace('rating_', '').capitalize():<15} : H={stat:.3f}, p={p:.4f}")
        else:
            print(f"{metric.replace('rating_', '').capitalize():<15} : Not enough groups for K-W test")

    print("\n" + "=" * 80)
    print("AVERAGE METRICS BY TOOL")
    print("=" * 80)
    mean_df = df.groupby("selected_use_case_name")[metrics].mean().round(3)
    # Rename columns to match image
    mean_df.columns = ["Clarity", "Correctness", "Helpfulness", "Trust"]
    # Reindex to match the order: RAG, HYBRID, SHAP, COUNTERFACTUAL
    existing_methods = [m for m in methods if m in mean_df.index]
    mean_df = mean_df.loc[existing_methods]
    print(mean_df)

    # BEST TOOL BY SEVERITY & CONFIDENCE based on rating_overall_avg
    print("\n" + "=" * 80)
    print("BEST TOOL BY SEVERITY & CONFIDENCE (Overall Avg)")
    print("=" * 80)

    results = []

    # Severities
    severities = ["severe", "moderate", "not depression"]
    for sev in severities:
        sub_df = df[df["prediction_label"] == sev]
        if not sub_df.empty:
            best_avg = sub_df.groupby("selected_use_case_name")["rating_overall_avg"].mean().max()
            best_tool = sub_df.groupby("selected_use_case_name")["rating_overall_avg"].mean().idxmax()
            results.append({"Severity/Condition": sev.capitalize(), "Best Tool": best_tool, "Overall Avg": round(best_avg, 3)})

    # Confidence
    # Lower (< 0.75)
    sub_df_low = df[df["prediction_confidence"] < 0.75]
    if not sub_df_low.empty:
        best_avg = sub_df_low.groupby("selected_use_case_name")["rating_overall_avg"].mean().max()
        best_tool = sub_df_low.groupby("selected_use_case_name")["rating_overall_avg"].mean().idxmax()
        results.append({"Severity/Condition": "Lower confidence (< 0.75)", "Best Tool": best_tool, "Overall Avg": round(best_avg, 3)})

    # Higher (> 0.75)
    sub_df_high = df[df["prediction_confidence"] >= 0.75]
    if not sub_df_high.empty:
        best_avg = sub_df_high.groupby("selected_use_case_name")["rating_overall_avg"].mean().max()
        best_tool = sub_df_high.groupby("selected_use_case_name")["rating_overall_avg"].mean().idxmax()
        results.append({"Severity/Condition": "Higher confidence (> 0.75)", "Best Tool": best_tool, "Overall Avg": round(best_avg, 3)})

    res_df = pd.DataFrame(results)
    print(res_df.to_string(index=False))

    # Save outputs to txt
    out_dir = os.path.join(root_dir, "analysis", "analysis_plots")
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, "statistical_analysis_results.txt")
    with open(out_file, "w") as f:
        f.write("KRUSKAL-WALLIS TEST ACROSS METHODS\n========================================\n")
        for metric in metrics + ["rating_overall_avg"]:
            groups = [group[metric].dropna().values for name, group in df.groupby("selected_use_case_name") if len(group[metric].dropna()) > 0]
            if len(groups) > 1:
                stat, p = kruskal(*groups)
                f.write(f"{metric.replace('rating_', '').capitalize():<15} : H={stat:.3f}, p={p:.4f}\n")
        f.write("\nAVERAGE METRICS BY TOOL\n========================================\n")
        f.write(mean_df.to_string())
        f.write("\n\nBEST TOOL BY SEVERITY & CONFIDENCE (Overall Avg)\n========================================\n")
        f.write(res_df.to_string(index=False))

    print(f"\nResults saved to {out_file}")

if __name__ == '__main__':
    main()
