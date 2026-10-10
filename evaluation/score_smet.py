#!/usr/bin/env python3
"""Score SMET predictions on the TRIAGE test split with the paper's metric definitions.

Ported from lib/utils.py (postprocess, average_precision_at_k, mean_average_precision,
compute_predictive_quality_metrics) and lib/smet_ranking.py (secondary-impact exclusion).
Standard library only.

Metrics
  MAP   : mean over CVEs of AP = (1/|T*|) * sum_i P@i * 1(t_i in T*)  (full ranking)
  R@k   : micro recall@k = sum_CVE |T*_CVE ∩ top-k| / sum_CVE |T*_CVE|
  P@k   : micro precision@k = sum_CVE |T*_CVE ∩ top-k| / (k * #CVEs)
"""
import argparse, csv, json, sys


def load_test_cves(path):
    with open(path, newline="", encoding="utf-8") as f:
        return [r["CVE ID"].strip() for r in csv.DictReader(f)]


def load_labels(path):
    labels = {}
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            labels.setdefault(r["CVE ID"].strip(), []).append(
                (r["attack_id"].strip(), r["mapping_type"].strip()))
    return labels


def load_predictions(path):
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    preds = {}
    for e in data:
        preds[e["target_cve"]] = e["predictions"]
    return preds


def build_entries(test_cves, labels, preds, exclude_secondary):
    entries, missing = [], []
    for cve in test_cves:
        if cve not in preds:
            missing.append(cve)
            continue
        gold = [a for a, t in labels.get(cve, []) if not (exclude_secondary and t == "secondary_impact")]
        entries.append({"target_cve": cve, "predictions": preds[cve], "true labels": gold})
    if missing:
        raise KeyError(f"{len(missing)} test CVEs have no SMET prediction, e.g. {missing[:3]}")
    return entries


def postprocess(entry):
    if len(entry["true labels"]) == 0:
        true_labels = {"N/A"}
    else:
        true_labels = set(entry["true labels"])
    predictions = entry["predictions"]
    predictions = ["N/A" if item in ["None", None] else item for item in predictions]
    return true_labels, predictions


def average_precision_at_k(relevant_items, retrieved_items, k=None):
    if not relevant_items:
        return 0.0
    total_relevant_in_ground_truth = len(relevant_items)
    if not k:
        k = len(retrieved_items)
    retrieved_items = retrieved_items[:k]
    num_relevant_found = 0
    sum_precision = 0.0
    for i, item in enumerate(retrieved_items):
        if item in relevant_items:
            num_relevant_found += 1
            precision_at_i = num_relevant_found / (i + 1)
            sum_precision += precision_at_i
            if item == "N/A":
                break
    return sum_precision / total_relevant_in_ground_truth


def mean_average_precision(data, k=None):
    ap_sum = 0.0
    for entry in data:
        relevant_items, retrieved_items = postprocess(entry)
        ap_sum += average_precision_at_k(relevant_items, retrieved_items, k)
    return ap_sum / len(data)


def predictive_quality(data, k):
    num_rel_at_k_total = 0
    num_rel_total = 0
    per_cve_recall = []
    for entry in data:
        true_labels, predictions = postprocess(entry)
        top = predictions[:k]
        rel_at_k = len(true_labels.intersection(set(top)))
        num_rel_at_k_total += rel_at_k
        num_rel_total += len(true_labels)
        per_cve_recall.append(rel_at_k / len(true_labels) if true_labels else 0.0)
    precision = num_rel_at_k_total / (k * len(data))
    recall_micro = num_rel_at_k_total / num_rel_total
    recall_macro = sum(per_cve_recall) / len(per_cve_recall)
    return precision, recall_micro, recall_macro


def score(entries):
    p5, r5, r5_macro = predictive_quality(entries, 5)
    p10, r10, r10_macro = predictive_quality(entries, 10)
    return {
        "n_cves": len(entries),
        "MAP": mean_average_precision(entries),
        "R@5": r5,
        "R@10": r10,
        "P@10": p10,
        "R@5_macro": r5_macro,
        "R@10_macro": r10_macro,
    }


def paper_reference(path):
    with open(path, encoding="utf-8") as f:
        ref = json.load(f)["comparison_smet_vs_triage"]
    return {
        "included": ref["secondary_impacts_included"]["SMET"],
        "excluded": ref["secondary_impacts_excluded"]["SMET"],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions", required=True, help="SMET predictions JSON (target_cve, predictions)")
    ap.add_argument("--label", required=True, help="name for this prediction source")
    ap.add_argument("--test-cves", default="data/pre_processed/cves_test.csv")
    ap.add_argument("--labels", default="data/pre_processed/labeled_cve_to_attack.csv")
    ap.add_argument("--reference", default="evaluation/paper_reference_metrics.json")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    test_cves = load_test_cves(args.test_cves)
    labels = load_labels(args.labels)
    preds = load_predictions(args.predictions)
    ref = paper_reference(args.reference)

    result = {"source": args.label, "predictions_file": args.predictions, "settings": {}}
    for setting, exclude in [("included", False), ("excluded", True)]:
        entries = build_entries(test_cves, labels, preds, exclude)
        s = score(entries)
        paper = ref[setting]
        result["settings"][setting] = {
            "computed": {k: round(v, 4) if isinstance(v, float) else v for k, v in s.items()},
            "paper": paper,
            "diff_computed_minus_paper": {m: round(s[m] - paper[m], 4) for m in ["MAP", "R@5", "R@10"]},
        }

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print(f"== {args.label}")
    for setting in ["included", "excluded"]:
        r = result["settings"][setting]
        c, p = r["computed"], r["paper"]
        print(f"  secondary {setting}: n={c['n_cves']}")
        for m in ["MAP", "R@10", "R@5"]:
            print(f"    {m:6s} computed={c[m]:.4f}  paper={p[m]:.2f}  diff={c[m]-p[m]:+.4f}")


if __name__ == "__main__":
    sys.exit(main())
