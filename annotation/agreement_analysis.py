#!/usr/bin/env python3
"""Inter-Annotator Agreement & Data Quality Analysis.

Computes Cohen's Kappa, Krippendorff's Alpha, confusion matrices,
disagreement analysis, and sample representativeness metrics
between two annotators' checkpoint files.

Usage:
    python annotation/agreement_analysis.py \
      --annotator1 annotation/annotation_oracle_checkpoint_Feng.json \
      --annotator2 annotation/annotation_oracle_checkpoint_Kalye.json \
      --gold Gold_Mention_Oracle/software_citation_ground_truth.json \
      --output_dir annotation/agreement_results/
"""

import argparse
import csv
import json
import logging
import math
import os
from collections import Counter, defaultdict
from pathlib import Path

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────
# Statistical Functions (no external deps)
# ──────────────────────────────────────────────────────────────────────

def cohens_kappa(y1: list, y2: list) -> float:
    """Cohen's Kappa for two raters with binary/nominal labels."""
    n = len(y1)
    if n == 0:
        return 0.0
    labels = sorted(set(y1) | set(y2))
    if len(labels) <= 1:
        return 1.0  # perfect agreement on single label

    # Build confusion matrix
    matrix = defaultdict(int)
    for a, b in zip(y1, y2):
        matrix[(a, b)] += 1

    po = sum(matrix[(l, l)] for l in labels) / n  # observed agreement

    # Expected agreement
    pe = 0.0
    for l in labels:
        row_sum = sum(matrix[(l, l2)] for l2 in labels) / n
        col_sum = sum(matrix[(l2, l)] for l2 in labels) / n
        pe += row_sum * col_sum

    if pe >= 1.0:
        return 1.0
    return (po - pe) / (1.0 - pe)


def krippendorffs_alpha_binary(y1: list, y2: list) -> float:
    """Krippendorff's Alpha for binary data, 2 coders, no missing."""
    n = len(y1)
    if n == 0:
        return 0.0

    # Coincidence matrix approach
    labels = sorted(set(y1) | set(y2))
    if len(labels) <= 1:
        return 1.0

    # Build coincidence matrix
    coincidence = defaultdict(float)
    for a, b in zip(y1, y2):
        coincidence[(a, b)] += 1
        coincidence[(b, a)] += 1

    # Observed disagreement
    total_pairs = 2 * n  # each item contributes 2 values
    do = 0.0
    for c in labels:
        for k in labels:
            if c != k:
                do += coincidence[(c, k)]
    do /= total_pairs

    # Expected disagreement
    n_c = {}
    for c in labels:
        n_c[c] = sum(coincidence[(c, k)] for k in labels)

    de = 0.0
    for c in labels:
        for k in labels:
            if c != k:
                de += n_c[c] * n_c[k]
    de /= (total_pairs * (total_pairs - 1))

    if de == 0:
        return 1.0
    return 1.0 - do / de


def confusion_matrix_2x2(y1: list, y2: list, pos_label=1) -> dict:
    """Build 2x2 confusion matrix and return as dict."""
    tp = fp = fn = tn = 0
    for a, b in zip(y1, y2):
        a_pos = (a == pos_label)
        b_pos = (b == pos_label)
        if a_pos and b_pos:
            tp += 1
        elif a_pos and not b_pos:
            fp += 1  # annotator1 says pos, annotator2 says neg
        elif not a_pos and b_pos:
            fn += 1
        else:
            tn += 1
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "total": tp + fp + fn + tn}


def interpret_kappa(k: float) -> str:
    """Landis & Koch (1977) interpretation."""
    if k < 0:
        return "Poor"
    elif k <= 0.20:
        return "Slight"
    elif k <= 0.40:
        return "Fair"
    elif k <= 0.60:
        return "Moderate"
    elif k <= 0.80:
        return "Substantial"
    else:
        return "Almost Perfect"


def ks_test_2sample(sample1: list, sample2: list) -> dict:
    """Two-sample Kolmogorov-Smirnov test (no scipy needed)."""
    all_vals = sorted(set(sample1 + sample2))
    n1, n2 = len(sample1), len(sample2)
    if n1 == 0 or n2 == 0:
        return {"statistic": 0.0, "p_value": 1.0}

    c1 = Counter(sample1)
    c2 = Counter(sample2)

    ecdf1 = ecdf2 = 0.0
    d_max = 0.0
    for v in all_vals:
        ecdf1 += c1.get(v, 0) / n1
        ecdf2 += c2.get(v, 0) / n2
        d_max = max(d_max, abs(ecdf1 - ecdf2))

    # Approximate p-value (Kolmogorov distribution)
    en = math.sqrt(n1 * n2 / (n1 + n2))
    lam = (en + 0.12 + 0.11 / en) * d_max
    if lam == 0:
        p = 1.0
    else:
        # Asymptotic formula
        p = 2.0 * math.exp(-2.0 * lam * lam)
        p = max(0.0, min(1.0, p))

    return {"statistic": round(d_max, 4), "p_value": round(p, 4)}


# ──────────────────────────────────────────────────────────────────────
# Data Loading
# ──────────────────────────────────────────────────────────────────────

ANNOTATION_DIMS = {
    "agreement": {"pos": 1, "neg": 0, "group": "overall"},
    "metadata_correct": {"pos": "yes", "neg": "no", "group": "overall"},
    "hallucination": {"pos": "no", "neg": "yes", "group": "overall"},  # NOTE: "no hallucination" is positive
    "citation_correct": {"pos": "yes", "neg": "no", "group": "overall"},
    "attribution": {"pos": 1, "neg": 0, "group": "principle"},
    "identification": {"pos": 1, "neg": 0, "group": "principle"},
    "accessibility": {"pos": 1, "neg": 0, "group": "principle"},
    "specificity": {"pos": 1, "neg": 0, "group": "principle"},
}


def load_annotations(path: str) -> dict:
    """Load annotation checkpoint and return {(pmcid, sw_name): annotation}."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    result = {}
    for pub in data.get("publications", []):
        pmcid = pub.get("publication_id", "")
        for sw in pub.get("software_citations", []):
            sw_name = sw.get("software_name", "")
            ann = sw.get("annotation", {})
            if ann.get("completed"):
                key = (pmcid, sw_name)
                # Flatten principle_score into top-level
                flat = dict(ann)
                ps = ann.get("principle_score", {})
                for k in ("attribution", "identification", "accessibility", "specificity"):
                    flat[k] = ps.get(k)
                result[key] = flat
    return result


def load_gold(path: str) -> dict:
    """Load gold data and return {pmcid: [software_names]}."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    result = {}
    sw_per_paper = []
    all_sw_names = []
    for pub in data:
        pmcid = pub.get("pmcid", "")
        names = [c.get("software_name", "") for c in pub.get("software_citations", [])]
        result[pmcid] = names
        sw_per_paper.append(len(names))
        all_sw_names.extend(names)
    return result, sw_per_paper, all_sw_names


# ──────────────────────────────────────────────────────────────────────
# Per-Annotator Statistics
# ──────────────────────────────────────────────────────────────────────

def per_annotator_stats(ann: dict, name: str) -> dict:
    """Compute per-annotator quality metrics from annotation data.

    For each dimension, compute the 'positive' rate:
    - agreement=1 → correct detection
    - metadata_correct='yes' → metadata is correct
    - hallucination='yes' → hallucination present (reported as hallucination_rate)
    - citation_correct='yes' → citation text is correct
    - attribution/identification/accessibility/specificity=1 → principle satisfied
    """
    n_total = len(ann)
    if n_total == 0:
        return {"name": name, "n_items": 0}

    # Count positive values for each dimension
    dim_counts = {}
    for dim, cfg in ANNOTATION_DIMS.items():
        pos_label = cfg["pos"]
        values = [a.get(dim) for a in ann.values() if a.get(dim) is not None]
        n = len(values)
        n_pos = sum(1 for v in values if v == pos_label)
        rate = n_pos / n if n else 0
        dim_counts[dim] = {"n": n, "n_positive": n_pos, "rate": round(rate, 4)}

    # Derive human-readable summary
    result = {
        "name": name,
        "n_items": n_total,
        "agreement_rate": dim_counts.get("agreement", {}).get("rate", 0),
        "metadata_correct_rate": dim_counts.get("metadata_correct", {}).get("rate", 0),
        "hallucination_rate": round(1.0 - dim_counts.get("hallucination", {}).get("rate", 0), 4),
        "citation_correct_rate": dim_counts.get("citation_correct", {}).get("rate", 0),
        "force11_scores": {
            "attribution": dim_counts.get("attribution", {}).get("rate", 0),
            "identification": dim_counts.get("identification", {}).get("rate", 0),
            "accessibility": dim_counts.get("accessibility", {}).get("rate", 0),
            "specificity": dim_counts.get("specificity", {}).get("rate", 0),
        },
        "force11_overall": round(
            sum(dim_counts.get(p, {}).get("rate", 0)
                for p in ("attribution", "identification", "accessibility", "specificity")) / 4, 4),
        "dimension_detail": dim_counts,
    }
    return result


# ──────────────────────────────────────────────────────────────────────
# Core Analysis
# ──────────────────────────────────────────────────────────────────────

def compute_agreement(ann1: dict, ann2: dict) -> dict:
    """Compute IAA metrics across all dimensions."""
    # Find common keys
    common_keys = sorted(set(ann1.keys()) & set(ann2.keys()))
    logger.info(f"Common items: {len(common_keys)}")

    results = {}
    for dim, cfg in ANNOTATION_DIMS.items():
        pos = cfg["pos"]
        # Extract values for this dimension
        vals1 = []
        vals2 = []
        for key in common_keys:
            v1 = ann1[key].get(dim)
            v2 = ann2[key].get(dim)
            if v1 is not None and v2 is not None:
                vals1.append(v1)
                vals2.append(v2)

        if not vals1:
            results[dim] = {"n": 0, "percent_agreement": 0, "cohens_kappa": 0,
                            "krippendorffs_alpha": 0, "interpretation": "N/A"}
            continue

        n = len(vals1)
        agree_count = sum(a == b for a, b in zip(vals1, vals2))
        pct = agree_count / n

        kappa = cohens_kappa(vals1, vals2)
        alpha = krippendorffs_alpha_binary(vals1, vals2)
        cm = confusion_matrix_2x2(vals1, vals2, pos_label=pos)

        # Positive rates per annotator
        pos_rate1 = sum(1 for v in vals1 if v == pos) / n
        pos_rate2 = sum(1 for v in vals2 if v == pos) / n

        results[dim] = {
            "n": n,
            "percent_agreement": round(pct, 4),
            "cohens_kappa": round(kappa, 4),
            "krippendorffs_alpha": round(alpha, 4),
            "interpretation": interpret_kappa(kappa),
            "confusion_matrix": cm,
            "pos_rate_annotator1": round(pos_rate1, 4),
            "pos_rate_annotator2": round(pos_rate2, 4),
            "group": cfg["group"],
        }

    # Overall averages
    kappas = [r["cohens_kappa"] for r in results.values() if r["n"] > 0]
    alphas = [r["krippendorffs_alpha"] for r in results.values() if r["n"] > 0]
    pcts = [r["percent_agreement"] for r in results.values() if r["n"] > 0]

    overall = {
        "n_items": len(common_keys),
        "n_dimensions": len(ANNOTATION_DIMS),
        "mean_percent_agreement": round(sum(pcts) / len(pcts), 4) if pcts else 0,
        "mean_cohens_kappa": round(sum(kappas) / len(kappas), 4) if kappas else 0,
        "mean_krippendorffs_alpha": round(sum(alphas) / len(alphas), 4) if alphas else 0,
        "interpretation": interpret_kappa(sum(kappas) / len(kappas)) if kappas else "N/A",
    }

    return {"dimensions": results, "overall": overall}


def compute_disagreements(ann1: dict, ann2: dict, name1: str, name2: str) -> list:
    """Find all disagreements between annotators."""
    common_keys = sorted(set(ann1.keys()) & set(ann2.keys()))
    items = []

    for key in common_keys:
        pmcid, sw_name = key
        a1 = ann1[key]
        a2 = ann2[key]

        agreed = []
        disagreed = []
        for dim in ANNOTATION_DIMS:
            v1 = a1.get(dim)
            v2 = a2.get(dim)
            if v1 is not None and v2 is not None:
                if v1 == v2:
                    agreed.append(dim)
                else:
                    disagreed.append(dim)

        items.append({
            "pmcid": pmcid,
            "software_name": sw_name,
            "agreed_dims": agreed,
            "disagreed_dims": disagreed,
            "n_disagreed": len(disagreed),
            name1: {d: a1.get(d) for d in ANNOTATION_DIMS},
            name2: {d: a2.get(d) for d in ANNOTATION_DIMS},
            f"note_{name1}": a1.get("annotator_note", ""),
            f"note_{name2}": a2.get("annotator_note", ""),
        })

    return items


def compute_sample_representativeness(sample_pmcids: set, gold_data: dict,
                                       gold_sw_per_paper: list,
                                       gold_all_sw: list) -> dict:
    """Compare sample distribution vs full dataset."""
    # Sample sw/paper distribution
    sample_sw_per_paper = [len(gold_data[p]) for p in sample_pmcids if p in gold_data]

    # Full dataset distribution
    full_sw_per_paper = gold_sw_per_paper

    # KS test
    ks = ks_test_2sample(sample_sw_per_paper, full_sw_per_paper)

    # Software name coverage
    sample_sw_names = set()
    for p in sample_pmcids:
        if p in gold_data:
            for n in gold_data[p]:
                sample_sw_names.add(n.lower())

    full_sw_names = set(n.lower() for n in gold_all_sw)
    coverage = len(sample_sw_names & full_sw_names) / len(full_sw_names) if full_sw_names else 0

    # Top-N overlap
    full_counter = Counter(n.lower() for n in gold_all_sw)
    sample_counter = Counter()
    for p in sample_pmcids:
        if p in gold_data:
            for n in gold_data[p]:
                sample_counter[n.lower()] += 1

    top20_full = set(n for n, _ in full_counter.most_common(20))
    top20_sample = set(n for n, _ in sample_counter.most_common(20))
    top_overlap = len(top20_full & top20_sample) / len(top20_full) if top20_full else 0

    # Distribution stats
    def dist_stats(vals):
        if not vals:
            return {}
        s = sorted(vals)
        n = len(s)
        return {
            "n": n, "min": s[0], "max": s[-1],
            "mean": round(sum(s) / n, 2),
            "median": s[n // 2],
            "distribution": dict(sorted(Counter(s).items())),
        }

    return {
        "sample_size": len(sample_pmcids),
        "full_size": len(gold_data),
        "sample_ratio": round(len(sample_pmcids) / len(gold_data), 4) if gold_data else 0,
        "sw_per_paper_sample": dist_stats(sample_sw_per_paper),
        "sw_per_paper_full": dist_stats(full_sw_per_paper),
        "ks_test": ks,
        "unique_sw_coverage": round(coverage, 4),
        "top20_software_overlap": round(top_overlap, 4),
        "sample_unique_sw": len(sample_sw_names),
        "full_unique_sw": len(full_sw_names),
    }


def compute_consensus(ann1: dict, ann2: dict) -> dict:
    """Build consensus labels: agreed items get the label, disagreed items flagged."""
    common_keys = sorted(set(ann1.keys()) & set(ann2.keys()))
    consensus = {}

    for key in common_keys:
        a1 = ann1[key]
        a2 = ann2[key]
        item = {"pmcid": key[0], "software_name": key[1]}

        for dim, cfg in ANNOTATION_DIMS.items():
            v1 = a1.get(dim)
            v2 = a2.get(dim)
            if v1 == v2:
                item[dim] = {"value": v1, "status": "agreed"}
            else:
                # Conservative: flag as needs_review
                item[dim] = {"value": None, "status": "disagreed",
                             "annotator1": v1, "annotator2": v2}
        consensus[key] = item

    # Summary
    total = len(common_keys)
    full_agree = sum(1 for k in common_keys
                     if all(ann1[k].get(d) == ann2[k].get(d) for d in ANNOTATION_DIMS))

    return {
        "total_items": total,
        "fully_agreed": full_agree,
        "partial_disagree": total - full_agree,
        "fully_agreed_pct": round(full_agree / total, 4) if total else 0,
        "items": consensus,
    }


# ──────────────────────────────────────────────────────────────────────
# Output Writers
# ──────────────────────────────────────────────────────────────────────

def write_summary(output_dir: Path, agreement: dict, sample_repr: dict,
                  consensus_summary: dict, name1: str, name2: str,
                  annotator_stats: list = None):
    """Write summary.json."""
    summary = {
        "annotators": [name1, name2],
        "annotator_statistics": annotator_stats or [],
        "agreement": agreement,
        "sample_representativeness": sample_repr,
        "consensus": {
            "total_items": consensus_summary["total_items"],
            "fully_agreed": consensus_summary["fully_agreed"],
            "partial_disagree": consensus_summary["partial_disagree"],
            "fully_agreed_pct": consensus_summary["fully_agreed_pct"],
        },
    }
    path = output_dir / "summary.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=str)
    logger.info(f"Written: {path}")


def write_per_item(output_dir: Path, items: list):
    """Write per_item_agreement.jsonl."""
    path = output_dir / "per_item_agreement.jsonl"
    with open(path, "w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False, default=str) + "\n")
    logger.info(f"Written: {path} ({len(items)} items)")


def write_disagreements_csv(output_dir: Path, items: list, name1: str, name2: str):
    """Write disagreements.csv for human review."""
    disagreed = [i for i in items if i["n_disagreed"] > 0]
    path = output_dir / "disagreements.csv"
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        header = ["pmcid", "software_name", "dimension", f"{name1}_value",
                   f"{name2}_value", f"note_{name1}", f"note_{name2}"]
        writer.writerow(header)
        for item in disagreed:
            for dim in item["disagreed_dims"]:
                writer.writerow([
                    item["pmcid"], item["software_name"], dim,
                    item[name1].get(dim, ""), item[name2].get(dim, ""),
                    item.get(f"note_{name1}", ""), item.get(f"note_{name2}", ""),
                ])
    logger.info(f"Written: {path} ({len(disagreed)} items with disagreements)")


def write_metrics_csv(output_dir: Path, agreement: dict,
                      annotator_stats: list = None):
    """Write metrics.csv — flat table of agreement + per-annotator metrics."""
    path = output_dir / "metrics.csv"
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["dimension", "group", "n", "percent_agreement",
                          "cohens_kappa", "krippendorffs_alpha", "interpretation",
                          "pos_rate_annotator1", "pos_rate_annotator2"])
        for dim, vals in agreement["dimensions"].items():
            writer.writerow([
                dim, vals.get("group", ""), vals["n"],
                vals["percent_agreement"], vals["cohens_kappa"],
                vals["krippendorffs_alpha"], vals["interpretation"],
                vals.get("pos_rate_annotator1", ""), vals.get("pos_rate_annotator2", ""),
            ])
        # Overall row
        ov = agreement["overall"]
        writer.writerow([
            "OVERALL", "", ov["n_items"],
            ov["mean_percent_agreement"], ov["mean_cohens_kappa"],
            ov["mean_krippendorffs_alpha"], ov["interpretation"], "", "",
        ])

        # Per-annotator statistics
        if annotator_stats:
            writer.writerow([])  # blank row separator
            writer.writerow(["annotator", "metric", "value"])
            for s in annotator_stats:
                name = s["name"]
                writer.writerow([name, "n_items", s["n_items"]])
                writer.writerow([name, "agreement_rate", s["agreement_rate"]])
                writer.writerow([name, "metadata_correct_rate", s["metadata_correct_rate"]])
                writer.writerow([name, "hallucination_rate", s["hallucination_rate"]])
                writer.writerow([name, "citation_correct_rate", s["citation_correct_rate"]])
                for p in ("attribution", "identification", "accessibility", "specificity"):
                    writer.writerow([name, f"force11_{p}", s["force11_scores"][p]])
                writer.writerow([name, "force11_overall", s["force11_overall"]])
    logger.info(f"Written: {path}")


# ──────────────────────────────────────────────────────────────────────
# Main Entry Point
# ──────────────────────────────────────────────────────────────────────

def agreement_analysis(annotator1_path: str, annotator2_path: str,
                       gold_path: str, output_dir: str,
                       name1: str = "Feng", name2: str = "Kalye"):
    """Run full agreement analysis and write results."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading annotator 1: {annotator1_path}")
    ann1 = load_annotations(annotator1_path)
    logger.info(f"  → {len(ann1)} completed annotations")

    logger.info(f"Loading annotator 2: {annotator2_path}")
    ann2 = load_annotations(annotator2_path)
    logger.info(f"  → {len(ann2)} completed annotations")

    # ── Per-Annotator Statistics ──
    logger.info("Computing per-annotator statistics...")
    stats1 = per_annotator_stats(ann1, name1)
    stats2 = per_annotator_stats(ann2, name2)
    annotator_stats = [stats1, stats2]

    print("\n" + "=" * 60)
    print("Per-Annotator Statistics")
    print("=" * 60)
    for s in annotator_stats:
        print(f"\n  [{s['name']}] ({s['n_items']} items)")
        print(f"    Agreement (correct):   {s['agreement_rate']:.1%}")
        print(f"    Metadata correct:      {s['metadata_correct_rate']:.1%}")
        print(f"    Hallucination rate:    {s['hallucination_rate']:.1%}")
        print(f"    Citation correct:      {s['citation_correct_rate']:.1%}")
        f11 = s['force11_scores']
        print(f"    FORCE11: attr={f11['attribution']:.3f} ident={f11['identification']:.3f} "
              f"access={f11['accessibility']:.3f} spec={f11['specificity']:.3f} "
              f"overall={s['force11_overall']:.3f}")

    # ── Agreement ──
    logger.info("Computing inter-annotator agreement...")
    agreement = compute_agreement(ann1, ann2)

    print("\n" + "=" * 60)
    print("Inter-Annotator Agreement")
    print("=" * 60)
    for dim, vals in agreement["dimensions"].items():
        print(f"  {dim:25s}  κ={vals['cohens_kappa']:+.3f}  "
              f"α={vals['krippendorffs_alpha']:+.3f}  "
              f"agree={vals['percent_agreement']:.1%}  "
              f"[{vals['interpretation']}]")
    ov = agreement["overall"]
    print("-" * 60)
    print(f"  {'OVERALL':25s}  κ={ov['mean_cohens_kappa']:+.3f}  "
          f"α={ov['mean_krippendorffs_alpha']:+.3f}  "
          f"agree={ov['mean_percent_agreement']:.1%}  "
          f"[{ov['interpretation']}]")

    # Disagreements
    logger.info("Computing disagreements...")
    items = compute_disagreements(ann1, ann2, name1, name2)
    n_disagree = sum(1 for i in items if i["n_disagreed"] > 0)
    print(f"\n  Items with any disagreement: {n_disagree}/{len(items)}")

    # Consensus
    logger.info("Computing consensus labels...")
    consensus = compute_consensus(ann1, ann2)

    # Sample representativeness
    sample_repr = {}
    if gold_path and Path(gold_path).exists():
        logger.info(f"Loading gold data: {gold_path}")
        gold_data, gold_sw_per_paper, gold_all_sw = load_gold(gold_path)
        sample_pmcids = set(k[0] for k in ann1.keys())
        sample_repr = compute_sample_representativeness(
            sample_pmcids, gold_data, gold_sw_per_paper, gold_all_sw
        )
        print(f"\n  Sample: {sample_repr['sample_size']}/{sample_repr['full_size']} "
              f"({sample_repr['sample_ratio']:.1%})")
        print(f"  KS test: D={sample_repr['ks_test']['statistic']:.4f}, "
              f"p={sample_repr['ks_test']['p_value']:.4f}")
        print(f"  Unique SW coverage: {sample_repr['unique_sw_coverage']:.1%}")
        print(f"  Top-20 SW overlap: {sample_repr['top20_software_overlap']:.0%}")

    # Write outputs
    write_summary(output_dir, agreement, sample_repr, consensus, name1, name2, annotator_stats)
    write_per_item(output_dir, items)
    write_disagreements_csv(output_dir, items, name1, name2)
    write_metrics_csv(output_dir, agreement, annotator_stats)

    print(f"\n✅ Results written to: {output_dir}/")
    print(f"   summary.json, metrics.csv, per_item_agreement.jsonl, disagreements.csv")


def main():
    parser = argparse.ArgumentParser(description="Inter-Annotator Agreement Analysis")
    parser.add_argument("--annotator1", required=True, help="Path to annotator 1 checkpoint JSON")
    parser.add_argument("--annotator2", required=True, help="Path to annotator 2 checkpoint JSON")
    parser.add_argument("--gold", default=None, help="Path to gold ground truth JSON (for representativeness)")
    parser.add_argument("--output_dir", default="annotation/agreement_results/",
                        help="Output directory")
    parser.add_argument("--name1", default="Feng", help="Annotator 1 name")
    parser.add_argument("--name2", default="Kalye", help="Annotator 2 name")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    agreement_analysis(
        annotator1_path=args.annotator1,
        annotator2_path=args.annotator2,
        gold_path=args.gold,
        output_dir=args.output_dir,
        name1=args.name1,
        name2=args.name2,
    )


if __name__ == "__main__":
    main()
