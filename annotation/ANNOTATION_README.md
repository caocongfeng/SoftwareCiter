# Annotation & Statistics — User Guide

## Overview

This project provides two independent analysis workflows for evaluating software citation quality:

| Workflow | Purpose | Key Scripts |
|----------|---------|-------------|
| **Annotation** | Human evaluation of citation quality via browser-based tool | `generate_annotation.py`, `annotation_server.py` |
| **Statistics** | Automated metric computation (pred vs gold) | `Statistics.py` |
| **Agreement** | Inter-annotator agreement analysis | `agreement_analysis.py` |

Both `evaluation/` and `Gold_Mention_Oracle/` pipelines share identical tooling.

---

## Prerequisites

- **Python 3.11+**
- **Annotation & Statistics scripts**: Python standard library only — no `pip install` needed
- **Visualization notebooks**: `matplotlib`, `numpy` (`pip install matplotlib numpy`)
- **Pipeline** (`run_oracle_pipeline.py`): requires `citation` conda environment

---

## Complete End-to-End Workflow

### Phase 1: Run Statistics (Automated Metrics)

Computes 4 levels of metrics: Paper Detection → Mention Extraction → Metadata Accuracy → Citation Quality.

```bash
# ── Step 1: Evaluation pipeline (30 papers) ──
python3.11 evaluation/Statistics.py \
  --gold evaluation/software_citation_ground_truth.json \
  --pred evaluation/evaluation_citations.json \
  --output_dir evaluation/statistics_results/

# ── Step 2: Oracle pipeline (1217 papers) ──
python3.11 Gold_Mention_Oracle/Statistics.py \
  --gold Gold_Mention_Oracle/software_citation_ground_truth.json \
  --pred Gold_Mention_Oracle/oracle_citations.json \
  --output_dir Gold_Mention_Oracle/statistics_results/

# ── Step 3: Visualize (open in Jupyter) ──
# evaluation/visualize_statistics.ipynb
# Gold_Mention_Oracle/visualize_statistics.ipynb
```

**Output** (`statistics_results/`):

| File | Description |
|------|-------------|
| `summary.json` | All aggregate metrics (P/R/F1, per-field accuracy, compliance) |
| `metrics.csv` | Flat table for LaTeX/Excel import |
| `per_paper_results.jsonl` | Per-paper: detection, mention counts, metadata |
| `per_software_results.jsonl` | Per-software: field-level matches, compliance, completeness |

---

### Phase 2: Generate Annotation Files (Human Evaluation)

Sample publications and generate browser-based annotation tools.

```bash
# ── Step 1: Sample 100 publications ──
python3.11 Gold_Mention_Oracle/generate_annotation.py --sample 100 --resample

# ── Step 2: Start annotation server (real-time save) ──
python3.11 annotation_server.py --dir Gold_Mention_Oracle --port 8765

# ── Step 3: Open browser ──
# → http://localhost:8765
```

For each software citation, annotate:
1. **Agreement** (0/1) — Does agent citation match expected?
2. **Metadata correct** (yes/no)
3. **Hallucination** (yes/no)
4. **Citation correct** (yes/no)
5. **FORCE11 principles** — Attribution, Identification, Accessibility, Specificity (0/1 each)
6. Click **Mark Done** → saves to disk automatically

---

### Phase 3: Inter-Annotator Agreement (After 2+ Annotators Complete)

Compute Cohen's κ, Krippendorff's α, and sample representativeness.

```bash
# ── Step 1: Run agreement analysis ──
python3.11 annotation/agreement_analysis.py \
  --annotator1 annotation/annotation_oracle_checkpoint_Feng.json \
  --annotator2 annotation/annotation_oracle_checkpoint_Kalye.json \
  --gold Gold_Mention_Oracle/software_citation_ground_truth.json \
  --output_dir annotation/agreement_results/

# ── Step 2: Visualize (open in Jupyter) ──
# annotation/visualize_agreement.ipynb
```

**Output** (`agreement_results/`):

| File | Description |
|------|-------------|
| `summary.json` | κ, α, % agreement per dimension, sample representativeness |
| `metrics.csv` | Flat table of all agreement metrics |
| `per_item_agreement.jsonl` | Per-item: agreed/disagreed dimensions |
| `disagreements.csv` | Human-readable table for adjudication |

---

## Sampling Config

The config file `annotation_config.json` controls sampling:

```json
{
  "exclusion": ["PMC123"],
  "inclusion": ["PMC456"],
  "sample": []
}
```

### Sampling Logic

| Condition | Behavior |
|-----------|----------|
| `sample` non-empty | Use exact list (reproducibility) |
| `inclusion ≥ sample_size` | Sample from inclusion set only |
| `inclusion < sample_size` | All inclusion + random fill from rest (excluding exclusion) |
| `--sample 0` | Return all (minus exclusion) |

```bash
# First run: sample and save to config
python3.11 Gold_Mention_Oracle/generate_annotation.py --sample 100 --resample

# Subsequent runs: reuse saved sample (reproducible)
python3.11 Gold_Mention_Oracle/generate_annotation.py --sample 100

# Change sample: clear and re-sample
python3.11 Gold_Mention_Oracle/generate_annotation.py --sample 50 --resample
```

> **Important**: After editing `exclusion`/`inclusion`, clear `sample` to `[]` and use `--resample`.

---

## Statistics — Metric Details

### Level 1: Paper-Level Detection

| Metric | Definition |
|--------|-----------|
| TP | Gold has software AND at least one pred name matches |
| FN | Gold has software BUT no pred name matches |
| FP | Gold has no software BUT pred has predictions |

### Level 2: Mention-Level Extraction

Two matching modes:
- **Exact**: lowercase string comparison
- **Normalized**: alias table + version stripping + containment check

### Level 3: Metadata-Level Accuracy

Fields compared: `version`, `authors`, `publisher`, `year`, `doi`, `url`, `license`

| Field | Normalization |
|-------|--------------|
| `version` | Strip "v"/"version", keep digits |
| `authors` | Lowercase, strip punctuation, sort tokens |
| `doi` | Strip `https://doi.org/`, lowercase |
| `url` | Strip `http(s)://`, trailing `/`, lowercase |
| `year` | Extract 4-digit year |
| `publisher` | Lowercase + strip |
| `license` | Lowercase, strip punctuation |

### Level 4: Citation-Level Quality

**FORCE11 Compliance** (0/1 each):

| Principle | Condition |
|-----------|-----------|
| Attribution | `authors` or `publisher` non-empty |
| Identification | `doi` non-empty |
| Accessibility | `doi` or `url` non-empty |
| Specificity | `version` non-empty |

**Completeness**: fields checked = name, version, authors, year, publisher, doi_or_url

### Enhanced Metrics

| Metric | Category | Description |
|--------|----------|-------------|
| **Macro P/R/F1** | Mention | Per-paper average (avoids large-paper bias) |
| **Jaccard Similarity** | Mention | Set overlap of normalized name sets |
| **Over-generation Ratio** | Mention | pred_count / gold_count per paper |
| **Field Recall/Precision** | Metadata | Per-field: recall, precision, F1 |
| **Hallucination Rate** | Metadata | Pred fills a field that gold doesn't have |
| **Weighted Score** | Metadata | Σ(weight × accuracy) with DOI/authors weighted higher |
| **ROUGE-L** | Citation | Word-level LCS similarity of citation text |
| **Error Taxonomy** | Overall | missed / hallucinated / partial_name / wrong_version |
| **Bootstrap 95% CI** | Overall | Confidence intervals via 1000-resample bootstrap |

---

## Agreement — Metric Details

### Metrics Computed

| Metric | Description |
|--------|-------------|
| **% Agreement** | Proportion of items where annotators agree |
| **Cohen's κ** | Chance-corrected agreement for 2 raters |
| **Krippendorff's α** | Robust chance-corrected agreement |

### Interpretation Scale (Landis & Koch, 1977)

| κ | Interpretation |
|---|----------------|
| < 0.00 | Poor |
| 0.00–0.20 | Slight |
| 0.21–0.40 | Fair |
| 0.41–0.60 | Moderate |
| 0.61–0.80 | Substantial |
| 0.81–1.00 | Almost Perfect |

### Sample Representativeness

- **KS test**: Compares software-per-paper distribution (sample vs full dataset)
- **Unique SW coverage**: % of unique software names covered by sample
- **Top-20 overlap**: Overlap of most frequent software between sample and full dataset

---

## CLI Reference

### Statistics.py

```bash
python3.11 {evaluation,Gold_Mention_Oracle}/Statistics.py \
  --gold <gold.json> --pred <pred.json> --output_dir <dir>
```

| Option | Default | Description |
|--------|---------|-------------|
| `--gold` | required | Path to ground truth JSON |
| `--pred` | required | Path to prediction JSON |
| `--output_dir` | `statistics_results/` | Output directory |

### agreement_analysis.py

```bash
python3.11 annotation/agreement_analysis.py \
  --annotator1 <file1.json> --annotator2 <file2.json> \
  --gold <gold.json> --output_dir <dir>
```

| Option | Default | Description |
|--------|---------|-------------|
| `--annotator1` | required | Annotator 1 checkpoint |
| `--annotator2` | required | Annotator 2 checkpoint |
| `--gold` | optional | Gold data (for representativeness) |
| `--output_dir` | `annotation/agreement_results/` | Output directory |
| `--name1/--name2` | Feng/Kalye | Annotator names |

### generate_annotation.py

```bash
python3.11 {evaluation,Gold_Mention_Oracle}/generate_annotation.py \
  [--sample N] [--resample] [--config <path>]
```

| Option | Default | Description |
|--------|---------|-------------|
| `--sample N` | 100 | Number of publications (0=all) |
| `--resample` | off | Force new sample |
| `--config` | `annotation_config.json` | Config file path |

### annotation_server.py

```bash
python3.11 annotation_server.py --dir <DIR> --port <PORT>
```

| Option | Default | Description |
|--------|---------|-------------|
| `--dir` | required | `Gold_Mention_Oracle` or `evaluation` |
| `--port` | 8765 | HTTP port |

---

## File Structure

```
software_citation_agent/
├── annotation_server.py                     # Annotation HTTP server (shared)
├── ANNOTATION_README.md                     # This file
│
├── annotation/                              # Inter-annotator agreement
│   ├── agreement_analysis.py               # Agreement computation script
│   ├── visualize_agreement.ipynb           # Agreement visualization
│   ├── annotation_oracle_checkpoint_Feng.json   # Annotator 1 data
│   ├── annotation_oracle_checkpoint_Kalye.json  # Annotator 2 data
│   └── agreement_results/                  # Output: summary, metrics, disagreements
│
├── evaluation/                              # Full pipeline evaluation
│   ├── Statistics.py                       # 4-level statistics (core implementation)
│   ├── visualize_statistics.ipynb          # Statistics visualization
│   ├── generate_annotation.py             # Annotation HTML generator
│   ├── software_citation_ground_truth.json # Gold data
│   ├── evaluation_citations.json          # Agent predictions
│   ├── statistics_results/                # Output: summary, metrics, per-paper/sw
│   ├── annotation_config.json             # Sampling config
│   └── SoMeSci.json                       # SoMeSci evidence data
│
└── Gold_Mention_Oracle/                     # Oracle (gold names → downstream) evaluation
    ├── Statistics.py                        # Statistics wrapper (imports from evaluation/)
    ├── visualize_statistics.ipynb           # Statistics visualization
    ├── generate_annotation.py              # Annotation HTML generator (wrapper)
    ├── software_citation_ground_truth.json  # Gold data
    ├── oracle_citations.json               # Oracle predictions
    ├── statistics_results/                  # Output: summary, metrics, per-paper/sw
    ├── annotation_config.json              # Sampling config
    └── SoMeSci.json                        # SoMeSci evidence data
```
