# Software Citation Agent

An AI-powered agent that reads academic papers, extracts software mentions, enriches metadata via web search, and produces **[FORCE11 Software Citation Principles](https://doi.org/10.7717/peerj-cs.86)**-compliant citations.

---

## ✨ Features

- 📄 **Multi-format Parser** — Reads XML (JATS/NLM), plain text; handles malformed XML robustly
- 🔍 **LLM-powered Extraction** — GPT-based software mention identification with context-aware filtering
- 🌐 **Web Search Enrichment** — DuckDuckGo search with domain disambiguation and 3× retry
- 📝 **FORCE11-compliant Citations** — Formatted citations following all 6 FORCE11 principles
- ✅ **Multi-stage Verification** — 9 automated checks + LLM consistency + targeted web re-search
- 📊 **4-Level Evaluation** — Paper detection, mention extraction, metadata accuracy, citation quality
- 🏷️ **Annotation Tools** — Browser-based EN/ZH annotation tools with real-time server save
- 📈 **Inter-Annotator Agreement** — Cohen's κ, Krippendorff's α, sample representativeness
- ⚡ **Parallel Execution** — Two-level concurrency (papers × mentions) with rate limiting
- 💾 **Checkpoint/Resume** — All pipelines support crash recovery

---

## 🏗️ Architecture

[softwareCiter-Page-2.png](./softwareCiter-Page-2.png)

### Module Overview

| Module | File | Description |
|--------|------|-------------|
| **Models** | `src/models.py` | Pydantic data models: `SoftwareMention`, `SoftwareMetadata`, `SoftwareCitation` |
| **Config** | `src/config.py` | Centralized configuration; per-module model assignments via `.env` |
| **Parser** | `src/parser.py` | XML tag stripping → heuristic or LLM section splitting |
| **Extractor** | `src/extractor.py` | LLM software extraction; chunking + dedup for long papers |
| **Searcher** | `src/searcher.py` | Multi-query DuckDuckGo → LLM metadata synthesis |
| **Citation Builder** | `src/citation_builder.py` | LLM-formatted FORCE11 citations |
| **Verifier** | `src/verifier.py` | 9 rule-based checks + LLM consistency + web re-search |
| **LLM Logger** | `src/llm_logger.py` | Thread-safe JSONL logging of all LLM calls |
| **Agent** | `src/agent.py` | Orchestrates the full 5-step pipeline |
| **CLI** | `src/main.py` | Command-line entry point |

---

## 🚀 Quick Start

### 1. Create Environment

```bash
conda create -n citation python=3.11 -y
conda activate citation
```

### 2. Install

```bash
pip install -e .
```

### 3. Configure

```bash
echo "OPENAI_API_KEY=sk-your-api-key-here" > .env
```

### 4. Run

```bash
# Process a single paper
python -m src.main --input paper.xml

# JSON output
python -m src.main --input paper.xml --output citations.json --format json
```

---

## 📊 Evaluation Pipelines

Three evaluation modes measure agent performance against ground truth (SoMeSci dataset, 1217 papers):

| Pipeline | Directory | Purpose | Approach |
|----------|-----------|---------|----------|
| **Full Evaluation** | `evaluation/` | End-to-end agent test | LLM + Search + Verification (5 steps) |
| **Gold Oracle** | `Gold_Mention_Oracle/` | Downstream ceiling test | Ground truth names → Search + Citation |
| **GPT-Only Baseline** | `baseline/` | LLM-only baseline | Single GPT call, no tools |

### Run Evaluation

```bash
# Full pipeline (processes papers through entire agent)
python3.11 evaluation/run_evaluation.py --resume --workers 4

# Oracle pipeline (uses gold software names, tests search→citation→verify only)
python3.11 Gold_Mention_Oracle/run_oracle_pipeline.py --resume --workers 4
```

### Gold Mention Oracle Pipeline

The Oracle pipeline **bypasses extraction** and feeds gold-standard software mentions directly into the downstream modules (search → citation → verify). This measures the **ceiling performance** assuming perfect software extraction.

**Input file**: `Gold_Mention_Oracle/enhanced_software_citation_ground_truth.json`

```json
{
  "pmcid": "PMC1274293",
  "article_id": "http://data.gesis.org/somesci/PMC1274293",
  "software_citations": [
    {
      "software_name": "SNPdetector",
      "original_data": {
        "version": "",
        "developer": "['Zhang J', 'Wheeler DA', ...]",
        "url": "http://lpg.nci.nih.gov",
        "citation_evidence": "We developed SNPdetector for ... <SEP> ..."
      },
      "enriched_metadata": { "name": "...", "version": "...", ... }
    }
  ]
}
```

**Processing flow** (skips Parse & Extract steps):

```
Gold Truth (original_data per software)
  │
  ├── 1. gold_to_mention()  → Convert original_data to SoftwareMention
  │     (name, version, publisher/developer, url, citation_evidence)
  │
  ├── 2. search_software_metadata()  → DuckDuckGo web search + LLM synthesis
  │
  ├── 3. build_citation()  → LLM generates FORCE11 citation text
  │
  └── 4. verify_citation()  → LLM verifies & corrects citation
         │
         └── Output: oracle_citations.json
```

**Key design details:**
- Uses `original_data` (raw paper info), NOT `enriched_metadata`, so the downstream pipeline enriches from scratch
- `developer` field may be a Python list string (e.g., `"['Name1', 'Name2']"`), handled by `ast.literal_eval`
- `citation_evidence` uses `<SEP>` to concatenate multiple evidence sentences
- Parallel: `--workers 4` (article-level) + `--sw-workers 3` (software-level per article)
- DuckDuckGo searches rate-limited via semaphore (max 3 concurrent)

### GPT-Only Baseline

The baseline sends a single GPT API call per paper — no web search, no verification, no multi-step pipeline. This isolates the value of the agent architecture by comparing against LLM parametric knowledge alone.

```bash
# Run baseline (same model as agent: gpt-5-mini)
python3.11 baseline/run_baseline.py --resume --workers 4

# Run on subset first
python3.11 baseline/run_baseline.py --limit 10

# Compute baseline statistics
python3.11 baseline/Statistics.py

# Visualize results
cd baseline/
jupyter notebook visualize_statistics.ipynb
# → Kernel → Restart & Run All
```

### Compute Statistics

8-level analysis: Paper Detection → Mention Extraction (micro/macro) → Metadata Accuracy → Citation Quality → Coverage Analysis → Error Taxonomy → ROUGE-L Similarity → Bootstrap 95% CI.

```bash
# Evaluation statistics
python3.11 evaluation/Statistics.py \
  --gold evaluation/software_citation_ground_truth.json \
  --pred evaluation/evaluation_citations.json \
  --output_dir evaluation/statistics_results/

# Oracle statistics
python3.11 Gold_Mention_Oracle/Statistics.py
```

**Computed metrics:**

| Level | Metrics |
|-------|---------|
| **Paper Detection** | Precision, Recall, F1 |
| **Mention Extraction** | Micro & Macro P/R/F1 (exact + normalized), Jaccard similarity, Over-generation ratio |
| **Metadata Accuracy** | Per-field exact/normalized accuracy, field-level recall/precision/hallucination, weighted score |
| **Citation Quality** | FORCE11 compliance (4 principles), field completeness, ROUGE-L text similarity |
| **Coverage Analysis** | Paper coverage, mention coverage, per-field metadata coverage, overall coverage |
| **Error Taxonomy** | Missed software, hallucinated software, partial name matches, wrong versions |
| **Confidence Intervals** | Bootstrap 95% CI for mention F1, citation completeness |

**Output files:** `summary.json`, `metrics.csv`, `per_paper_results.jsonl`, `per_software_results.jsonl`, `tp.json`, `fp.json`, `fn.json`

### Visualize Statistics

Publication-quality figures (300 DPI PNG + vector PDF) for all computed metrics.

```bash
# Evaluation figures
cd evaluation/
jupyter notebook visualize_statistics.ipynb
# Run all cells (Kernel → Restart & Run All)

# Oracle figures
cd Gold_Mention_Oracle/
jupyter notebook visualize_statistics.ipynb
```

The notebook generates **7 publication-ready figures** saved to `statistics_results/figures/`:

| Figure | File | Content |
|--------|------|---------|
| Metrics Table | `table_paper_mention.{png,pdf}` | Paper & mention detection P/R/F1 (micro + macro) |
| Metadata Radar | `radar_metadata.{png,pdf}` | Per-field exact vs normalized accuracy |
| Field Detail | `metadata_field_detail.{png,pdf}` | Recall, precision, hallucination per metadata field |
| Citation Quality | `citation_quality.{png,pdf}` | FORCE11 compliance + field completeness (dual radar) |
| Coverage Analysis | `coverage_analysis.{png,pdf}` | Agent coverage of gold standard (paper, mention, fields) |
| Error Taxonomy | `error_taxonomy.{png,pdf}` | Distribution of error types |
| Confidence Intervals | `confidence_intervals.{png,pdf}` | Forest plot with 95% bootstrap CI |

### Step-Level Extraction Analysis

The evaluation pipeline generates detailed LLM interaction logs (`eval_LLM_log.jsonl`) capturing intermediate results at each step. This analysis extracts those results and compares the **extraction step** against the SoMeSci gold standard (human-annotated).

**Step 1: Extract intermediate results and export unified files**

```bash
# Extract all step results + export all unified files
python3.11 evaluation/extract_step_results.py --export

# Or export specific steps:
python3.11 evaluation/extract_step_results.py --export --step gold        # SoMeSci → gold_extraction.json
python3.11 evaluation/extract_step_results.py --export --step extraction  # Agent → agent_extraction.json
python3.11 evaluation/extract_step_results.py --export --step search      # Agent search → agent_search.json
python3.11 evaluation/extract_step_results.py --export --step citation    # Agent citation → agent_citation.json
```

All exported files use the **same unified format** (per-PMCID, per-mention):

```json
{
  "PMC6261107": {
    "publication_id": "PMC6261107",
    "mentions": [
      {"name": "SPSS", "version": "21", "publisher": "IBM", "url": "", "context": "..."}
    ]
  }
}
```

Output files:
- `evaluation_step_results.json` — full intermediate results (all steps, raw format)
- `gold_extraction.json` — SoMeSci gold standard (unified format)
- `agent_extraction.json` — agent extraction step (unified format)
- `agent_search.json` — agent search enrichment (unified format)
- `agent_citation.json` — agent citation output (unified format)

**Step 2: Compare extraction — Human (SoMeSci) vs Agent (LLM)**

```bash
python3.11 evaluation/extraction_analysis.py
```

Compares `gold_extraction.json` vs `agent_extraction.json`. Output per-publication for human verification:

```json
{
  "publication_id": "PMC6261107",
  "n_gold": 1, "n_pred": 1, "n_tp": 1, "n_fp": 0, "n_fn": 0,
  "precision": 1.0, "recall": 1.0,
  "matches": [{
    "gold": {"name": "SPSS", "version": "21", "publisher": "", ...},
    "pred": {"name": "SPSS", "version": "21", "publisher": "IBM", ...},
    "match_type": "exact", "version_correct": true
  }],
  "missed": [],
  "hallucinated": []
}
```

Computed metrics (same framework as `Statistics.py`):

| Level | Metrics |
|-------|---------|
| **Paper Detection** | P/R/F1 (paper has ≥1 software) |
| **Mention (Micro)** | Exact P/R/F1, Normalized P/R/F1 |
| **Mention (Macro)** | Per-paper averaged P/R/F1 ± std |
| **Jaccard** | Mean/median/std of per-paper normalized Jaccard |
| **Over-Generation** | Mean/median pred/gold ratio |
| **Field Analysis** | Per-field recall/precision/hallucination (version, publisher, url) |
| **Bootstrap CI** | 95% CI for normalized P/R/F1 and Jaccard |

Output files:
- `extraction_comparison_results.json` — per-publication detail
- `extraction_comparison_summary.json` — aggregate metrics

**Step 3: Visualize**

```bash
cd evaluation/
jupyter notebook visualize_extraction.ipynb
# → Kernel → Restart & Run All
```

Generates **8 publication-ready figures** in `extraction_analysis_figures/`:

| Figure | File | Content |
|--------|------|---------|
| Metrics Table | `extraction_metrics_table.{png,pdf}` | Paper + mention detection (6 rows) |
| Recall Distribution | `recall_distribution.{png,pdf}` | Per-paper recall histogram |
| Precision Distribution | `precision_distribution.{png,pdf}` | Per-paper precision histogram |
| Match Types | `match_types.{png,pdf}` | Exact/normalized/missed/hallucinated counts |
| Field Analysis | `field_analysis.{png,pdf}` | Recall/precision/hallucination per field |
| Over-Generation | `over_generation.{png,pdf}` | pred/gold ratio distribution |
| P vs R Scatter | `precision_vs_recall.{png,pdf}` | Per-paper scatter with F1 iso-lines |
| Confidence Intervals | `confidence_intervals.{png,pdf}` | Forest plot with 95% bootstrap CI |

### Per-Step Pipeline Statistics

Analyzes each pipeline step's contribution to the final citation quality:

```bash
python3.11 evaluation/step_statistics.py
```

| Step | What it measures | Key metric |
|------|-----------------|------------|
| **1. Extraction** | Gold (SoMeSci) vs agent name detection | Micro/Macro P/R/F1 |
| **2. Search** | Metadata field fill rate: extraction → search | Enrichment gain per field |
| **3. Citation Build** | FORCE11 compliance + completeness (pre-verify) | Compliance overall |
| **4. Verification** | FORCE11 compliance + completeness (post-verify) | Delta improvement |

**Visualize:**

```bash
cd evaluation/
jupyter notebook visualize_step_statistics.ipynb
# → Kernel → Restart & Run All
```

Generates **7 figures** in `step_statistics_figures/`:

| Figure | File | Content |
|--------|------|---------|
| Pipeline Overview | `pipeline_overview.{png,pdf}` | Step-by-step metrics table |
| Extraction P/R/F1 | `extraction_prf1.{png,pdf}` | Exact/normalized, micro/macro |
| Search Enrichment | `search_enrichment.{png,pdf}` | Before vs after field fill rate |
| FORCE11 Pre/Post | `force11_pre_post.{png,pdf}` | Compliance before/after verify |
| Completeness | `completeness_pre_post.{png,pdf}` | Field coverage before/after |
| Verify Waterfall | `verify_waterfall.{png,pdf}` | Verification step delta |
| Pipeline Flow | `pipeline_flow.{png,pdf}` | End-to-end metric progression |

### Human Annotation

Browser-based annotation tools for manual quality review.

```bash
# Generate annotation files (sample 100 publications)
python3.11 Gold_Mention_Oracle/generate_annotation.py --sample 100 --resample

# Start annotation server (real-time save to disk)
python3.11 annotation_server.py --dir Gold_Mention_Oracle --port 8765
# → Open http://localhost:8765
```

### Inter-Annotator Agreement

Complete workflow for inter-annotator agreement analysis and publication-quality visualization.

#### Step 1 — Compute Agreement Statistics

```bash
python3.11 annotation/agreement_analysis.py \
  --annotator1 annotation/annotation_oracle_checkpoint_Feng.json \
  --annotator2 annotation/annotation_oracle_checkpoint_Kalye.json \
  --gold Gold_Mention_Oracle/software_citation_ground_truth.json \
  --output_dir annotation/agreement_results/
```

Output: `annotation/agreement_results/summary.json` — per-annotator quality metrics (agreement rate, metadata correctness, hallucination rate, citation correctness, FORCE11 principle scores) and inter-annotator agreement (Cohen's κ, Krippendorff's α, confusion matrices, interpretation).

#### Step 2 — Generate Publication-Quality Figures

```bash
cd annotation/
jupyter notebook visualize_agreement.ipynb
# Run all cells (Kernel → Restart & Run All)
```

The notebook produces **4 publication-ready figures** (300 DPI PNG + vector PDF) saved to `annotation/agreement_results/figures/`:

| Figure | File | Content |
|--------|------|---------|
| Per-Annotator Quality | `annotator_quality.{png,pdf}` | Agreement, metadata correctness, citation correctness per annotator |
| Hallucination Rate | `hallucination_rate.{png,pdf}` | Side-by-side hallucination rate comparison |
| FORCE11 Scores | `force11_scores.{png,pdf}` | Attribution, Identification, Accessibility, Specificity per annotator |
| Agreement Table | `agreement_table.{png,pdf}` | Cohen's κ, Krippendorff's α, % agreement for all 8 dimensions |

> **Note**: The notebook uses `serif` fonts, 300 DPI, and removes top/right spines for clean academic aesthetics. Both PNG (for preview) and PDF (for LaTeX `\includegraphics`) are generated.

See [`ANNOTATION_README.md`](ANNOTATION_README.md) for full documentation on annotation, statistics, and agreement workflows.

---

## 📁 Project Structure

```
software_citation_agent/
├── src/                               # Core pipeline modules
│   ├── main.py                        # CLI entry point
│   ├── agent.py                       # Pipeline orchestrator
│   ├── config.py                      # Configuration
│   ├── models.py                      # Pydantic data models
│   ├── parser.py                      # Document parser (XML/TXT)
│   ├── extractor.py                   # LLM software extraction
│   ├── searcher.py                    # Web search & metadata enrichment
│   ├── citation_builder.py            # FORCE11 citation formatting
│   ├── verifier.py                    # Citation verification & correction
│   └── llm_logger.py                 # LLM interaction logging
│
├── evaluation/                        # Full pipeline evaluation
│   ├── run_evaluation.py              # Main pipeline (parse→extract→search→cite→verify)
│   ├── evaluate_citations.py          # Legacy metrics (name matching + field accuracy)
│   ├── Statistics.py                  # 8-level statistics + coverage + error taxonomy
│   ├── extract_step_results.py        # Extract per-step results from LLM logs
│   ├── extraction_analysis.py         # Extraction vs gold standard comparison
│   ├── generate_annotation.py         # Annotation HTML generator
│   ├── check_progress.py              # Real-time progress monitor
│   ├── visualize_statistics.ipynb     # Publication-quality statistics visualization
│   ├── visualize_extraction.ipynb     # Extraction-level analysis visualization
│   ├── software_citation_ground_truth.json  # Gold standard (1217 pubs)
│   ├── evaluation_citations.json      # Agent predictions
│   ├── eval_LLM_log.jsonl             # LLM interaction log (all pipeline steps)
│   ├── statistics_results/            # Statistics output
│   │   ├── summary.json               #   All metrics (JSON)
│   │   ├── metrics.csv                #   Flat metrics (CSV)
│   │   ├── tp.json / fp.json / fn.json #  TP/FP/FN split detail
│   │   └── figures/                   #   Publication-quality figures (PNG + PDF)
│   ├── gold_extraction.json           # SoMeSci gold standard (unified format)
│   ├── agent_extraction.json          # Agent extraction (unified format)
│   ├── agent_search.json              # Agent search enrichment (unified format)
│   ├── agent_citation.json            # Agent citation text (unified format)
│   ├── extraction_comparison_results.json  # Per-pub extraction comparison
│   ├── extraction_comparison_summary.json  # Aggregate comparison metrics
│   └── extraction_analysis_figures/   # Extraction analysis figures (PNG + PDF)
│
├── Gold_Mention_Oracle/               # Oracle pipeline (GT names → downstream)
│   ├── run_oracle_pipeline.py         # Oracle pipeline
│   ├── evaluate_oracle.py             # Oracle evaluation
│   ├── Statistics.py                  # Statistics wrapper (imports evaluation/)
│   ├── generate_annotation.py         # Annotation generator wrapper
│   ├── check_oracle_progress.py       # Progress monitor
│   ├── visualize_statistics.ipynb     # Publication-quality statistics visualization
│   ├── oracle_citations.json          # Oracle predictions (1217 pubs)
│   └── statistics_results/            # Statistics output
│       └── figures/                   #   Publication-quality figures (PNG + PDF)
│
├── annotation/                        # Human annotation & agreement
│   ├── agreement_analysis.py          # Inter-annotator agreement computation
│   ├── visualize_agreement.ipynb      # Publication-quality agreement visualization
│   ├── annotation_oracle_checkpoint_Feng.json   # Annotator 1 checkpoint
│   ├── annotation_oracle_checkpoint_Kalye.json  # Annotator 2 checkpoint
│   └── agreement_results/             # Agreement output
│       ├── summary.json               #   Statistics & agreement metrics
│       └── figures/                   #   Publication-quality figures (PNG + PDF)
│
├── baseline/                          # GPT-only baseline (no agent tools)
│   ├── run_baseline.py                # Single-call GPT baseline runner
│   ├── Statistics.py                  # Statistics wrapper (imports evaluation/)
│   ├── visualize_statistics.ipynb     # Publication-quality statistics visualization
│   ├── parser_cache/                  # Pre-parsed papers (1217 JSON files)
│   ├── software_citation_ground_truth.json  # Gold standard
│   ├── baseline_citations.json        # Baseline predictions (generated)
│   └── statistics_results/            # Statistics output
│       └── figures/                   #   Publication-quality figures (PNG + PDF)
│
├── tests/                             # Unit & integration tests
├── annotation_server.py               # HTTP server for annotation
├── ANNOTATION_README.md               # Annotation & statistics docs
├── pyproject.toml                     # Dependencies
├── .env                               # API keys (not committed)
└── README.md                          # This file
```

---

## 🔧 Configuration

```bash
# .env file
OPENAI_API_KEY=sk-your-api-key-here

# Per-module model assignments (defaults)
MODEL_EXTRACTOR=gpt-5-mini
MODEL_SEARCHER=gpt-5-mini
MODEL_CITATION=gpt-5-mini
MODEL_VERIFIER=gpt-5-mini
LLM_TEMPERATURE=0
SEARCH_MAX_RESULTS=10
```

---

## ✅ Testing

```bash
python -m pytest tests/ -v
```

---

## 📦 Requirements

- **Python** ≥ 3.11
- **OpenAI API key**
- **Internet** for DuckDuckGo search

| Package | Purpose |
|---------|---------|
| `langchain` + `langchain-openai` | LLM framework |
| `duckduckgo-search` | Web search |
| `lxml` | XML parsing |
| `pydantic` | Data validation |
| `rich` | Terminal UI |

---

## 📄 License

This project is developed for academic research purposes.
