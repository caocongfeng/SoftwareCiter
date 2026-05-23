# Software Citation Benchmark Annotation Guidelines (English)

## 1. Task Goal

This annotation task is designed to assess the quality of automatically generated software citations in a software citation benchmark.

For each sample, the system already provides:
- a software entity,
- an `enriched_metadata` object,
- and a final `citation_text`.

Annotators must evaluate:

1. whether `enriched_metadata` is factually correct,
2. whether `citation_text` is correct and well-formed,
3. and to what extent the citation satisfies software citation principles.

**What is NOT evaluated:**
- publication → software extraction,
- whether `original_data` was extracted correctly from the publication,
- whether the publication mention itself is correct.

In other words, this task starts from an already identified software entity and evaluates:

> **From structured metadata to valid software citation**

---

## 2. Annotation Basis: Software Citation Principles

This task is guided by the FORCE11 Software Citation Principles, which state that software citations should support proper credit, unique identification, persistence, accessibility, and specificity. In particular:
- software should be treated as a legitimate citable research output,
- citations should provide proper credit and attribution,
- citations should include machine-actionable unique identifiers when possible,
- citations should facilitate access,
- and citations should be as specific as possible to the version used.

For annotation, we operationalize these principles into two dimensions:

### A. Factual Correctness
Whether the metadata and citation are factually correct.

### B. Principle Compliance
Whether the citation satisfies these four operational dimensions:
1. **Attribution**: author(s) or organization provided
2. **Identification**: DOI or other stable identifier provided
3. **Accessibility**: DOI or URL allows access/lookup
4. **Specificity**: version information provided

---

## 3. Input Structure

Each sample typically contains:
- `pmcid`
- `software_name`
- `citation_text`
- `enriched_metadata`
  - `name`
  - `version`
  - `authors`
  - `year`
  - `publisher`
  - `doi`
  - `url`
  - `license`
  - `description`
- `original_data`
- `verification_issues`
- `verification_status`

### Notes
- `software_name`: current target software entity
- `enriched_metadata`: structured metadata to be verified
- `citation_text`: final software citation to be verified
- `original_data`: publication-derived gold information; helpful as context, but **not itself the annotation target**
- `verification_issues` / `verification_status`: system-generated hints; use only as reference, not as annotation answers

---

## 4. Overall Annotation Workflow

### Step 1: Verify software identity
First verify whether the current sample clearly refers to the intended software entity.

Check:
- whether `software_name` and `enriched_metadata.name` refer to the same software,
- whether the software name in `citation_text` is consistent,
- whether a command, module, plugin, function, or descriptive phrase has been mistakenly treated as standalone software.

Even if the entity itself appears wrong, continue with field-level annotation.

### Step 2: Evaluate enriched_metadata field by field
You must annotate the following fields:

- `name`
- `version`
- `authors`
- `year`
- `publisher`
- `doi`
- `url`

For each field, choose exactly one label:
- **correct**
- **incorrect**
- **missing**

Detailed rules are below.

---

## 5. Field-level Annotation Rules

### 5.1 `name`
**Labels**: correct / incorrect / missing

**Goal**: Determine whether `enriched_metadata.name` is the correct software name.

**Mark as `correct` if:**
- only casing differs: `PyTorch` vs `pytorch`
- minor spacing differs: `R Studio` vs `RStudio`
- common naming variants still clearly refer to the same software: `GraphPad Prism` vs `Prism` (only if context is unambiguous)

**Mark as `incorrect` if:**
- it refers to a different software
- a command/function/plugin/submodule is mistaken for software
- the name is truncated or expanded in a way that changes identity

**Mark as `missing` if:**
- empty
- placeholder such as `unknown` or `n/a`

### 5.2 `version`
**Labels**: correct / incorrect / missing

**Goal**: Determine whether the version is correct for the target software.

**Mark as `correct` if:** the following are considered equivalent:
- `1.2.0` vs `v1.2.0`
- `6` vs `Version 6`
- `2012` vs `v2012`

**Mark as `incorrect` if:**
- clearly the wrong version
- belongs to another software
- hallucinated version

**Mark as `missing` if:**
- empty
- `unknown`
- no usable version given

**Important**: If no trustworthy source provides a version and the field is empty, mark `missing`, not `correct`.

### 5.3 `authors`
**Labels**: correct / incorrect / missing

**Goal**: Determine whether the author / developer information is correct.
This may be:
- an individual author,
- multiple authors,
- an organization,
- a development team.

**Mark as `correct` if:**
- minor organizational variants: `IBM` vs `IBM Corp.`
- author order differs but the set is effectively the same
- organizational authorship is acceptable for that software

**Mark as `incorrect` if:**
- completely wrong authors
- paper authors are mistakenly used as software authors
- unrelated organizations are listed
- hallucinated authors are introduced

**Mark as `missing` if:**
- empty
- `unknown`
- placeholder only

### 5.4 `year`
**Labels**: correct / incorrect / missing

**Goal**: Determine whether the year is a plausible and correct software-related year (e.g., release year, archived record year, or citation source year).

**Mark as `correct` if:**
- matches a trustworthy source
- minor formatting differences are acceptable, if policy allows

**Mark as `incorrect` if:**
- clearly wrong year
- future year
- strongly inconsistent with the version/source

**Mark as `missing` if:**
- empty
- `n.d.`
- `unknown`

### 5.5 `publisher`
**Labels**: correct / incorrect / missing

**Goal**: Determine whether the publisher / repository / organization is appropriate.

**Mark as `correct` if:**
- organization name is a minor variant: `GraphPad` vs `GraphPad Software`
- official host/platform is used appropriately: Zenodo, GitHub releases, institutional software page, etc.

**Mark as `incorrect` if:**
- unrelated organization or unrelated website
- journal venue is incorrectly used as the software publisher when the target should be the software itself
- hallucinated publisher

**Mark as `missing` if:**
- empty
- `unknown`

### 5.6 `doi`
**Labels**: correct / incorrect / missing

**Goal**: Determine whether the DOI exists and refers to the correct software object or version-specific archived record.

**Mark as `correct` if:**
- DOI format is valid
- DOI resolves to the correct software/software version landing page

**Mark as `incorrect` if:**
- DOI format is invalid
- DOI belongs to another software
- DOI refers only to a software paper when the target should be the software artifact
- hallucinated DOI

**Mark as `missing` if:**
- empty
- placeholder
- no DOI provided

### 5.7 `url`
**Labels**: correct / incorrect / missing

**Goal**: Determine whether the URL points to a reasonable location for identifying or accessing the software.

**Mark as `correct` if:**
- official homepage
- official download page
- official GitHub / GitLab repository
- Zenodo / Figshare / institutional repository landing page

**Mark as `incorrect` if:**
- unrelated page
- wrong software page
- only a paper/news page with no software access or identification value
- clearly irrelevant or misleading URL

**Mark as `missing` if:**
- empty
- no URL provided

---

## 6. Metadata-level Judgments

### 6.1 `metadata_correct`
Choose:
- **yes**
- **no**

**Recommended rule for `yes`:**
- `name` is correct,
- there is no obvious hallucination,
- the core identity fields are trustworthy,
- some fields may still be missing.

**Use `no` if:**
- the software entity itself is wrong,
- a key field is clearly wrong and undermines trust,
- metadata contains hallucinated information.

### 6.2 `hallucination`
Choose:
- **yes**
- **no**

Mark `yes` if the metadata contains clearly fabricated information, such as:
- non-existent DOI,
- unsupported author,
- invented version,
- unsupported publisher.

---

## 7. Citation Annotation Rules

### 7.1 What to check in `citation_text`
Evaluate whether:
1. it is consistent with the metadata,
2. the citation format is basically valid,
3. it contains incorrect or hallucinated information.

### 7.2 `citation_correct`
Choose:
- **yes**
- **no**

**Mark as `yes` if:**
- the citation is consistent with metadata,
- the structure is basically reasonable,
- it may be incomplete, but not factually wrong.

**Mark as `no` if:**
- it conflicts with metadata,
- cites the wrong object,
- uses an obviously unreasonable citation form,
- contains hallucinated or clearly incorrect facts.

---

## 8. Principle Compliance Scoring

For each sample, assign a binary score (0/1) to the following:

1. **Attribution**: author(s) or organization provided
2. **Identification**: DOI or other stable identifier provided
3. **Accessibility**: DOI or URL allows access/lookup
4. **Specificity**: version information provided

---

## 9. Citation Completeness Scoring

Check whether the following fields are present in the citation / metadata:

- software name
- version
- author / creator
- year
- publisher / repository
- DOI or URL

For each field:
- 1 = present
- 0 = absent

**completeness_score** = number of present fields / 6

---

## 10. Error Types (multi-select)

- `wrong_name`
- `wrong_version`
- `wrong_author`
- `wrong_year`
- `wrong_publisher`
- `wrong_doi`
- `wrong_url`
- `missing_version`
- `missing_year`
- `missing_doi`
- `missing_url`
- `hallucinated_info`
- `entity_mismatch`
- `software_paper_instead_of_software`
- `incomplete_metadata`
- `citation_metadata_inconsistent`

---

## 11. Recommended Output Fields

Each annotation result should include:

- `metadata_eval`
  - name / version / authors / year / publisher / doi / url
- `metadata_correct`
- `hallucination`
- `citation_correct`
- `principle_score`
  - attribution / identification / accessibility / specificity
- `completeness_score`
- `error_type`
- `annotator_note`

---

## 12. Examples

### Example 1: SPSS
- name: correct
- version: correct
- authors: correct
- year: missing
- publisher: correct
- doi: missing
- url: correct

Recommended:
- metadata_correct = yes
- citation_correct = yes
- principle_score = {attribution:1, identification:0, accessibility:1, specificity:1}
- error_type = [missing_year, missing_doi]

### Example 2: Prism
- name: correct
- version: correct
- authors: correct
- year: missing
- publisher: correct
- doi: missing
- url: missing

Recommended:
- metadata_correct = yes
- citation_correct = yes
- principle_score = {attribution:1, identification:0, accessibility:0, specificity:1}
- error_type = [missing_year, missing_doi, missing_url]

---

## 13. Final Principles for Annotators

Please always follow this priority order:

1. **Check factual correctness first**
2. **Then assess principle compliance**
3. **Missing is not the same as incorrect**
4. **Do not over-penalize minor formatting differences**
5. **If information is clearly fabricated, mark it incorrect and set hallucination = yes**
