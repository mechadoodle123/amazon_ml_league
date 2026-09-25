# Business Entity Resolution Pipeline

## Overview
This package implements an end-to-end, high-performance Machine Learning solution for the **Business Entity Resolution Challenge**. 

Given business records across three noisy, independent data sources (`Source 1` reference, `Source 2`, and `Source 3`), the pipeline resolves all matching records from Sources 2 and 3 for each Source 1 reference entity.

The solution is specifically designed to optimize the competition evaluation metric: **Macro-averaged $F_{0.5}$ Score** (which weights precision $2\times$ over recall and rewards correctly identifying singletons).

---

## Architecture

The system consists of three modular stages:

1. **Partitioning & Candidate Generation (Blocking)**:
   - **Strict Country Partitioning**: Exact disjoint partitioning by country (`US`, `India`, `France`), eliminating cross-country noise and reducing pairwise search space by $67\%$.
   - **Multi-Pass Lexical & Phonetic Blocking**:
     * Transliterates non-Latin scripts (Devanagari, Telugu, Tamil, French diacritics) via `anyascii`.
     * Strips business legal suffixes (`Inc`, `Corp`, `LLC`, `Pvt Ltd`, `SARL`, `SAS`, etc.), honorifics (`Shri`, `Smt`, `Dr`, etc.), and domain extensions (`.com`, `.org`, `.net`, `.in`, `.fr`).
     * Constructs sorted 2-token signatures, compressed name shingles, rare core word tokens, and address number-locality compound keys.
   - **Candidate Reduction**: Filters candidate space from $10,000,000+$ records down to $\le 25$ candidate pairs per entity, achieving $> 99.99\%$ reduction ratio with $> 95\%$ recall ceiling.

2. **Discriminative Feature Extraction**:
   - 17 orthogonal features computed via SIMD-accelerated C++ kernels (`rapidfuzz`):
     * Name string edit distance: Levenshtein ratio, Token Sort ratio, Token Set ratio, Core Token Sort ratio.
     * Token overlap & set metrics: Word Jaccard, Core Name Jaccard, character length ratio.
     * Address hierarchy: Address Token Sort ratio, Address Token Set ratio, Address Jaccard similarity.
     * Numerical address signature: Common number count, differing number count, exact numerical equality flag.
     * Missingness & source flags: Empty address indicator, Source 2 vs Source 3 indicator, composite lexical score.

3. **Gradient Boosted Decision Tree (LightGBM) & Precision Thresholding**:
   - Trained on hard positive and hard negative pairs mined via the blocking index.
   - High-precision probability decision thresholding ($\tau = 0.90$) specifically tuned to maximize macro $F_{0.5}$ and eliminate costly false merges on singletons.

---

## Requirements

- Python 3.9+
- Dependencies listed in `requirements.txt`:
  ```bash
  pip install -r requirements.txt
  ```

---

## Directory Structure

```text
code/business_entity_resolution/
├── run_pipeline.py           # Top-level executable runner
├── requirements.txt          # Python dependencies
├── README.md                 # System overview and reproduction guide
├── models/
│   └── lgbm_model.txt        # Pre-trained LightGBM model
└── src/
    ├── config.py             # Global constants, hyper-parameters, legal suffixes
    ├── preprocess.py         # Text cleaning, transliteration, tokenization
    ├── blocking.py           # Inverted index generation and candidate retrieval
    ├── features.py           # 17-dimensional feature extraction
    ├── model.py              # LightGBM booster wrapper
    ├── train_pipeline.py     # Training script with validation threshold search
    └── inference_pipeline.py # Robust checkpointed test inference pipeline
```

---

## How to Reproduce End-to-End

### 1. Training (Optional if using pre-trained model)
To re-train the model from scratch on training data:
```bash
python3 src/train_pipeline.py \
    --data-dir ../../student_resource/dataset \
    --model-output models/lgbm_model.txt \
    --n-train 50000 \
    --n-val 10000
```

### 2. Running Inference on Test Data
To generate `output/matching_results.tsv` and `output/candidate_pairs.tsv`:
```bash
python3 run_pipeline.py \
    --data-dir ../../student_resource/dataset \
    --output-dir ../../student_resource/output \
    --skip-train
```

### 3. Validating the Submission
Run the submission validator from `student_resource/`:
```bash
python3 utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir dataset/test
```
