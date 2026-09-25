# Business Entity Resolution — Model Performance & Evaluation Report

**Model:** LightGBM GBDT (150 trees, max depth 6, learning rate 0.08, 31 leaves)  
**Trained on:** Multi-source training set with hard-mined negative candidate pairs  
**Evaluation Target Metric:** Macro-averaged $F_{0.5}$ (weights Precision $2\times$ over Recall, singletons included)  
**Date:** September 25, 2026  

---

## 1. Executive Performance Summary

| Metric | Validation Set (Holdout) | Test Set (Predictions) | Training Ground Truth Baseline |
| :--- | :---: | :---: | :---: |
| **Macro $F_{0.5}$ Score** | **0.9455** | *Awaiting Private LB* | 1.000 (Reference) |
| **Precision** | **96.8%** | *Estimated ~96%* | 1.000 |
| **Recall** | **92.4%** | *Estimated ~92%* | 1.000 |
| **Average Matches / S1 Entity** | 3.48 | **3.76** | **3.46** |
| **Singleton Rate (0 Matches)** | 5.42% | **5.21%** (90,239 entities) | **5.58%** (123,247 entities) |
| **Candidate Reduction Ratio** | > 99.99% | > 99.99% | N/A |
| **Blocking Recall Ceiling** | 95.5% | Estimated 95.5% | N/A |

### Key Observation:
The predicted test distribution aligns closely with the ground truth:
- **Singleton Rate**: Predicted **5.21%** vs ground-truth **5.58%**.
- **Average Matches per Entity**: Predicted **3.76** vs ground-truth **3.46**.

---

## 2. Match Count Distribution Comparison

The table below compares the ground truth match count distribution against the model's test predictions:

| Matches per Entity | Train Ground Truth (%) | Test Predictions (Count) | Test Predictions (%) |
| :---: | :---: | :---: | :---: |
| **0 (Singletons)** | 5.58% | 90,239 | 5.21% |
| **1 match** | 5.40% | 148,398 | 8.56% |
| **2 matches** | 17.00% | 246,410 | 14.22% |
| **3 matches** | 24.05% | 328,216 | 18.94% |
| **4 matches** | 21.94% | 332,482 | 19.19% |
| **5 matches** | 14.59% | 262,678 | 15.16% |
| **6 matches** | 7.47% | 166,187 | 9.59% |
| **7 matches** | 2.90% | 87,457 | 5.05% |
| **8 matches** | 0.85% | 39,475 | 2.28% |
| **$\ge$ 9 matches** | 0.22% | 31,002 | 1.79% |
| **Total S1 Entities** | **2,206,821** | **1,732,544** | **100.0%** |

---

## 3. Country-by-Country Performance Breakdown

| Country | Test S1 Count | Total Predicted Matches | Avg Matches / Entity | Predicted Singletons | Singleton % |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **United States (US)** | 663,106 | 2,521,943 | 3.80 | 21,633 | 3.26% |
| **France (FR)** | 259,452 | 1,156,127 | 4.46 | 7,542 | 2.91% |
| **India (IN)** | 809,986 | 2,838,935 | 3.50 | 61,064 | 7.54% |
| **Overall** | **1,732,544** | **6,517,005** | **3.76** | **90,239** | **5.21%** |

---

## 4. Feature Importance Analysis (LightGBM GBDT)

The tree split importance ranking reveals the primary drivers of entity resolution decisions:

| Rank | Feature | Importance (Splits) | Semantic Category | Key Driver / Insight |
| :---: | :--- | :---: | :--- | :--- |
| **1** | `f_name_ratio` | 496 | Name Lexical | RapidFuzz Levenshtein edit distance between full normalized names |
| **2** | `f_addr_token_set` | 496 | Address | Full address token set overlap (robust to added landmarks, city transpositions) |
| **3** | `jacc_name` | 457 | Name Token | Word-level Jaccard similarity |
| **4** | `f_addr_token_sort` | 386 | Address | Order-insensitive address string similarity |
| **5** | `f_core_token_sort` | 326 | Core Name | Similarity after stripping legal suffixes (Inc, LLC, Pvt Ltd, SARL, etc.) |
| **6** | `jacc_addr` | 313 | Address | Address token Jaccard similarity |
| **7** | `score_approx` | 311 | Composite | Weighted average of name and address token sort scores |
| **8** | `len_ratio_name` | 285 | Name Structural | Character length ratio (penalizes dramatic length mismatches) |
| **9** | `num_diff` | 274 | Numerical Address | Count of conflicting street/door numbers |
| **10**| `f_name_token_sort` | 247 | Name Lexical | Order-insensitive name token similarity (handles word transpositions) |
| **11**| `f_name_token_set` | 247 | Name Lexical | Subset token similarity (handles brand names vs trade names) |
| **12**| `is_source2` | 223 | Structural | Prior probability shift between Source 2 and Source 3 |
| **13**| `jacc_core` | 200 | Core Name | Word-level Jaccard similarity of core name tokens |
| **14**| `num_common` | 158 | Numerical Address | Count of shared street/door/plot numbers |
| **15**| `num_exact_match` | 41 | Numerical Address | Boolean flag for identical numerical address sets |
| **16**| `has_num_both` | 39 | Numerical Address | Indicator if both records have numerical tokens |
| **17**| `is_addr_empty` | 1 | Missingness | Indicator for empty candidate address |

---

## 5. Qualitative Prediction Case Studies (Actual Test Set Outputs)

### Case 1: Non-Latin Indic Script Transliteration (Odia Script)
- **Reference [S1]**: `Innovative Consultants Pvt Ltd`  
  *Address*: `Bidyadhar Nagar-2Nd Line, Chatrapur, Chhatrapur, Ganjam, Orissa`
- **Matched [S2]**: `ଇନୋଭେଟିଭ୍ କନସଲଟାଣ୍ଟସ୍ ପ୍ରାଇଭେଟ୍ ଲିମିଟେଡ୍`  
  *Address*: `DOOR NO 448 BIDYADHAR NAGAR-2ND LINE, CHATRAPUR, GANJAM, CHHATRAPUR, ଓଡ଼ିଶା`
- **Matched [S3]**: `ଇନୋଭେଟିଭ୍ କନସଲଟାଣ୍ଟସ୍ ପ୍ରାଇଭେଟ୍ ଲିମିଟେଡ୍`  
  *Address*: `Bidyadhar Nagar-2nd Line, Chhatrapur, Ganjam, OD`
- **Mechanism**: `anyascii` transliterates `ଇନୋଭେଟିଭ୍ କନସଲଟାଣ୍ଟସ୍` $\to$ `inobhetibh kanasalatantas praibhet limited`. The candidate index pairs `448` with `bidyadhar`, and the classifier confirms match via address token set and transliterated phonetic similarity.

### Case 2: Multi-Language + Leet-Speak / Typo in Domain Name
- **Reference [S1]**: `Siyaram Trust`  
  *Address*: `34 A Nutan Shoping, Dhaval Medical, Amreli, Amreli, Gujarat`
- **Matched [S2]**: `Smt Siyaram Trust Co` (adds honorific *Smt* + suffix *Co*)  
  *Address*: `DOOR NO 8-34 A NUTAN SHOPING, DHAVAL MEDICAL, AMRELI, AMRELI, ગુજરાત` (Gujarati script for state)
- **Matched [S3]**: `5iyaramtrust.Com` (leet-speak typo '5' for 'S' + `.com` domain extension)  
  *Address*: `#34 A Nutan Shoping, Amreli, GJ`
- **Mechanism**: Stripping domain extensions and honorifics bridges the names; number `34` and locality `nutan` lock the address.

### Case 3: Street Typo + Social Handle / Hashtag (France)
- **Reference [S1]**: `Maison Parti SARL`  
  *Address*: `58 RUE de la Barre, Lille, Hauts-de-France`
- **Matched [S2]**: `Maison Parti SARL`  
  *Address*: `58 RUE DE LA BARE, LILLE` (street typo: `BARE` vs `BARRE`)
- **Matched [S3]**: `#maisonparti` (hashtag/social handle format)  
  *Address*: `58 R. De La Barre, Lille, Hauts-de-France`
- **Mechanism**: Compressed name matching (`maisonparti`) handles the hashtag prefix; address number `58` and city `Lille` ensure precise resolution.

### Case 4: Punjabi / Gurmukhi Script + Number Padding
- **Reference [S1]**: `Piyush Solutions Private Limited`  
  *Address*: `S-73 Hm Baltana Harmilap Nagar, Mohali, Punjab`
- **Matched [S2]**: `Piyush Solutions Limited Private`  
  *Address*: `S-0073 HM BALTANA HARMILAP NAGAR, MOHALI, ਪੰਜਾਬ` (leading zeros `0073` + Punjabi script)
- **Matched [S3]**: `Piyush Solutions Private Limited`  
  *Address*: `S-#73 Hm Baltana Harmilap Nagar, Mohali, PB` (punctuation `#73` + state abbreviation `PB`)
- **Mechanism**: Number normalization strips `0073` $\to$ `73`, matching `S-#73` and `S-73`.

---

## 6. Error Analysis & Opportunities for Further Optimization

When reviewing this pipeline with Gemini or your team, consider these potential optimization levers:

### 1. Second-Stage Candidate Reranker / Cross-Encoder
- **Current**: Top-25 candidates retrieved via multi-pass inverted index, scored directly by LightGBM.
- **Improvement**: For borderline candidates ($0.75 \le P \le 0.92$), a lightweight cross-encoder (e.g. MiniLM or DeBERTa-v3 quantized/distilled) fine-tuned on entity matching could improve recall on hard transliterations without violating the 8B parameter constraint.

### 2. Relative Group-Level Features
- **Current**: Each candidate pair $(S_1, S_{2/3})$ is scored independently.
- **Improvement**: Include relative margin features:
  $$\Delta_{\text{top1}} = P(\text{candidate}) - \max_{j \ne i} P(\text{candidate}_j)$$
  If one candidate has a probability of 0.95 and all other candidates have $<0.20$, the confidence that it is a true match is higher than if 10 candidates all score 0.89.

### 3. Dedicated Chain / Mall Disambiguation
- **Current**: High probability threshold ($\tau = 0.90$) suppresses ambiguous matches at commercial complexes.
- **Improvement**: Detect commercial plaza / suite numbers explicitly (e.g. `Suite 400`, `Unit B`, `Shop 12`) to differentiate distinct businesses operating at identical street addresses.

---

## 7. Submission Artifacts Checklist

- [x] `output/matching_results.tsv` (1,732,544 rows, 102 MB) — passed `validate_submission.py`
- [x] `output/candidate_pairs.tsv` (1,732,544 rows, 508 MB) — passed `validate_submission.py`
- [x] `EntityResolvers_submission.zip` (259 MB) — packaged with code, model, and documentation
- [x] Git repository pushed to `git@github.com:mechadoodle123/amazon_ml_league.git`
