import os
import sys
import argparse
import time
import random
import numpy as np
from collections import defaultdict, Counter

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import DEFAULT_DATA_DIR, MODELS_DIR, FEATURE_NAMES, MAX_BUCKET_SIZE, MAX_CANDIDATES
from blocking import generate_blocking_keys
from features import extract_pair_features
from model import create_model, save_model

def compute_macro_f05(ground_truth: dict, predictions: dict) -> float:
    scores = []
    for s1_id, true_set in ground_truth.items():
        pred_set = predictions.get(s1_id, set())
        tp = len(true_set & pred_set)
        fp = len(pred_set - true_set)
        fn = len(true_set - pred_set)
        
        if len(true_set) == 0:
            scores.append(1.0 if len(pred_set) == 0 else 0.0)
        else:
            if len(pred_set) == 0:
                scores.append(0.0)
            else:
                prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                denom = 0.25 * prec + rec
                f05 = (1.25 * prec * rec / denom) if denom > 0 else 0.0
                scores.append(f05)
    return float(np.mean(scores))

def train(data_dir: str, model_path: str, n_train: int = 50000, n_val: int = 10000):
    print("="*80)
    print("STARTING ENTITY RESOLUTION MODEL TRAINING")
    print(f"Data Dir: {data_dir}")
    print(f"Target Model Path: {model_path}")
    print("="*80)
    
    t_start = time.time()
    n_total = n_train + n_val
    s1_all = {}
    gt_all = {}
    needed_s1 = set()
    needed_m = set()

    gt_path = os.path.join(data_dir, "train/train_ground_truth.tsv")
    print(f"Reading ground truth: {gt_path} ...")
    with open(gt_path, "r", encoding="utf-8") as f:
        f.readline()
        for line in f:
            if len(s1_all) >= n_total:
                break
            parts = line.rstrip("\n").split("\t")
            s1_id = parts[0]
            m_ids = parts[1].split(",") if len(parts) > 1 and parts[1].strip() else []
            s1_all[s1_id] = None
            gt_all[s1_id] = set(m_ids)
            needed_s1.add(s1_id)
            needed_m.update(m_ids)

    print(f"Sampled {len(s1_all):,} S1 entities ({len(needed_m):,} true matching links).")

    # Load S1 records
    loaded_s1 = 0
    s1_path = os.path.join(data_dir, "train/train_source1.tsv")
    with open(s1_path, "r", encoding="utf-8") as f:
        f.readline()
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if parts[0] in needed_s1:
                s1_all[parts[0]] = (parts[1], parts[2], parts[3])
                loaded_s1 += 1
                if loaded_s1 == len(needed_s1):
                    break

    print(f"Loaded {loaded_s1:,} Source 1 records.")

    # Load matching S2 and S3 records + distractors
    s23_pool = {}
    distractors_added = 0
    DISTRACTOR_LIMIT = 250000

    for src_name in ["train/train_source2.tsv", "train/train_source3.tsv"]:
        path = os.path.join(data_dir, src_name)
        with open(path, "r", encoding="utf-8") as f:
            f.readline()
            for line in f:
                parts = line.rstrip("\n").split("\t")
                mid = parts[0]
                if mid in needed_m:
                    s23_pool[mid] = (parts[1], parts[2] if len(parts) > 2 else "", parts[3] if len(parts) > 3 else "")
                elif distractors_added < DISTRACTOR_LIMIT:
                    s23_pool[mid] = (parts[1], parts[2] if len(parts) > 2 else "", parts[3] if len(parts) > 3 else "")
                    distractors_added += 1

    print(f"Pool size: {len(s23_pool):,} records. Building inverted blocking index...")
    t_idx = time.time()
    inverted_index = defaultdict(list)
    for mid, rec in s23_pool.items():
        for k in generate_blocking_keys(rec[0], rec[1], rec[2]):
            inverted_index[k].append(mid)
    print(f"Index built with {len(inverted_index):,} keys in {time.time() - t_idx:.2f}s.")

    s1_keys = list(s1_all.keys())
    train_s1_keys = set(s1_keys[:n_train])
    val_s1_keys = set(s1_keys[n_train:])
    gt_val = {k: gt_all[k] for k in val_s1_keys}

    print("\nExtracting feature vectors for training candidate pairs...")
    X_train = []
    y_train = []
    
    for s1_id in train_s1_keys:
        rec1 = s1_all[s1_id]
        true_m = gt_all[s1_id]
        keys = generate_blocking_keys(rec1[0], rec1[1], rec1[2])
        key_hits = Counter()
        for k in keys:
            bucket = inverted_index.get(k, [])
            if 0 < len(bucket) <= MAX_BUCKET_SIZE:
                for mid in bucket:
                    key_hits[mid] += 1
        
        cands = {mid for mid, _ in key_hits.most_common(MAX_CANDIDATES)}
        pos = list(cands & true_m)
        negs = list(cands - true_m)
        
        # Include all true positive candidates and sample hard negatives
        selected_negs = random.sample(negs, min(len(negs), 8))
        
        for mid in pos:
            feats = extract_pair_features(rec1, s23_pool[mid], mid)
            X_train.append(feats)
            y_train.append(1)
        for mid in selected_negs:
            feats = extract_pair_features(rec1, s23_pool[mid], mid)
            X_train.append(feats)
            y_train.append(0)

    X_train = np.array(X_train, dtype=np.float32)
    y_train = np.array(y_train, dtype=np.int32)
    print(f"X_train shape: {X_train.shape} (Positives: {np.sum(y_train):,}, Negatives: {len(y_train)-np.sum(y_train):,})")

    print("\nTraining LightGBM model...")
    clf = create_model(n_estimators=150, learning_rate=0.08, max_depth=6)
    clf.fit(X_train, y_train)

    print("\nFeature importances:")
    for name, imp in sorted(zip(FEATURE_NAMES, clf.feature_importances_), key=lambda x: -x[1]):
        print(f"  {name:20s}: {imp}")

    print(f"\nEvaluating on {len(val_s1_keys):,} validation entities...")
    val_pairs = []
    X_val_list = []
    for s1_id in val_s1_keys:
        rec1 = s1_all[s1_id]
        keys = generate_blocking_keys(rec1[0], rec1[1], rec1[2])
        key_hits = Counter()
        for k in keys:
            bucket = inverted_index.get(k, [])
            if 0 < len(bucket) <= MAX_BUCKET_SIZE:
                for mid in bucket:
                    key_hits[mid] += 1
        top_cands = [mid for mid, _ in key_hits.most_common(MAX_CANDIDATES)]
        for mid in top_cands:
            feats = extract_pair_features(rec1, s23_pool[mid], mid)
            val_pairs.append((s1_id, mid))
            X_val_list.append(feats)

    X_val = np.array(X_val_list, dtype=np.float32)
    val_probs = clf.predict_proba(X_val)[:, 1]

    s1_scored = defaultdict(list)
    for (s1_id, mid), prob in zip(val_pairs, val_probs):
        s1_scored[s1_id].append((mid, prob))

    best_thresh = 0.90
    best_f05 = 0.0
    for thresh in np.arange(0.70, 0.96, 0.05):
        preds = {}
        for s1_id in val_s1_keys:
            preds[s1_id] = {mid for mid, p in s1_scored[s1_id] if p >= thresh}
        f05 = compute_macro_f05(gt_val, preds)
        print(f"  Threshold {thresh:.2f} --> Validation Macro F_0.5 = {f05:.4f}")
        if f05 > best_f05:
            best_f05 = f05
            best_thresh = thresh

    print(f"\n>>> Best Validation Macro F_0.5: {best_f05:.4f} at threshold {best_thresh:.2f} <<<")

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    save_model(clf, model_path)
    print(f"Model saved to {model_path} (size: {os.path.getsize(model_path):,} bytes).")
    print(f"Total training pipeline completed in {time.time() - t_start:.2f}s.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Entity Resolution Matching Model")
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR, help="Path to dataset root folder")
    parser.add_argument("--model-output", default=os.path.join(MODELS_DIR, "lgbm_model.txt"), help="Path to save trained model")
    parser.add_argument("--n-train", type=int, default=50000, help="Number of training S1 entities")
    parser.add_argument("--n-val", type=int, default=10000, help="Number of validation S1 entities")
    args = parser.parse_args()

    train(args.data_dir, args.model_output, args.n_train, args.n_val)
