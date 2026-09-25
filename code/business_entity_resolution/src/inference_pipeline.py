import os
import sys
import gc
import time
import argparse
from collections import defaultdict, Counter
import numpy as np

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import DEFAULT_DATA_DIR, OUTPUT_DIR, MODELS_DIR, MAX_CANDIDATES, MAX_BUCKET_SIZE, MATCH_THRESHOLD
from preprocess import clean_text
from blocking import generate_blocking_keys, build_country_inverted_index
from features import extract_pair_features
from model import load_model, predict_probabilities

def run_inference(
    data_dir: str,
    model_path: str,
    output_dir: str,
    match_threshold: float = MATCH_THRESHOLD,
    max_cands: int = MAX_CANDIDATES
):
    print("="*80)
    print("STARTING ROBUST SEQUENTIAL ENTITY RESOLUTION INFERENCE PIPELINE")
    print(f"Data directory:     {data_dir}")
    print(f"Model path:         {model_path}")
    print(f"Output directory:   {output_dir}")
    print(f"Decision threshold: {match_threshold}")
    print(f"Max candidates:     {max_cands}")
    print("="*80)

    t_start = time.time()
    os.makedirs(output_dir, exist_ok=True)
    temp_dir = os.path.join(output_dir, "checkpoints")
    os.makedirs(temp_dir, exist_ok=True)

    test_s1_path = os.path.join(data_dir, "test/test_source1.tsv")
    test_s2_path = os.path.join(data_dir, "test/test_source2.tsv")
    test_s3_path = os.path.join(data_dir, "test/test_source3.tsv")

    print(f"\n1. Loading test Source 1 entities from {test_s1_path}...")
    s1_order = []
    s1_by_country = defaultdict(list)
    with open(test_s1_path, "r", encoding="utf-8") as f:
        f.readline()  # skip header
        for line in f:
            parts = line.rstrip("\n").split("\t")
            s1_id = parts[0]
            name = parts[1]
            addr = parts[2] if len(parts) > 2 else ""
            country = parts[3] if len(parts) > 3 else "US"
            s1_order.append(s1_id)
            s1_by_country[country].append((s1_id, name, addr, country))

    print(f"Loaded {len(s1_order):,} total Source 1 entities.")
    for c, items in s1_by_country.items():
        print(f"  {c:10s}: {len(items):,} entities")

    print(f"\n2. Loading trained LightGBM model from {model_path}...")
    booster = load_model(model_path)
    print("Model loaded successfully.")

    countries = ["France", "US", "India"]
    BATCH_SIZE = 5000

    for country in countries:
        ckpt_path = os.path.join(temp_dir, f"{country.lower()}_ckpt.tsv")
        s1_list = s1_by_country.get(country, [])
        if not s1_list:
            continue

        # Check if already completed from a previous run
        if os.path.isfile(ckpt_path):
            with open(ckpt_path, "r", encoding="utf-8") as f_check:
                lines_in_ckpt = sum(1 for _ in f_check)
            if lines_in_ckpt == len(s1_list):
                print(f"\nFound existing valid checkpoint for {country} ({lines_in_ckpt:,} rows). Skipping to next country.")
                continue
            else:
                print(f"\nIncomplete checkpoint for {country} ({lines_in_ckpt:,}/{len(s1_list):,} rows). Recomputing...")
                os.remove(ckpt_path)

        t_c = time.time()
        print(f"\n" + "-"*60)
        print(f"Processing Country: {country} ({len(s1_list):,} S1 entities)...")
        print(f"-"*60)

        print(f"  Building inverted index from S2 and S3 for {country}...")
        inverted_index, s23_records = build_country_inverted_index(
            test_s2_path, test_s3_path, country
        )
        print(f"  Index built: {len(s23_records):,} records, {len(inverted_index):,} blocking keys in {time.time() - t_c:.2f}s.")

        print(f"  Scoring S1 entities sequentially and streaming checkpoint to {ckpt_path}...")
        n_processed = 0
        total_cands = 0
        total_matches = 0

        with open(ckpt_path, "w", encoding="utf-8") as f_out:
            for start_idx in range(0, len(s1_list), BATCH_SIZE):
                batch_s1 = s1_list[start_idx : start_idx + BATCH_SIZE]
                batch_pairs = []
                batch_features = []
                batch_cands = {}

                for s1_id, name, addr, c_name in batch_s1:
                    r1 = (name, addr, c_name)
                    keys = generate_blocking_keys(name, addr, c_name)
                    key_hits = Counter()
                    for k in keys:
                        bucket = inverted_index.get(k, [])
                        if 0 < len(bucket) <= MAX_BUCKET_SIZE:
                            for mid in bucket:
                                key_hits[mid] += 1

                    top_cands = [mid for mid, _ in key_hits.most_common(max_cands)]
                    batch_cands[s1_id] = top_cands
                    total_cands += len(top_cands)

                    for mid in top_cands:
                        r2 = s23_records[mid]
                        feats = extract_pair_features(r1, r2, mid)
                        batch_pairs.append((s1_id, mid))
                        batch_features.append(feats)

                # Batch LightGBM prediction
                matched_by_s1 = {s1[0]: [] for s1 in batch_s1}
                if batch_features:
                    X_batch = np.array(batch_features, dtype=np.float32)
                    probs = predict_probabilities(booster, X_batch)
                    for (s1_id, mid), p in zip(batch_pairs, probs):
                        if p >= match_threshold:
                            matched_by_s1[s1_id].append(mid)

                for s1_id, _, _, _ in batch_s1:
                    c_str = ",".join(batch_cands[s1_id])
                    m_str = ",".join(matched_by_s1[s1_id])
                    total_matches += len(matched_by_s1[s1_id])
                    # Format: s1_id \t candidate_str \t matched_str
                    f_out.write(f"{s1_id}\t{c_str}\t{m_str}\n")

                n_processed += len(batch_s1)
                if n_processed % 25000 == 0 or n_processed == len(s1_list):
                    elapsed = time.time() - t_c
                    print(f"    Scored {n_processed:,} / {len(s1_list):,} S1 entities ({n_processed / elapsed:.1f} S1/sec)...")

        elapsed_c = time.time() - t_c
        print(f"  Finished {country} in {elapsed_c:.2f}s ({len(s1_list) / elapsed_c:.1f} S1/sec).")
        print(f"  Avg candidates/S1: {total_cands / len(s1_list):.2f}, Avg matches/S1: {total_matches / len(s1_list):.2f}")

        # Free memory immediately
        del inverted_index, s23_records
        gc.collect()

    print(f"\n3. Compiling final output files from checkpoints to {output_dir}...")
    # Load all country checkpoints
    ckpt_data = {}
    for country in countries:
        ckpt_path = os.path.join(temp_dir, f"{country.lower()}_ckpt.tsv")
        print(f"  Reading {ckpt_path} ...")
        with open(ckpt_path, "r", encoding="utf-8") as f_in:
            for line in f_in:
                parts = line.rstrip("\n").split("\t")
                s1_id = parts[0]
                c_str = parts[1] if len(parts) > 1 else ""
                m_str = parts[2] if len(parts) > 2 else ""
                ckpt_data[s1_id] = (c_str, m_str)

    print(f"  Loaded {len(ckpt_data):,} records from checkpoints.")

    matching_tsv_path = os.path.join(output_dir, "matching_results.tsv")
    candidate_tsv_path = os.path.join(output_dir, "candidate_pairs.tsv")

    print(f"  Writing {matching_tsv_path} ...")
    with open(matching_tsv_path, "w", encoding="utf-8") as f_match:
        f_match.write("source1_entity_id\tmatched_entity_ids\n")
        for s1_id in s1_order:
            _, m_str = ckpt_data.get(s1_id, ("", ""))
            f_match.write(f"{s1_id}\t{m_str}\n")

    print(f"  Writing {candidate_tsv_path} ...")
    with open(candidate_tsv_path, "w", encoding="utf-8") as f_cand:
        f_cand.write("source1_entity_id\tcandidate_entity_ids\n")
        for s1_id in s1_order:
            c_str, _ = ckpt_data.get(s1_id, ("", ""))
            f_cand.write(f"{s1_id}\t{c_str}\n")

    print(f"\nOutput files generated successfully:")
    print(f"  matching_results.tsv : {os.path.getsize(matching_tsv_path):,} bytes")
    print(f"  candidate_pairs.tsv  : {os.path.getsize(candidate_tsv_path):,} bytes")
    print(f"Total inference pipeline completed in {time.time() - t_start:.2f}s.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Entity Resolution Inference on Test Data")
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR, help="Path to dataset root folder")
    parser.add_argument("--model-path", default=os.path.join(MODELS_DIR, "lgbm_model.txt"), help="Path to trained model")
    parser.add_argument("--output-dir", default=OUTPUT_DIR, help="Path to write matching_results.tsv and candidate_pairs.tsv")
    parser.add_argument("--threshold", type=float, default=MATCH_THRESHOLD, help="Match probability threshold")
    parser.add_argument("--max-candidates", type=int, default=MAX_CANDIDATES, help="Max candidates per S1 entity")
    args = parser.parse_args()

    run_inference(args.data_dir, args.model_path, args.output_dir, args.threshold, args.max_candidates)
