#!/usr/bin/env python3
"""
End-to-End Pipeline Runner for Business Entity Resolution Challenge.

Usage:
    python3 run_pipeline.py [--data-dir path/to/dataset] [--output-dir path/to/output]
"""

import os
import sys
import argparse

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from config import DEFAULT_DATA_DIR, OUTPUT_DIR, MODELS_DIR, MATCH_THRESHOLD, MAX_CANDIDATES
from train_pipeline import train
from inference_pipeline import run_inference

def main():
    parser = argparse.ArgumentParser(description="Run End-to-End Business Entity Resolution Pipeline")
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR, help="Path to dataset root folder")
    parser.add_argument("--output-dir", default=OUTPUT_DIR, help="Path to output directory for submission TSVs")
    parser.add_argument("--model-path", default=os.path.join(MODELS_DIR, "lgbm_model.txt"), help="Path to trained model")
    parser.add_argument("--threshold", type=float, default=MATCH_THRESHOLD, help="Match probability threshold")
    parser.add_argument("--skip-train", action="store_true", help="Skip training if model file already exists")
    args = parser.parse_args()

    # Step 1: Train model if not existing or not skipped
    if not args.skip_train or not os.path.isfile(args.model_path):
        print("\n>>> STEP 1: TRAINING ENTITY RESOLUTION MODEL <<<")
        train(args.data_dir, args.model_path, n_train=50000, n_val=10000)
    else:
        print(f"\n>>> STEP 1: Reusing existing trained model at {args.model_path} <<<")

    # Step 2: Run inference on test data
    print("\n>>> STEP 2: RUNNING INFERENCE ON TEST DATA <<<")
    run_inference(args.data_dir, args.model_path, args.output_dir, args.threshold, MAX_CANDIDATES)

    print("\n>>> PIPELINE EXECUTION COMPLETE! <<<")

if __name__ == "__main__":
    main()
