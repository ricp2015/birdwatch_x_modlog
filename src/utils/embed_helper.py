"""
embed_helper.py
===============
Standalone embedding script, called as a subprocess by step3_expert_finder.py.

Running in a fresh process avoids the macOS fork+PyTorch segfault that occurs
when sentence-transformers is imported inside a process that was forked.

Arguments:
  --texts-path   path to a .pkl file containing a list of strings to embed
  --out-path     path where the output .npy array will be saved
  --model-name   HuggingFace model name
  --batch-size   sentences per batch (default 64)
  --normalize    if set, L2-normalise the output embeddings
"""

import argparse
import os
import pickle

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np
from sentence_transformers import SentenceTransformer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--texts-path",  required=True)
    parser.add_argument("--out-path",    required=True)
    parser.add_argument("--model-name",  default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--batch-size",  type=int, default=64)
    parser.add_argument("--normalize",   action="store_true")
    args = parser.parse_args()

    with open(args.texts_path, "rb") as f:
        texts = pickle.load(f)

    model = SentenceTransformer(args.model_name)

    embeddings = model.encode(
        texts,
        batch_size=args.batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=args.normalize,
    ).astype(np.float32)

    np.save(args.out_path, embeddings)
    print(f"Saved {embeddings.shape} to {args.out_path}")


if __name__ == "__main__":
    main()