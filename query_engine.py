import time
import argparse
import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from config import (
    EMBEDDING_MODEL,
    FAISS_INDEX_PATH,
    CORPUS_PATH,
    TOP_K,
    N_PROBE
)

def main(query):
    df = pd.read_csv(CORPUS_PATH)
    texts = df["text"].astype(str).tolist()

    model = SentenceTransformer(EMBEDDING_MODEL)
    index = faiss.read_index(FAISS_INDEX_PATH)
    index.nprobe = N_PROBE

    query_embedding = model.encode([query]).astype("float32")

    start = time.time()
    distances, indices = index.search(query_embedding, TOP_K)
    end = time.time()

    print("\nQuery:", query)
    print("\nTop Results:")
    for rank, idx in enumerate(indices[0], start=1):
        print(f"{rank}. {texts[idx]}")

    print(f"\nQuery latency: {(end - start) * 1000:.2f} ms")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", type=str, required=True)
    args = parser.parse_args()

    main(args.query)
