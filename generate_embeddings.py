import time
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from config import EMBEDDING_MODEL, EMBEDDINGS_PATH, CORPUS_PATH

def main():
    df = pd.read_csv(CORPUS_PATH)
    sentences = df["text"].astype(str).tolist()

    model = SentenceTransformer(EMBEDDING_MODEL)

    start = time.time()
    embeddings = model.encode(
        sentences,
        batch_size=64,
        show_progress_bar=True,
        convert_to_numpy=True
    )
    end = time.time()

    np.save(EMBEDDINGS_PATH, embeddings)

    print(f"Embeddings generated: {embeddings.shape}")
    print(f"Embedding time: {end - start:.2f} seconds")

if __name__ == "__main__":
    main()
