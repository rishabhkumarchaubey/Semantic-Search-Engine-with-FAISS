import time
import faiss
import numpy as np
from config import (
    EMBEDDING_DIM,
    EMBEDDINGS_PATH,
    FAISS_INDEX_PATH,
    N_LIST
)

def main():
    print("Loading embeddings...")
    embeddings = np.load(EMBEDDINGS_PATH).astype("float32")

    print(f"Embeddings loaded: {embeddings.shape}")

    n_vectors = embeddings.shape[0]
    nlist = min(N_LIST, max(1, n_vectors // 10))

    print(f"Using IVF with {nlist} clusters")

    quantizer = faiss.IndexFlatL2(EMBEDDING_DIM)
    index = faiss.IndexIVFFlat(
        quantizer,
        EMBEDDING_DIM,
        nlist,
        faiss.METRIC_L2
    )

    print("Training FAISS index...")
    start = time.time()
    index.train(embeddings)

    print("Adding vectors to index...")
    index.add(embeddings)
    end = time.time()

    faiss.write_index(index, FAISS_INDEX_PATH)

    print("FAISS index saved successfully")
    print(f"Total vectors indexed: {index.ntotal}")
    print(f"Indexing time: {end - start:.2f} seconds")

if __name__ == "__main__":
    main()
 