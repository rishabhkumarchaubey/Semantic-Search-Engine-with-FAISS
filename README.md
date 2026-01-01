# Semantic Search Engine with FAISS

## Project Overview

This project implements a **Semantic Search Engine** using **Sentence Transformers** for generating dense text embeddings and **FAISS (Facebook AI Similarity Search)** for efficient approximate nearest neighbor search. The system allows users to input a natural language query and retrieves the **top-5 semantically similar sentences** from a large text corpus.

The project is designed to be **submission-ready**, **GitHub-friendly**, and suitable for **academic evaluation, viva, and portfolio demonstration**.

---

## Key Features

- Sentence-level semantic understanding using pretrained Transformer models
- Fast similarity search using FAISS IVF (Inverted File Index)
- Command-line interface (CLI) for querying
- Deduplication of near-identical results
- Performance metrics including indexing time and query latency
- Modular and clean code structure

---

## Technologies Used

- **Python 3.x**
- **sentence-transformers**
- **FAISS (CPU)**
- **Pandas**
- **NumPy**

---

## Project Structure

```
semantic-search-faiss/
│
├── data/
│   ├── corpus.csv
│   ├── embeddings.npy
│   └── faiss.index
│
├── src/
│   ├── config.py
│   ├── create_corpus.py
│   ├── generate_embeddings.py
│   ├── build_faiss_index.py
│   └── query_engine.py
│
├── metrics/
│   └── performance_metrics.txt
│
├── assets/
│   └── final_query_output.png
│
├── requirements.txt
└── README.md
```

---

## Dataset

- **Corpus Size:** 5,000 sentences
- **Format:** CSV file with a single column named `text`
- The corpus contains diverse healthcare and AI-related sentences with slight variations to simulate a realistic dataset.

Example (`data/corpus.csv`):

```csv
text
AI transforms healthcare diagnostics (sample 1)
AI improves patient outcome prediction (sample 1)
Deep learning enhances medical imaging (sample 1)
```

---

## Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/semantic-search-faiss.git
cd semantic-search-faiss
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Execution Pipeline (Step-by-Step)

### Step 1: Create the Corpus

```bash
python src/create_corpus.py
```

Output:

```
Created corpus.csv with 5000 unique rows
```

---

### Step 2: Generate Sentence Embeddings

```bash
python src/generate_embeddings.py
```

Sample Output:

```
Embeddings generated: (5000, 384)
Embedding time: ~40 seconds
```

Embeddings are saved as:

```
data/embeddings.npy
```

---

### Step 3: Build FAISS Index (IVF)

```bash
python src/build_faiss_index.py
```

Sample Output:

```
Total vectors indexed: 5000
Indexing time: ~2 seconds
```

FAISS index is saved as:

```
data/faiss.index
```

---

### Step 4: Query the Semantic Search Engine

```bash
python src/query_engine.py --query "AI in healthcare"
```

Sample Output:

```
Query: AI in healthcare

Top Results:
1. AI transforms healthcare diagnostics (sample 521)
2. AI improves patient outcome prediction (sample 87)
3. AI enables early disease detection (sample 39)
4. Deep learning enhances medical imaging (sample 118)
5. Neural networks analyze medical records (sample 301)

Query latency: ~1.2 ms
```

---

## Sample Output Screenshot

![Semantic Search Output](assets/final_query_output.png)

---

## Performance Metrics

| Metric                | Value           |
| --------------------- | --------------- |
| Corpus Size           | 5,000 sentences |
| Embedding Dimension   | 384             |
| Index Type            | FAISS IVF       |
| Indexing Time         | ~2 seconds      |
| Average Query Latency | ~1–3 ms         |
| Top-K Results         | 5               |

---

## Design Decisions

- **SentenceTransformer (all-MiniLM-L6-v2)** was chosen for its balance between speed and semantic quality.
- **FAISS IVF** was used to demonstrate scalable approximate nearest neighbor search.
- Result deduplication ensures meaningful and diverse outputs.
- Modular scripts improve readability and maintainability.

---

## Limitations

- Uses CPU-based FAISS (GPU version not enabled)
- Dataset is synthetic (can be replaced with real-world corpora)
- No web interface (CLI-based only)

---

## Future Enhancements

- Add FastAPI-based REST API
- Support FAISS GPU indexing
- Evaluate Recall@K and Precision@K
- Integrate real-world datasets (news, research papers)
- Add cosine similarity scores to results

---

## Conclusion

This project demonstrates an end-to-end implementation of a **semantic search system** using modern NLP embeddings and efficient vector search. It highlights how semantic similarity can be leveraged for fast and meaningful information retrieval at scale.

---

## Author

**Name:** Rishabh Kumar
**Course:** B.Tech in Artificial Intelligence & Machine Learning
**Date:** January 2026

---

## License

This project is for **educational and academic use**.
