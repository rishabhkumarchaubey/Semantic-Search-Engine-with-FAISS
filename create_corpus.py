import pandas as pd

base_sentences = [
    "AI transforms healthcare diagnostics",
    "AI improves patient outcome prediction",
    "Machine learning assists doctors in diagnosis",
    "Deep learning enhances medical imaging",
    "Healthcare startups innovate using AI",
    "AI enables early disease detection",
    "Artificial intelligence supports clinical decision making",
    "Neural networks analyze medical records",
]

corpus = []
for i in range(1, 626):  # 8 × 625 = 5000
    for s in base_sentences:
        corpus.append(f"{s} (sample {i})")

df = pd.DataFrame({"text": corpus})
df.to_csv("data/corpus.csv", index=False)

print("Created corpus.csv with", len(df), "unique rows")
