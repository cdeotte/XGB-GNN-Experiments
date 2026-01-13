import pandas as pd

for index in range(61):
    author_features = pd.read_parquet(f"arxiv_author_graphs/nodes_{index}.pqt")
    author_targets = pd.read_parquet(f"arxiv_author_graphs/node_targets_{index}.pqt")
    author_edges = pd.read_parquet(f"arxiv_author_graphs/edges_{index}.pqt")

print(author_features.head())
print(author_targets.head())
print(author_edges.head())
