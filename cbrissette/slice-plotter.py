import os
import re
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import glob

# Configuration
INPUT_DIR = "arxiv_author_graphs"

def extract_k_from_filename(filename):
    match = re.search(r'_(\d+)\.pqt', filename)
    return int(match.group(1)) if match else -1

def analyze_slices():
    edge_files = glob.glob(os.path.join(INPUT_DIR, "edges_*.pqt"))
    edge_files.sort(key=extract_k_from_filename)
    
    results = []
    print(f"Found {len(edge_files)} slices. Analyzing...")

    for edge_file in edge_files:
        k = extract_k_from_filename(edge_file)
        node_file = os.path.join(INPUT_DIR, f"nodes_{k}.pqt")
        
        if not os.path.exists(node_file):
            continue

        # Load
        edges_df = pd.read_parquet(edge_file)
        nodes_df = pd.read_parquet(node_file)
        
        # Build Graph
        G = nx.Graph()
        G.add_nodes_from(nodes_df['id'])
        if not edges_df.empty:
            G.add_edges_from(zip(edges_df['source_id'], edges_df['target_id']))

        # Metrics
        num_nodes = G.number_of_nodes()
        num_edges = G.number_of_edges()
        num_components = nx.number_connected_components(G)
        
        if num_nodes > 0:
            avg_degree = (2 * num_edges) / num_nodes
        else:
            avg_degree = 0

        results.append({
            'slice': k,
            'num_nodes': num_nodes,
            'num_edges': num_edges, # Added to data collection
            'num_components': num_components,
            'avg_degree': avg_degree
        })
        
        print(f"Slice {k}: {num_nodes} nodes, {num_edges} edges")

    return pd.DataFrame(results)

def plot_results(df):
    if df.empty:
        print("No data to plot.")
        return

    # UPDATED LAYOUT: 3 Rows, 2 Columns
    # Row 1: Size (Nodes & Edges)
    # Row 2: Topology (Components & Degree)
    # Row 3: Distributions (Histograms)
    fig, axes = plt.subplots(3, 2, figsize=(14, 15))
    fig.suptitle(f"Graph Topology Analysis ({len(df)} Time Slices)", fontsize=16)

    # --- Plot 1: Number of Nodes (NEW) ---
    axes[0, 0].plot(df['slice'], df['num_nodes'], marker='.', color='green')
    axes[0, 0].set_title("Network Growth (Nodes)")
    axes[0, 0].set_ylabel("Count")
    axes[0, 0].grid(True, linestyle='--', alpha=0.6)

    # --- Plot 2: Number of Edges (Bonus) ---
    axes[0, 1].plot(df['slice'], df['num_edges'], marker='.', color='darkgreen')
    axes[0, 1].set_title("Network Growth (Edges)")
    axes[0, 1].set_ylabel("Count")
    axes[0, 1].grid(True, linestyle='--', alpha=0.6)

    # --- Plot 3: Connected Components ---
    axes[1, 0].plot(df['slice'], df['num_components'], marker='.', color='skyblue')
    axes[1, 0].set_title("Connected Components per Slice")
    axes[1, 0].set_ylabel("Count")
    axes[1, 0].grid(True, linestyle='--', alpha=0.6)

    # --- Plot 4: Average Degree ---
    axes[1, 1].plot(df['slice'], df['avg_degree'], marker='.', color='salmon')
    axes[1, 1].set_title("Average Degree per Slice")
    axes[1, 1].set_ylabel("Avg Degree")
    axes[1, 1].grid(True, linestyle='--', alpha=0.6)

    # --- Plot 5: Histogram (Components) ---
    axes[2, 0].hist(df['num_components'], bins=20, color='skyblue', edgecolor='black')
    axes[2, 0].set_title("Distribution of Components")
    axes[2, 0].set_xlabel("Number of Components")
    axes[2, 0].set_ylabel("Frequency")

    # --- Plot 6: Histogram (Degree) ---
    axes[2, 1].hist(df['avg_degree'], bins=20, color='salmon', edgecolor='black')
    axes[2, 1].set_title("Distribution of Average Degree")
    axes[2, 1].set_xlabel("Average Degree")
    axes[2, 1].set_ylabel("Frequency")

    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    plt.savefig("time-slice-stats.png")

if __name__ == "__main__":
    df_results = analyze_slices()
    plot_results(df_results)