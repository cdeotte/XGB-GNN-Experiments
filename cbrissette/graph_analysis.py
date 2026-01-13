import os
import pandas as pd
import numpy as np
from pathlib import Path
from relbench.datasets import get_dataset

# Configuration
DATASET_NAME = "rel-arxiv"
OUTPUT_DIR = "arxiv_author_graphs"
LOOK_BACK_WINDOW_MONTHS = 12  # Window for defining the Graph Topology (Edges)
TIME_DELTA_MONTHS = 1         # Step size for sliding
DAYS_IN_MONTH = 30

# Feature Lookback Windows (from Notebook)
YEAR_1 = 365
YEAR_3 = 365 * 3

Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

def find_col_by_keyword(df, keyword):
    candidates = [c for c in df.columns if keyword.lower() in c.lower()]
    id_candidates = [c for c in candidates if 'id' in c.lower()]
    if id_candidates: return id_candidates[0]
    elif candidates: return candidates[0]
    return None

def load_data():
    print(f"Loading {DATASET_NAME}...")
    dataset = get_dataset(name=DATASET_NAME)
    db = dataset.make_db() 
    
    tables = db.table_dict
    print(f"Found tables: {list(tables.keys())}")
    
    # 1. Papers
    if 'papers' in tables: paper_df = tables['papers'].df
    elif 'paper' in tables: paper_df = tables['paper'].df
    else: raise KeyError("Missing 'papers' table.")

    # 2. Author Links
    if 'paperAuthors' in tables: link_df = tables['paperAuthors'].df
    elif 'writes' in tables: link_df = tables['writes'].df
    else: raise KeyError("Missing 'paperAuthors' table.")

    # 3. Citations (New)
    if 'citations' in tables: cit_df = tables['citations'].df
    else: cit_df = pd.DataFrame() # Fallback empty if missing

    # 4. Categories (New)
    if 'paperCategories' in tables: cat_df = tables['paperCategories'].df
    else: cat_df = pd.DataFrame()

    return paper_df, link_df, cit_df, cat_df

def entropy(counts):
    """Calculates entropy of a distribution (from notebook)."""
    if len(counts) == 0:
        return 0.0
    p = np.array(counts, dtype=float)
    p = p / p.sum()
    return -(p * np.log(p + 1e-12)).sum()

def get_node_features(active_nodes_df, link_df, paper_df, cit_df, cat_df, current_date, paper_pk, time_col):
    """
    Generates rich node features for the active authors based on history up to current_date.
    """
    author_col = find_col_by_keyword(link_df, 'author')
    paper_col_link = find_col_by_keyword(link_df, 'paper')
    
    # Filter history: Everything BEFORE current_date
    mask_history = paper_df[time_col] < current_date
    paper_hist = paper_df[mask_history].copy()
    
    # Filter links to historical papers
    link_hist = link_df[link_df[paper_col_link].isin(paper_hist[paper_pk])].copy()
    
    # --- FIX START: Handle duplicate date columns ---
    # If the link table already has the time column (e.g. Submission_Date), 
    # drop it to prevent pandas from creating 'Submission_Date_x' and '_y' during merge.
    if time_col in link_hist.columns:
        link_hist = link_hist.drop(columns=[time_col])
    # --- FIX END ---

    # Join dates to links for time-based calcs
    link_hist = pd.merge(link_hist, paper_hist[[paper_pk, time_col]], 
                         left_on=paper_col_link, right_on=paper_pk, how='inner')

    # Filter to only relevant authors to speed up groupby
    relevant_authors = active_nodes_df['id'].unique()
    link_hist_filtered = link_hist[link_hist[author_col].isin(relevant_authors)]

    # --- 1. Productivity Features ---
    # Group by Author
    auth_grp = link_hist_filtered.groupby(author_col)
    
    # Total Papers
    feats = auth_grp.size().rename("total_papers").to_frame()
    
    # Recency (Days since last paper)
    # This line was crashing before because time_col was renamed
    last_date = auth_grp[time_col].max()
    
    feats["days_since_last_paper"] = (current_date - last_date).dt.days
    feats["days_since_last_paper"] = feats["days_since_last_paper"].fillna(10000)

    # Papers last 1Y and 3Y
    date_1y = current_date - pd.Timedelta(days=YEAR_1)
    date_3y = current_date - pd.Timedelta(days=YEAR_3)
    
    # Last 1 Year
    mask_1y = link_hist_filtered[time_col] >= date_1y
    count_1y = link_hist_filtered[mask_1y].groupby(author_col).size().rename("papers_last_1y")
    feats = feats.join(count_1y).fillna(0)
    
    # Last 3 Years
    mask_3y = link_hist_filtered[time_col] >= date_3y
    count_3y = link_hist_filtered[mask_3y].groupby(author_col).size().rename("papers_last_3y")
    feats = feats.join(count_3y).fillna(0)

    # --- 2. Topic Features ---
    if not cat_df.empty:
        paper_cat_col = find_col_by_keyword(cat_df, 'paper')
        cat_id_col = find_col_by_keyword(cat_df, 'category')
        
        # Merge Categories onto the author-paper links
        auth_paper_cat = pd.merge(link_hist_filtered[[author_col, paper_col_link]], 
                                  cat_df[[paper_cat_col, cat_id_col]],
                                  left_on=paper_col_link, right_on=paper_cat_col)
        
        cat_counts = auth_paper_cat.groupby([author_col, cat_id_col]).size().reset_index(name='n')
        
        # Top Category
        cat_counts = cat_counts.sort_values([author_col, 'n'], ascending=[True, False])
        top_cat = cat_counts.groupby(author_col).first()
        
        # Fraction
        total_cat_recs = auth_paper_cat.groupby(author_col).size()
        
        # Entropy
        cat_ent = cat_counts.groupby(author_col)['n'].apply(entropy).rename("category_entropy")

        topic_feats = pd.DataFrame(index=feats.index)
        topic_feats['top_category'] = top_cat[cat_id_col]
        topic_feats['top_category_frac'] = top_cat['n'] / total_cat_recs
        topic_feats = topic_feats.join(cat_ent)
        
        feats = feats.join(topic_feats)
        feats['top_category'] = feats['top_category'].fillna(-1)
        feats['top_category_frac'] = feats['top_category_frac'].fillna(0)
        feats['category_entropy'] = feats['category_entropy'].fillna(0)

    # --- 3. Citation Features ---
    if not cit_df.empty:
        ref_col = 'References_Paper_ID' if 'References_Paper_ID' in cit_df.columns else cit_df.columns[1]
        
        if time_col in cit_df.columns:
            cit_hist = cit_df[cit_df[time_col] < current_date]
        else:
            cit_hist = cit_df
            
        paper_cite_counts = cit_hist[ref_col].value_counts().rename("n_cites")
        
        link_w_cites = link_hist_filtered.copy()
        link_w_cites['cites'] = link_w_cites[paper_col_link].map(paper_cite_counts).fillna(0)
        
        cite_grp = link_w_cites.groupby(author_col)['cites']
        
        cite_feats = pd.DataFrame({
            "total_citations": cite_grp.sum(),
            "avg_citations_per_paper": cite_grp.mean(),
            "max_citations_single_paper": cite_grp.max()
        })
        
        recent_paper_ids = paper_hist.loc[paper_hist[time_col] >= date_3y, paper_pk]
        
        recent_links = link_w_cites[link_w_cites[paper_col_link].isin(recent_paper_ids)]
        if not recent_links.empty:
            cite_3y = recent_links.groupby(author_col)['cites'].sum().rename("citations_last_3y")
            cite_feats = cite_feats.join(cite_3y).fillna(0)
        else:
            cite_feats["citations_last_3y"] = 0
            
        feats = feats.join(cite_feats).fillna(0)

    # Final Cleanup
    feats = feats.reset_index().rename(columns={author_col: 'id'})
    result = pd.merge(active_nodes_df[['id']], feats, on='id', how='left')
    result = result.fillna(0)
    
    if 'days_since_last_paper' in result.columns:
        result.loc[result['days_since_last_paper'] == 0, 'days_since_last_paper'] = 10000
        
    return result

def build_coauthorship_edges(paper_ids_in_window, link_df):
    paper_col = find_col_by_keyword(link_df, 'paper')
    author_col = find_col_by_keyword(link_df, 'author')
    
    relevant_links = link_df[link_df[paper_col].isin(paper_ids_in_window)].copy()
    if relevant_links.empty:
        return pd.DataFrame(columns=['source_id', 'target_id']), pd.DataFrame(columns=['id'])

    # Clique expansion
    merged = pd.merge(relevant_links, relevant_links, on=paper_col, suffixes=('_1', '_2'))
    edges = merged[merged[author_col + '_1'] != merged[author_col + '_2']].copy()
    edges = edges[[author_col + '_1', author_col + '_2']]
    edges.columns = ['source_id', 'target_id']
    edges.drop_duplicates(inplace=True)
    
    active_authors = relevant_links[author_col].unique()
    nodes_df = pd.DataFrame({'id': active_authors})
    return edges, nodes_df

def calculate_targets(nodes_df, link_df, paper_df, start, end, time_col, paper_pk):
    # Future Productivity
    paper_col = find_col_by_keyword(link_df, 'paper')
    author_col = find_col_by_keyword(link_df, 'author')
    
    mask = (paper_df[time_col] > start) & (paper_df[time_col] <= end)
    fut_papers = paper_df[mask][paper_pk]
    
    fut_links = link_df[link_df[paper_col].isin(fut_papers)]
    counts = fut_links[author_col].value_counts().reset_index()
    counts.columns = ['id', 'paper_count']
    
    targets = pd.merge(nodes_df[['id']], counts, on='id', how='left').fillna(0)
    return targets

def main():
    paper_df, link_df, cit_df, cat_df = load_data()
    
    # Time setup
    time_col = 'Submission_Date' if 'Submission_Date' in paper_df.columns else \
               next(c for c in paper_df.columns if 'date' in c.lower())
    paper_df[time_col] = pd.to_datetime(paper_df[time_col])
    
    # Fix Citations time column if missing (assume citation date = paper date as proxy if table lacks date)
    if not cit_df.empty and time_col not in cit_df.columns:
        # If citations don't have dates, we can't do temporal filtering on them easily.
        # We might join with paper_df to get the date of the CITING paper if possible.
        # For this script, we'll skip complex citation joining and just use what's there
        pass

    min_time = paper_df[time_col].min()
    max_time = paper_df[time_col].max()
    
    window_delta = pd.Timedelta(days=LOOK_BACK_WINDOW_MONTHS * DAYS_IN_MONTH)
    step_delta = pd.Timedelta(days=TIME_DELTA_MONTHS * DAYS_IN_MONTH)
    
    k = 0
    current_end = min_time + window_delta
    paper_pk = find_col_by_keyword(paper_df, 'id') or 'id'

    print(f"Processing range: {min_time.date()} to {max_time.date()}")

    while current_end <= max_time:
        slice_start = current_end - window_delta
        slice_end = current_end
        target_end = current_end + step_delta
        
        print(f"Slice {k}: {slice_end.date()}")
        
        # 1. Active Papers (Graph Window)
        mask = (paper_df[time_col] >= slice_start) & (paper_df[time_col] <= slice_end)
        active_papers = paper_df.loc[mask, paper_pk]
        
        if active_papers.empty:
            print(f"  Skipping (no papers)")
        else:
            # 2. Build Graph Topology
            edges_df, nodes_df = build_coauthorship_edges(active_papers, link_df)
            
            if nodes_df.empty:
                print(f"  Skipping (no nodes)")
            else:
                # 3. Add Features (History up to slice_end)
                nodes_with_feats = get_node_features(
                    nodes_df, link_df, paper_df, cit_df, cat_df, 
                    current_date=slice_end, 
                    paper_pk=paper_pk, 
                    time_col=time_col
                )
                
                # 4. Targets
                targets_df = calculate_targets(
                    nodes_df, link_df, paper_df, 
                    start=slice_end, end=target_end, 
                    time_col=time_col, paper_pk=paper_pk
                )
                
                # 5. Save
                nodes_with_feats.to_parquet(os.path.join(OUTPUT_DIR, f"nodes_{k}.pqt"), index=False)
                edges_df.to_parquet(os.path.join(OUTPUT_DIR, f"edges_{k}.pqt"), index=False)
                targets_df.to_parquet(os.path.join(OUTPUT_DIR, f"node_targets_{k}.pqt"), index=False)
                
                print(f"  > Nodes: {len(nodes_df)} | Features: {nodes_with_feats.shape[1]}")

        k += 1
        current_end += step_delta

if __name__ == "__main__":
    main()