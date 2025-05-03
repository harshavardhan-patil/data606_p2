from preprocessing import clean_data, split_data
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix, lil_matrix
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import train_test_split
from pathlib import Path
from tqdm import tqdm
from sklearn.decomposition import TruncatedSVD


def run_cf_with_svd(ratings_df, test_size = 0.25):
    print(f"Original data size: {len(ratings_df)}")
    cleaned_ratings = clean_data(ratings_df)
    user_ids = cleaned_ratings['User-ID'].unique()
    book_ids = cleaned_ratings['ISBN'].unique()
    print(f"Cleaned data size: {len(cleaned_ratings)}")

    # Split the data
    train_df, test_df  = split_data(cleaned_ratings, test_size)
    print(f"Training set size: {len(train_df)}")
    print(f"Testing set size: {len(test_df)}")


    user_to_index = {user_id: idx for idx, user_id in enumerate(user_ids)}
    book_to_index = {book_id: idx for idx, book_id in enumerate(book_ids)}

    # Create sparse matrices for train and test sets
    n_users = len(user_ids)
    n_books = len(book_ids)

    # Initialize sparse matrices
    train_matrix = lil_matrix((n_users, n_books), dtype=np.float32)
    test_matrix = lil_matrix((n_users, n_books), dtype=np.float32)

    # Fill train matrix
    for _, row in tqdm(train_df.iterrows()):
        user_idx = user_to_index.get(row['User-ID'])
        book_idx = book_to_index.get(row['ISBN'])
        if user_idx is not None and book_idx is not None:
            train_matrix[user_idx, book_idx] = row['Book-Rating']

    # Fill test matrix
    for _, row in tqdm(test_df.iterrows()):
        user_idx = user_to_index.get(row['User-ID'])
        book_idx = book_to_index.get(row['ISBN'])
        if user_idx is not None and book_idx is not None:
            test_matrix[user_idx, book_idx] = row['Book-Rating']

    # Convert to CSR format for efficient operations
    train_matrix_csr = train_matrix.tocsr()
    test_matrix_csr = test_matrix.tocsr()

    return train_matrix_csr, test_matrix, test_df, user_to_index, book_to_index

def evaluate_model(train_matrix_csr, test_matrix, test_df, user_to_index, book_to_index, k_neighbors, n_components=20):
    """
    Fast item-item collaborative filtering with TruncatedSVD
    """
    # Use fewer components for speed
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    item_features = svd.fit_transform(train_matrix_csr.T)
    
    # Sample test set for faster evaluation
    sample_size = min(5000, len(test_df))
    sample_test_df = test_df.sample(n=sample_size, random_state=42)
    
    # Build a fast KNN model
    model = NearestNeighbors(n_neighbors=k_neighbors, algorithm='ball_tree', 
                        metric='euclidean', n_jobs=-1)  # Use all CPU cores
    model.fit(item_features)
    
    # Get only relevant test indices to avoid processing zeros
    test_indices = []
    for _, row in tqdm(sample_test_df.iterrows()):
        user_idx = user_to_index.get(row['User-ID'])
        book_idx = book_to_index.get(row['ISBN'])
        if user_idx is not None and book_idx is not None and test_matrix[user_idx, book_idx] > 0:
            test_indices.append((user_idx, book_idx))
    
    # Extract all user ratings at once to avoid repeated matrix lookups
    user_ratings_cache = {}
    
    predictions = []
    actuals = []
    
    for user_idx, book_idx in tqdm(test_indices):
        actual_rating = test_matrix[user_idx, book_idx]
        
        # Cache user ratings to avoid repeated lookups
        if user_idx not in user_ratings_cache:
            user_ratings_cache[user_idx] = train_matrix_csr[user_idx].toarray().flatten()
        
        # Get similar items
        dists, indices = model.kneighbors(item_features[book_idx].reshape(1, -1))
        similar_items = indices.flatten()
        
        # Just take the average of k most similar items
        rated_items = []
        for item in similar_items:
            if user_ratings_cache[user_idx][item] > 0:
                rated_items.append(user_ratings_cache[user_idx][item])
            if len(rated_items) >= 3:  # Only use top 3 for speed
                break
        
        if rated_items:
            # Simple average - much faster than weighted
            pred = sum(rated_items) / len(rated_items)
            predictions.append(pred)
            actuals.append(actual_rating)
    
    if not predictions:
        return float('inf')
    
    # Calculate MAD
    mad = np.mean(np.abs(np.array(predictions) - np.array(actuals)))
    return mad