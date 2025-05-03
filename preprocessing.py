import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm

def clean_data(ratings_df):
    """
    Clean the ratings dataframe by removing missing ratings (Book-Rating = 0)
    and handling any anomalies.
    """
    # Remove ratings with value 0 (missing ratings)
    cleaned_ratings_df = ratings_df[ratings_df['Book-Rating'] > 0].copy()
    
    # Check for and handle any rating values outside the valid range (1-10)
    cleaned_ratings_df = cleaned_ratings_df[
        (cleaned_ratings_df['Book-Rating'] >= 1) & 
        (cleaned_ratings_df['Book-Rating'] <= 10)
    ]
    
    return cleaned_ratings_df

def split_data(ratings_df, test_size=0.25, random_state=42):
    """
    Split the ratings data into training and testing sets based on (User-ID, ISBN) pairs.
    Ensures users in the test set have ratings in the training set.
    """
    train_df, test_df = train_test_split(
        ratings_df, 
        test_size=test_size, 
        random_state=random_state
    )
    
    # Get unique users in the test set
    test_users = test_df['User-ID'].unique()
    
    # Check if all test users have at least one rating in the training set
    users_not_in_train = set(test_users) - set(train_df['User-ID'].unique())
    
    ratings_to_move = []
    # If there are users in test not in train, move one rating for each to train
    if users_not_in_train:
        for user in tqdm(users_not_in_train):
            # Find all ratings for this user in the test set
            user_ratings = test_df[test_df['User-ID'] == user]
            
            if not user_ratings.empty:
                # Move one rating to train set
                rating_to_move = user_ratings.iloc[0:1]
                ratings_to_move.append(rating_to_move)

        ratings_to_move_df = pd.concat(ratings_to_move, ignore_index=True)
        ratings_to_move = pd.Series(ratings_to_move)
        test_df = test_df.drop([r.index[0] for r in ratings_to_move])
        train_df = pd.concat([train_df, ratings_to_move_df])
    
    return train_df, test_df