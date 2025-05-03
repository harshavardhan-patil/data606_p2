# Collaborative Filtering for Book Rating Prediction

This project implements a collaborative filtering algorithm to predict book ratings based on user preferences.

## Algorithm Description

### Implementation Overview

This implementation uses **Item-Item Collaborative Filtering** enhanced with dimension reduction techniques for improved efficiency and scalability. The key components of the algorithm are:

1. **Data Preprocessing**:
   - Cleaning the dataset by removing missing ratings (ratings with value 0)
   - Ensuring ratings are within the valid range (1-10)
   - Splitting data into training (75%) and testing (25%) sets, ensuring users in the test set have at least one rating in the training set

2. **Matrix Construction**:
   - Converting user-item ratings into sparse matrices for efficient computation
   - Using LIL (List of Lists) matrix format for construction and CSR (Compressed Sparse Row) for operations

3. **Dimension Reduction with TruncatedSVD**:
   - Applying Singular Value Decomposition to reduce the dimensionality of the item feature space
   - Using 20 latent features, balancing between accuracy and computational efficiency

4. **Item Similarity Computation**:
   - Using scikit-learn's NearestNeighbors with a ball-tree algorithm and Euclidean distance metric
   - Finding similar books in the reduced latent feature space

5. **Rating Prediction**:
   - Predicting a user's rating for a book by averaging the user's ratings for similar books
   - Optimized by caching user ratings to avoid repeated matrix lookups

6. **Evaluation**:
   - Measuring model performance using Mean Absolute Difference (MAD) between predicted and actual ratings
   - Evaluating different neighborhood sizes (k values) and sampling ratios

### Design Choices

- **SVD for Dimension Reduction**: Using TruncatedSVD significantly speeds up similarity calculations compared to working with the full sparse matrix
- **Ball-tree Algorithm**: Chosen for its efficiency with high-dimensional data in Euclidean space
- **Parallel Processing**: Utilizing all available CPU cores (`n_jobs=-1`) for faster neighbor computations
- **Rating Cache**: Implementing a user ratings cache to reduce redundant computations
- **Sample-based Evaluation**: Using a subset of 5,000 test ratings for faster evaluation without significant loss in accuracy assessment

## Running the Code

1. Install the required dependencies
```bash
pip install -r requirments.txt
```
2. Place the csv files in data/ folder
3. Run the evaluation.ipynb notebook

## Results

### Effect of Neighborhood Size (k)

| k Value | Mean Absolute Difference (MAD) |
|---------|--------------------------------|
| 5       | 1.2424                         |
| 10      | 1.2437                         |
| 15      | 1.2428                         |
| 20      | 1.2639                         |
| 50      | 1.2483                         |
| 100     | 1.2612                         |

This pattern suggests that very small neighborhood sizes (k=5 and k=15) capture the most relevant similar books without introducing noise from less similar items. The slight performance degradation with larger k values indicates that including too many neighbors might introduce irrelevant books into the prediction calculation, adding noise rather than signal.
The results demonstrate that simply increasing the number of neighbors doesn't guarantee better predictions, and there's a trade-off between having enough information (more neighbors) and maintaining relevance (fewer but more similar neighbors).

### Effect of Sample Ratio

| Sample Ratio | Mean Absolute Difference (MAD) |
|-------------|--------------------------------|
| 0.60        | 1.0161                         |
| 0.65        | 0.8869                         |
| 0.70        | 1.1054                         |
| 0.75        | 1.2424                         |
| 0.80        | 0.9189                         |
| 0.85        | 1.0280                         |
| 0.90        | 1.1519                         |

There is a non-monotonic relationship between sample ratio and performance. This suggests that the model performance is sensitive to the specific distribution of ratings in the training and test sets.
Simply increasing the training set size doesn't always lead to better performance - the quality and distribution of the training data matters more than quantity
The fluctuations in MAD across different sample ratios highlight the importance of carefully selecting the training-test split in collaborative filtering systems and potentially implementing cross-validation strategies to ensure robust performance evaluation.

## Future Improvements

- Implement user-user collaborative filtering for comparison
- Explore hybrid approaches combining content-based and collaborative filtering
- Try Approximate Nearest Neighbor search for speedup.
- Add more sophisticated prediction aggregation methods beyond simple averaging
