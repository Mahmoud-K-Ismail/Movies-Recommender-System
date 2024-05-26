import pandas as pd
import numpy as np
from scipy.stats import pearsonr


reviews = pd.read_csv('non_personalised_stereotyped_rec.csv')
print(f"#rows     (users): {reviews.shape[0]}")
print(f"#columns (movies): {reviews.shape[1]}")
reviews.head(20)

means = reviews.iloc[:,2:].apply(np.mean).sort_values(ascending=False)
print(f"Best rating: {means.head(1)}")

top_column = reviews.iloc[:, 2:].count().idxmax()
print(f"Best Popularity: {top_column}")

top_column = reviews.iloc[:, 2:][reviews.iloc[:, 2:] >= 4].count().idxmax()
print(f"Best Good popularity: {top_column}")

def return_best_n(statistics, n):
    statistics = pd.DataFrame({'statistic':statistics})
    return statistics.sort_values('statistic', ascending = False).iloc[:n]

def coocurrence_with_toystory(col, toystory_ratings):
    x = np.sum((~np.isnan(col)) & (~np.isnan(toystory_ratings)))/np.sum(~np.isnan(toystory_ratings))
    return x

toystory_col = reviews['1: Toy Story (1995)']

coocurence_toystory = reviews.iloc[:,2:].apply(lambda col : coocurrence_with_toystory(col, toystory_col))
return_best_n(coocurence_toystory,4)[1:4]

def pearson(col, toystory):
    valid_rows = np.logical_and(~np.isnan(col),~np.isnan(toystory))
    return pearsonr(col[valid_rows], toystory[valid_rows])[0]

pearson_corr = reviews.iloc[:,2:].apply(lambda col : pearson(col, toystory_col))
return_best_n(pearson_corr,4)[1:4]

reviews = pd.read_csv('content_based_filtering.csv')
docs_topics = reviews.iloc[:20,:11]
docs_topics.index = docs_topics.iloc[:,0]
docs_topics.drop('Unnamed: 0', axis = 1, inplace=True)
print(f"#Rows/Users: {reviews.shape[0]} \n#Columns/Movies: {reviews.shape[1]}")
docs_topics.head()

user_reviews = reviews.iloc[:20,[0,14,15]]
user_reviews.index = user_reviews.iloc[:,0]
user_reviews.drop('Unnamed: 0', axis = 1, inplace=True)
user_reviews.head()

def get_taste_vector(user_col, docs_topics):
    return docs_topics.apply(lambda doc_col : np.dot(user_col, doc_col))

user_tastes = user_reviews.apply(lambda col : get_taste_vector(col, docs_topics))
user_tastes

def get_doc_scores(user_taste, docs_topics):
    return docs_topics.apply(lambda doc_topic: np.dot(user_taste, doc_topic), axis = 1)

doc_scores = user_tastes.apply(lambda user_taste: get_doc_scores(user_taste, docs_topics))
doc_scores.head()

print('Predicted prefered docs for User 1 by content')
doc_sorted = doc_scores["User 1"].sort_values(ascending = False)
print (*((doc_sorted.head(3).index).tolist()))

print('Predicted prefered docs for User 2 by content')
doc_sorted = doc_scores["User 2"].sort_values(ascending = False)
print (*((doc_sorted.head(3).index).tolist()))

norm_docs_topics = docs_topics.apply(lambda doc: doc / np.linalg.norm(doc), axis=1)
norm_docs_topics.head(10)

norm_user_tastes = user_reviews.apply(lambda col : get_taste_vector(col, norm_docs_topics)/np.linalg.norm(col))
print("Normalized users' taste vectors:")
print(norm_user_tastes)

'''print('Predicted prefered docs for User 1 by content')
doc_sorted = norm_doc_scores["User 1"].sort_values(ascending = False)
print (*((doc_sorted.head(3).index).tolist()))

print('Predicted prefered docs for User 2 by content')
doc_sorted = norm_doc_scores["User 2"].sort_values(ascending = False)
print (*((doc_sorted.head(3).index).tolist()))'''

norm_doc_scores = norm_user_tastes.apply(lambda norm_user_tastes: get_doc_scores(norm_user_tastes, norm_docs_topics))

print('Predicted prefered docs for User 1 by content')
doc_sorted = norm_doc_scores["User 1"].sort_values(ascending = False)
print ((doc_sorted.head(3)))

print('Predicted prefered docs for User 2 by content')
doc_sorted = norm_doc_scores["User 2"].sort_values(ascending = False)
print (doc_sorted.head(3))



IDF = norm_docs_topics.apply(lambda col: 1/np.sum(col != 0))
IDF

def TF_IDF_scores(norm_docs_topics, IDF, user_taste_vector):
    return norm_docs_topics.apply(lambda row: np.dot(row, user_taste_vector * IDF), axis=1)
tfidf_scores = norm_user_tastes.apply(lambda user_vector: TF_IDF_scores(norm_docs_topics, IDF, user_vector))
tfidf_scores.head()


print('Predicted prefered docs for User 1 by content')
doc_sorted = tfidf_scores["User 1"].sort_values(ascending = False)
print (*((doc_sorted.head(3).index).tolist()))

print('Predicted prefered docs for User 2 by content')
doc_sorted = tfidf_scores["User 2"].sort_values(ascending = False)
print (*((doc_sorted.head(3).index).tolist()))

print('Final Comparison for User 1')
m1 = doc_scores.sort_values('User 1', ascending=False)['User 1'].head(10)
m2 = norm_doc_scores.sort_values('User 1', ascending=False)['User 1'].head(10)
m3 = tfidf_scores.sort_values('User 1', ascending=False)['User 1'].head(10)
final_DF_user1 = pd.DataFrame({'Rank':m1.index, 'Score':m1.values,
                            'Normalized rank':m2.index, 'Normalized score':m2.values,
                            'IDF rank':m3.index, 'IDF score':m3.values})
final_DF_user1 = final_DF_user1[['Rank','Score',
                            'Normalized rank', 'Normalized score',
                            'IDF rank', 'IDF score']]
final_DF_user1

print('Final Comparison for User 2')
m1 = doc_scores.sort_values('User 2', ascending=False)['User 2'].head(10)
m2 = norm_doc_scores.sort_values('User 2', ascending=False)['User 2'].head(10)
m3 = tfidf_scores.sort_values('User 2', ascending=False)['User 2'].head(10)
final_DF_user2 = pd.DataFrame({'Rank':m1.index, 'Score':m1.values,
                            'Normalized rank':m2.index, 'Normalized score':m2.values,
                            'IDF rank':m3.index, 'IDF score':m3.values})
final_DF_user2 = final_DF_user2[['Rank','Score',
                            'Normalized rank', 'Normalized score',
                            'IDF rank', 'IDF score']]
final_DF_user2

df = pd.read_csv('User-User Collaborative Filtering - movie-row.csv', dtype=object, index_col=0)

# rpreprocessing
def process_col(col):
    return col.astype(str).apply(lambda val: val.replace(',','.')).astype(float)
df = df.apply(process_col)

print(f"Dataset shape: {df.shape}")
df.head()

df_corr = df.corr(method = 'pearson')
df_corr.head()

def find_k_nearest_users(user_sim_col, k = 5):
    return user_sim_col[user_sim_col.index != user_sim_col.name].nlargest(n = k).index.tolist()

k_neighboors = df_corr.apply(lambda col: find_k_nearest_users(col))
k_neighboors

#TODO
# Function to predict the score of a user by averaging the scores of its neighbors
def predict_user_score(user, k_neighbours, ratings_df):
    neighbour_indices = k_neighbours[user]
    neighbour_scores = ratings_df[neighbour_indices]
    return neighbour_scores.mean(skipna=True,axis=1)

predicted_scores = {}

# Iterate over each user
for user in df.columns:
    predicted_scores[user] = predict_user_score(user, k_neighboors, df)

predicted_scores_df = pd.DataFrame(predicted_scores)

predicted_scores_df.head()

#TODO
def predict_user_score(user, k_neighbours, df, df_corr):
    neighbour_indices = k_neighbours[user]
    neighbour_scores = df[neighbour_indices]
    similarities = df_corr.loc[user, neighbour_indices]

    # Ensure non-negative similarities
    similarities[similarities < 0] = 0

    # Calculate weighted scores
    weighted_scores = neighbour_scores.mul(similarities, axis=1)

    # Calculate weighted average
    weighted_average = weighted_scores.sum(axis=1) / similarities.sum()

    return weighted_average

predicted_scores = {}

# Iterate over each user
for user in df.columns:
    predicted_scores[user] = predict_user_score(user, k_neighboors, df, df_corr)

predicted_scores_df = pd.DataFrame(predicted_scores)
predicted_scores_df.head()

def cos_similarity(item1, item2):
    item1_values = ~np.isnan(item1)
    item2_values = ~np.isnan(item2)
    all_values = np.logical_and(item1_values,item2_values)
    return np.dot(item1[all_values], item2[all_values])/(np.linalg.norm(item1[item1_values]) * np.linalg.norm(item2[item2_values]))


k_neighbours_cosine = df_corr.apply(lambda col: find_k_nearest_users(col))
k_neighbours_cosine

def predict_user_score_cos(user, k_neighbours, df, df_corr):
    neighbour_indices = k_neighbours[user]
    neighbour_scores = df[neighbour_indices]
    similarities = neighbour_scores.apply(lambda neighbour: cos_similarity(df[user], neighbour), axis=0)

    # Ensure non-negative similarities
    similarities[similarities < 0] = 0

    # Calculate weighted scores
    weighted_scores = neighbour_scores.mul(similarities, axis=1)

    # Calculate weighted average
    weighted_average = weighted_scores.sum(axis=1) / similarities.sum()

    return weighted_average

predicted_scores_cos = {}

# Iterate over each user
for user in df.columns:
    predicted_scores_cos[user] = predict_user_score_cos(user, k_neighboors, df, df_corr)

predicted_scores_df_cos = pd.DataFrame(predicted_scores_cos)
predicted_scores_df_cos.head()


df = pd.read_csv('Item Item Collaborative Filtering - Ratings.csv', index_col=0, nrows=20)

# preprocessing
df.drop('Mean', axis=1, inplace=True)
def process_col(col):
    return col.astype(str).apply(lambda val: val.replace(',','.')).astype(float)
df = df.apply(process_col)

print(f"Dataset shape: {df.shape}")
df.head()

# Compute cosine similarity matrix using a single lambda function
df_cos = df.apply(lambda item1: df.apply(lambda item2: cos_similarity(item1, item2), axis=1), axis=1)

# Display the resulting DataFrame
df_cos.head()

def predict_rating(user_ratings, item_similarity):
    weighted_sum = 0
    sum_of_similarities = 0

    for item_index, rating in enumerate(user_ratings):
        # Check if the user has rated the item and the item similarity is valid
        if not np.isnan(rating) and not np.isnan(item_similarity.iloc[item_index]):
            # Compute the weighted sum and sum of similarities
            weighted_sum += rating * item_similarity.iloc[item_index]
            sum_of_similarities += item_similarity.iloc[item_index]

    if sum_of_similarities == 0:
        return np.nan
    else:
        return weighted_sum / sum_of_similarities


predictions = df.apply(lambda user_ratings: df_cos.apply(lambda item_similarity: predict_rating(user_ratings, item_similarity)), axis=1)
predictions.head()

# Compute cosine similarity matrix using a single lambda function
df_pearson = df.corr(method='pearson')

# Display the resulting DataFrame
df_pearson.head()

def predict_rating(user_ratings, item_similarity):
    weighted_sum = 0
    sum_of_similarities = 0

    for item_index, rating in enumerate(user_ratings):
        similarity = item_similarity.iloc[item_index]

        # Check if the similarity is valid and positive
        if not np.isnan(rating) and similarity > 0:
            # Compute the weighted sum and sum of positive similarities
            weighted_sum += rating * similarity
            sum_of_similarities += similarity

    # Avoid division by zero
    if sum_of_similarities == 0:
        return np.nan
    else:
        return weighted_sum / sum_of_similarities

predictions = df.apply(lambda user_ratings: df_pearson.apply(lambda item_similarity: predict_rating(user_ratings, item_similarity)), axis=1)
predictions.head()