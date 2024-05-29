#------------------------------------------------------------------------------------------------------------------------------------#
                                            # MADE BY: Mahmoud-K-Ismail
                                            #Dataset used from http://files.grouplens.org/datasets/movielens/ml-100k.zip
#------------------------------------------------------------------------------------------------------------------------------------#
                                            # General Desription of the Program 
#                                           -----------------------------------

# This program is a GUI for a movie recommender system that uses content-based recommendations.
# The user can select a movie they liked from a dropdown menu, and the program will recommend two movies based on co-occurrence with the selected movie and Pearson correlation with the selected movie.
# The program uses a dataset of user-item ratings to calculate the recommendations.
# The selected movie and the recommended movies are displayed in the GUI.
# The code uses the Tkinter library to create the GUI and the pandas library to handle the data.
# The necessary functions are defined in the code.
# The give_recommendations function is called when the user clicks the "Get Recommendations" button.
# The recommended movies are displayed in the GUI using the output_text variable.
# The program is run by calling the main loop of the Tkinter application.
# The recommendations are given based on different algorithms and approaches: co-occurrence based, pearson based and more, it shows the diffence
# -- in the recommendations that could happen based on the algorithm used.

#------------------------------------------------------------------------------------------------------------------------------------#

import pandas as pd
import numpy as np
from scipy.stats import pearsonr
import tkinter as tk
from tkinter import ttk

# Define necessary functions
def return_best_n(statistics, n):
    statistics = pd.DataFrame({'statistic': statistics})
    return statistics.sort_values('statistic', ascending=False).iloc[:n]


# Calculate co-occurrence of a movie column with a Movie ratings
def coocurrence_with_movie(col, toystory_ratings):
    # Calculate the number of non-NA values in both columns
    x = np.sum((~np.isnan(col)) & (~np.isnan(toystory_ratings))) 
    # Divide by the number of non-NA values in Movie ratings column
    x = x / np.sum(~np.isnan(toystory_ratings))
    return x

def pearson(col, toystory):
    valid_rows = np.logical_and(~np.isnan(col), ~np.isnan(toystory))
    if np.sum(valid_rows) < 2:
        return 0  # Not enough data points to compute correlation
    return pearsonr(col[valid_rows], toystory[valid_rows])[0]



# Load data
reviews = pd.read_csv('user_item_matrix_with_names.csv')

# Ensure consistency in column names
reviews.columns = reviews.columns.str.strip()

# Define recommendation function
def give_recommendations():
    """
    Recommend movies based on the selected movie.

    This function takes the selected movie from the user input, finds its ratings in the reviews data,
    and then recommends movies using both co-occurrence with the selected movie and Pearson correlation.
    The recommended movies are displayed in the GUI.

    Returns:
        None
    """
    # Get the selected movie from the user input
    selected_movie = movie_var.get().strip()
    print(f"Selected movie: {selected_movie}")  # Debugging output
    
    # Check if the selected movie is in the dataset
    if selected_movie not in reviews.columns:
        output_text.set("Selected movie not found in the dataset.")
        return

    # Get the ratings of the selected movie
    movie_col = reviews[selected_movie]

    # Calculate co-occurrence with the selected movie
    coocurence_movie = reviews.iloc[:,2:].apply(lambda col : coocurrence_with_movie(col, movie_col))
    # Get the top 3 recommended movies based on co-occurrence
    recommended_movies = return_best_n(coocurence_movie, 4)[1:4]

    # Calculate Pearson correlation with the selected movie
    pearson_corr = reviews.iloc[:,2:].apply(lambda col : pearson(col, movie_col))
    # Get the top 3 recommended movies based on Pearson correlation
    recommended_pearson = return_best_n(pearson_corr, 4)[1:4]
    
    print(f"Recommended Movies Using Pearson: {recommended_pearson.index}")  # Debugging output

    # Display the recommended movies in the GUI
    output_text.set("Recommended movies using Co-occurrence: "
                     + "\n ".join(recommended_movies.index)
                     + "\n\n" 
                     + "Recommended movies using Pearson: "
                     + "\n ".join(recommended_pearson.index))

# Initialize main window
root = tk.Tk()
root.title("Movies Recommender System GUI")

# Initialize variables
output_text = tk.StringVar()

# Create and pack widgets
tk.Label(root, text="Select a movie you liked:").pack(pady=5)
movie_var = tk.StringVar()
movie_dropdown = ttk.Combobox(root, textvariable=movie_var)
movie_dropdown['values'] = reviews.columns[2:].tolist()  # Assuming first two columns are not movie titles
movie_dropdown.pack(pady=5)

tk.Button(root, text="Get Recommendations", command=give_recommendations).pack(pady=10)
tk.Label(root, textvariable=output_text, wraplength=400, justify="left").pack(pady=5)

tk.Button(root, text="Quit", command=root.quit).pack(pady=10)

# Run the application
root.mainloop()



'''
#Functions that can be used if a content-based recommendations are needed


#TODO: make this suitable for Movie recommendation
def get_taste_vector(user_col, docs_topics):
    return docs_topics.apply(lambda doc_col: np.dot(user_col, doc_col))

def get_doc_scores(user_taste, docs_topics):
    return docs_topics.apply(lambda doc_topic: np.dot(user_taste, doc_topic), axis=1)

def TF_IDF_scores(norm_docs_topics, IDF, user_taste_vector):
    return norm_docs_topics.apply(lambda row: np.dot(row, user_taste_vector * IDF), axis=1)


'''