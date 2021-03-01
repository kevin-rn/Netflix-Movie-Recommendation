import numpy as np
import pandas as pd
from random import randint

# -*- coding: utf-8 -*-
""""
FRAMEWORK FOR DATAMINING CLASS

#### IDENTIFICATION
NAME: Hugo
SURNAME: Koat
STUDENT ID: 4665945
KAGGLE ID: hugokoat

NAME: Kevin
SURNAME: Nanhekhan
STUDENT ID: 4959094
KAGGLE ID: kevinrn

### NOTES
This file is an example of what your code should look like. It is written in Python 3.6.
To know more about the expectations, please refer to the guidelines.
"""

#####
##
## DATA IMPORT
##
#####

#Where data is located
movies_file = './data/movies.csv'
users_file = './data/users.csv'
ratings_file = './data/ratings.csv'
predictions_file = './data/predictions.csv'
submission_file = 'data/submission.csv'


# Read the data using pandas
movies_description = pd.read_csv(movies_file, delimiter=';', dtype={'movieID':'int', 'year':'int', 'movie':'str'}, names=['movieID', 'year', 'movie'])
users_description = pd.read_csv(users_file, delimiter=';', dtype={'userID':'int', 'gender':'str', 'age':'int', 'profession':'int'}, names=['userID', 'gender', 'age', 'profession'])
ratings_description = pd.read_csv(ratings_file, delimiter=';', dtype={'userID':'int', 'movieID':'int', 'rating':'int'}, names=['userID', 'movieID', 'rating'])
predictions_description = pd.read_csv(predictions_file, delimiter=';', names=['userID', 'movieID'], header=None)

#####
##
## COLLABORATIVE FILTERING: User-User approach
##
#####
def predict_collaborative_filtering(ratings, predictions, k = 25):
    """
    Calculates the ratings for user/movie index listed in predictions through User-User Collaborative Filtering with K-Nearest Neighbour.
    :param ratings Movies x Users Matrix containing all ratings (previous ratings and NaN)
    :param predictions List of tuples (User Id, Movie Id) indicating what ratings should be predicted.
    :param k Number of neighbours to use values from.
    :return New Movies x User Matrix containing the previous ratings and the predicted ratings
    """
    # Calculate similarity (pearson correlation) scores of users.
    utility_matrix = ratings.transpose().corr()
    utility_matrix.fillna(0)

    # Iterate over all values that need to be predicted.
    for row in predictions.to_numpy():
        # Get indices of userID en movieID to predict rating for.
        user_index = row[0]
        movie_index = row[1]

        # Retrieve the similarity weights as vector and also the ratings of the users for a particular movie.
        similarity_weights = utility_matrix[user_index]
        rating_users = ratings[movie_index]

        # Only keep weights where a rating exists (also removes the current rating to predict for).
        similarity_weights = similarity_weights[rating_users > 0].to_numpy()
        rating_users = rating_users[rating_users > 0].to_numpy()

        # Sorts the arrays on similarity scores and takes the last k values (highest k similarities)
        sort_indices = similarity_weights.argsort()
        sorted_similarity_weights = similarity_weights[sort_indices]
        sorted_rating_users = rating_users[sort_indices]
        sorted_similarity_weights = sorted_similarity_weights[-k:]
        sorted_rating_users = sorted_rating_users[-k:]

        # Check if sum of weights isn't 0 to avoid division by 0 and calculate the weighted average or leave it as 0.
        total_weight = sum(sorted_similarity_weights)
        if total_weight != 0:
            weighted_average = sum(sorted_similarity_weights * sorted_rating_users)/total_weight
            if weighted_average <= 1:
                ratings[movie_index][user_index] = 1
            elif weighted_average >= 5:
                ratings[movie_index][user_index] = 5
            elif 1 < weighted_average < 5:
                ratings[movie_index][user_index] = weighted_average
            else:
                ratings[movie_index][user_index] = 0

    return ratings



#####
##
## LATENT FACTORS
##
#####

def predict_latent_factors(ratings, k=25):
    """
    Calculates the ratings for user/movie index listed in predictions through Matrix Factorization with SVD.
    :param ratings Movies x Users Matrix containing all ratings (previous ratings and NaN)
    :param k Number of features to use for the dimensionality reduction.
    :return New Movies x User Matrix containing the previous ratings and the predicted ratings
    """
    # Replace non-rated values with user average (row mean)
    ratings[ratings == 0] = np.nan
    mean_rating = np.nanmean(ratings, axis=1).reshape(-1, 1)
    rating_matrix = ratings - mean_rating
    rating_matrix.fillna(value=0, inplace=True)

    # Calculate prediction matrix
    U, sigma, Vt = np.linalg.svd(rating_matrix, full_matrices=False)
    sigma = np.diag(sigma)

    # Dimensionality reduction: Use only first k features instead of all
    sigma = sigma[0:k, 0:k]
    U = U[:, 0:k]
    Vt = Vt[0:k, :]

    # Calculate the rating matrix
    R = np.dot(U, np.dot(sigma, Vt)) + mean_rating
    return R

#####
##
## FINAL PREDICTORS
##
#####

def predict_final(movies, users, ratings, predictions):
    """
    Calculates the ratings for user/movie index listed in predictions through a combination of different Collaborative Filtering algorithms.
    :param movies Dataframe containing the columns Movie ID, Year and Movie
    :param users Dataframe containing the columns User Id, Gender, Age, Profession
    :param ratings Dataframe containing the columns User Id, Movie Id and rating
    :return An enumerated list containing the predicted ratings for predictions.
    """
    # Creates an user/movie matrix containing all of the ratings and add missing rows and columns containing NaN values.
    rating_matrix = pd.pivot_table(ratings, index='userID', columns='movieID', values='rating')
    rating_matrix = rating_matrix.reindex(users['userID'].values, axis=0)
    rating_matrix = rating_matrix.reindex(movies['movieID'].values, axis=1).fillna(0)

    # Calls method to perform collaborative filtering which returns a new rating matrix (dataframe)
    rating_matrix = predict_collaborative_filtering(ratings=rating_matrix, predictions=predictions, k=25)
    print('-Finished executing cf nearest neighbour-')

    # Calls method to perform Matrix Factorization latent factor which returns a new rating matrix (numpy array)
    rating_matrix = predict_latent_factors(ratings=rating_matrix, k=25)
    print('-Finished executing cf matrix factorization-')

    # Enumerates the predicted ratings starting from 1 and returns a list of tuples in the format: (index, predicted rating)
    # In case latent factors is commented out and only collaborative filtering is run, add .to_numpy() to rating_matrix.
    final_rating = rating_matrix
    prediction_ratings = []
    for row in predictions.to_numpy():
        prediction_ratings.append(final_rating[row[0]-1][row[1]-1])
    return list(enumerate(prediction_ratings, 1))

#####
##
## RANDOM PREDICTORS
##
#####

#By default, predicted rate is a random classifier
def predict_random(movies, users, ratings, predictions):
    number_predictions = len(predictions)
    return [[idx, randint(1, 5)] for idx in range(1, number_predictions + 1)]

#####
##
## SAVE RESULTS
##
#####    

## //!!\\ TO CHANGE by your prediction function
predictions = predict_final(movies_description, users_description, ratings_description, predictions_description)

#Save predictions, should be in the form 'list of tuples' or 'list of lists'
with open(submission_file, 'w') as submission_writer:

    #Formates data
    predictions = [map(str, row) for row in predictions]
    predictions = [','.join(row) for row in predictions]
    predictions = 'Id,Rating\n'+'\n'.join(predictions)

    #Writes it down
    submission_writer.write(predictions)