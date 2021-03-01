import numpy as np
import pandas as pd
from random import randint

# -*- coding: utf-8 -*-
"""
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
This files is an example of what your code should look like. 
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
movies_description = pd.read_csv(movies_file, delimiter=';', names=['movieID', 'year', 'movie'])
users_description = pd.read_csv(users_file, delimiter=';', names=['userID', 'gender', 'age', 'profession'])
ratings_description = pd.read_csv(ratings_file, delimiter=';', names=['userID', 'movieID', 'rating'])
predictions_description = pd.read_csv(predictions_file, delimiter=';', names=['userID', 'movieID'])


#####
##
## RANDOM PREDICTORS
##
#####

def predict(movies, users, ratings, predictions):
    number_predictions = len(predictions)

    return [[idx, randint(1, 5)] for idx in range(1, number_predictions + 1)]

#####
##
## Calculates Root Mean Squared error
##
#####

def rmse(predicted_ratings, true_ratings):
    return np.sqrt(np.mean((predicted_ratings - true_ratings) ** 2))

### User-User Collaborative Filtering variations: ###

#####
##
## COLLABORATIVE FILTERING: User-User approach with Pearson measure and Baseline modelling.
##
#####

def predict_cf_user_pearson_deviations(ratings, predictions, k=25):
    """
    Calculates the ratings for user/movie index listed in predictions through User-User Collaborative Filtering with K-Nearest Neighbour.
    :param ratings Movies x Users Matrix containing all ratings (previous ratings and NaN)
    :param predictions List of tuples (User Id, Movie Id) indicating what ratings should be predicted.
    :param k Number of neighbours to use values from.
    :return New Movies x User Matrix containing the previous ratings and the predicted ratings
    """

    # Calculate overall mean rating and deviation of all movies (mean score - overall mean)
    rating_matrix = ratings.replace(0, np.NaN).to_numpy()
    overall_mean_rating = np.nanmean(rating_matrix.flatten())
    user_means = np.nanmean(rating_matrix, axis=1)

    # Set NaN means to 0 since not all users have a rating
    user_means[np.isnan(user_means)] = 0
    user_deviations = user_means - overall_mean_rating

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
        user_indices = np.arange(0, len(rating_users))

        # Only keep weights where a rating exists (also removes the current rating to predict for).
        similarity_weights = similarity_weights[rating_users > 0].to_numpy()
        user_indices = user_indices[rating_users > 0]
        rating_users = rating_users[rating_users > 0].to_numpy()

        # Sorts the arrays on similarity scores and takes the last k values (highest k similarities)
        sort_indices = similarity_weights.argsort()
        sorted_similarity_weights = similarity_weights[sort_indices]
        sorted_rating_users = rating_users[sort_indices]
        sorted_user_indices = user_indices[sort_indices]
        sorted_similarity_weights = sorted_similarity_weights[-k:]
        sorted_rating_users = sorted_rating_users[-k:]
        sorted_user_indices = sorted_user_indices[-k:]

        # Set to [] in case sorted_user_indices results into a NaN
        if np.isnan(sorted_user_indices).any():
            sorted_user_indices = []

        # Calculate baseline estimate for the to be predicted rating
        movie_deviation = sorted_rating_users.mean() - overall_mean_rating

        # Set to 0 in case movie_deviation results into a NaN in order for it not to be counted
        if np.isnan(movie_deviation):
            movie_deviation = 0

        overall_baseline_estimate = overall_mean_rating + movie_deviation + user_deviations[movie_index-1]
        baselines_estimate_users = overall_mean_rating + movie_deviation + user_deviations[sorted_user_indices]

        # Check if sum of weights isn't 0 to avoid division by 0 and calculate the weighted average or leave it as 0.
        total_weight = sum(sorted_similarity_weights)
        if total_weight != 0:
            weighted_average = sum(sorted_similarity_weights * (sorted_rating_users - baselines_estimate_users))/total_weight
            predicted_rating = overall_baseline_estimate + weighted_average
            if predicted_rating <= 1:
                ratings[movie_index][user_index] = 1
            elif predicted_rating >= 5:
                ratings[movie_index][user_index] = 5
            elif 1 < predicted_rating < 5:
                ratings[movie_index][user_index] = predicted_rating
            else:
                ratings[movie_index][user_index] = overall_baseline_estimate
        else:
            ratings[movie_index][user_index] = overall_baseline_estimate

    return ratings

#####
##
## COLLABORATIVE FILTERING: User-User approach with Cosine measure.
##
#####

def predict_cf_user_cosine(ratings, predictions, k = 25):
    """
    Calculates the ratings for user/movie index listed in predictions through User-User Collaborative Filtering with K-Nearest Neighbour.
    :param ratings Movies x Users Matrix containing all ratings (previous ratings and NaN)
    :param predictions List of tuples (User Id, Movie Id) indicating what ratings should be predicted.
    :param k Number of neighbours to use values from.
    :return New Movies x User Matrix containing the previous ratings and the predicted ratings
    """
    # Calculate similarity (cosine correlation) scores of users.
    d = ratings @ ratings.T
    norm = (ratings.dot(ratings.T)).sum(0) ** .5
    utility_matrix = d / norm / norm.T
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
## COLLABORATIVE FILTERING: User-User approach with Cosine measure and Baseline modelling.
##
#####

def predict_cf_user_cosine_deviations(ratings, predictions, k=25):
    """
    Calculates the ratings for user/movie index listed in predictions through User-User Collaborative Filtering with K-Nearest Neighbour.
    :param ratings Movies x Users Matrix containing all ratings (previous ratings and NaN)
    :param predictions List of tuples (User Id, Movie Id) indicating what ratings should be predicted.
    :param k Number of neighbours to use values from.
    :return New Movies x User Matrix containing the previous ratings and the predicted ratings
    """

    # Calculate overall mean rating and deviation of all movies (mean score - overall mean)
    rating_matrix = ratings.replace(0, np.NaN).to_numpy()
    overall_mean_rating = np.nanmean(rating_matrix.flatten())
    user_means = np.nanmean(rating_matrix, axis=1)

    # Set NaN means to 0 since not all users have a rating
    user_means[np.isnan(user_means)] = 0
    user_deviations = user_means - overall_mean_rating

    # Calculate similarity (cosine correlation) scores of users.
    d = ratings @ ratings.T
    norm = (ratings.dot(ratings.T)).sum(0) ** .5
    utility_matrix = d / norm / norm.T
    utility_matrix.fillna(0)
    utility_matrix.fillna(0)

    # Iterate over all values that need to be predicted.
    for row in predictions.to_numpy():
        # Get indices of userID en movieID to predict rating for.
        user_index = row[0]
        movie_index = row[1]

        # Retrieve the similarity weights as vector and also the ratings of the users for a particular movie.
        similarity_weights = utility_matrix[user_index]
        rating_users = ratings[movie_index]
        user_indices = np.arange(0, len(rating_users))

        # Only keep weights where a rating exists (also removes the current rating to predict for).
        similarity_weights = similarity_weights[rating_users > 0].to_numpy()
        user_indices = user_indices[rating_users > 0]
        rating_users = rating_users[rating_users > 0].to_numpy()

        # Sorts the arrays on similarity scores and takes the last k values (highest k similarities)
        sort_indices = similarity_weights.argsort()
        sorted_similarity_weights = similarity_weights[sort_indices]
        sorted_rating_users = rating_users[sort_indices]
        sorted_user_indices = user_indices[sort_indices]
        sorted_similarity_weights = sorted_similarity_weights[-k:]
        sorted_rating_users = sorted_rating_users[-k:]
        sorted_user_indices = sorted_user_indices[-k:]

        # Set to [] in case sorted_user_indices results into a NaN
        if np.isnan(sorted_user_indices).any():
            sorted_user_indices = []

        # Calculate baseline estimate for the to be predicted rating
        movie_deviation = sorted_rating_users.mean() - overall_mean_rating

        # Set to 0 in case user_deviation results into a NaN in order for it not to be counted
        if np.isnan(movie_deviation):
            movie_deviation = 0

        overall_baseline_estimate = overall_mean_rating + movie_deviation + user_deviations[movie_index-1]
        baselines_estimate_users = overall_mean_rating + movie_deviation + user_deviations[sorted_user_indices]

        # Check if sum of weights isn't 0 to avoid division by 0 and calculate the weighted average or leave it as 0.
        total_weight = sum(sorted_similarity_weights)
        if total_weight != 0:
            weighted_average = sum(sorted_similarity_weights * (sorted_rating_users - baselines_estimate_users))/total_weight
            predicted_rating = overall_baseline_estimate + weighted_average
            if predicted_rating <= 1:
                ratings[movie_index][user_index] = 1
            elif predicted_rating >= 5:
                ratings[movie_index][user_index] = 5
            elif 1 < predicted_rating < 5:
                ratings[movie_index][user_index] = predicted_rating
            else:
                ratings[movie_index][user_index] = overall_baseline_estimate
        else:
            ratings[movie_index][user_index] = overall_baseline_estimate

    return ratings



### Item-Item Collaborative Filtering variations: ###

#####
##
## COLLABORATIVE FILTERING: Item-Item approach with Pearson Measure.
##
#####

def predict_cf_item_pearson(ratings, predictions, k=25):
    """
    Calculates the ratings for user/movie index listed in predictions through Item-Item Collaborative Filtering with K-Nearest Neighbour.
    :param ratings Movies x Users Matrix containing all ratings (previous ratings and NaN)
    :param predictions List of tuples (User Id, Movie Id) indicating what ratings should be predicted.
    :param k Number of neighbours to use values from.
    :return New Movies x User Matrix containing the previous ratings and the predicted ratings
    """
    # Calculate similarity (pearson correlation) scores of movies.
    utility_matrix = ratings.corr()
    utility_matrix.fillna(0)
    rating_matrix = ratings.T

    # Iterate over all values that need to be predicted.
    for row in predictions.to_numpy():
        # Get indices of userID en movieID to predict rating for.
        user_index = row[0]
        movie_index = row[1]

        # Retrieve the similarity weights as vector and also the ratings for the movies from a particular user.
        similarity_weights = utility_matrix[movie_index]
        rating_movies = rating_matrix[user_index]

        # Only keep weights where a rating exists.
        similarity_weights = similarity_weights[rating_movies > 0].to_numpy()
        rating_movies = rating_movies[rating_movies > 0].to_numpy()

        # Sorts the arrays on similarity scores and takes the last k values (highest k similarities)
        sort_indices = similarity_weights.argsort()
        sorted_similarity_weights = similarity_weights[sort_indices]
        sorted_rating_movies = rating_movies[sort_indices]
        sorted_similarity_weights = sorted_similarity_weights[-k:]
        sorted_rating_movies = sorted_rating_movies[-k:]

        # Check if sum of weights isn't 0 to avoid division by 0 and Calculate the weighted average.
        total_weight = sum(sorted_similarity_weights)
        if total_weight != 0:
            weighted_average = sum(sorted_similarity_weights * sorted_rating_movies)/total_weight
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
## COLLABORATIVE FILTERING: Item-Item approach with Pearson Measure and Baseline modelling.
##
#####

def predict_cf_item_pearson_deviations(ratings, predictions, k=25):
    """
    Calculates the ratings for user/movie index listed in predictions through Item-Item Collaborative Filtering with modelling the baseline.
    :param ratings Movies x Users Matrix containing all ratings (previous ratings and NaN)
    :param predictions List of tuples (User Id, Movie Id) indicating what ratings should be predicted.
    :param k Number of neighbours to use values from.
    :return New Movies x User Matrix containing the previous ratings and the predicted ratings
    """

    # Calculate overall mean rating and deviation of all movies (mean score - overall mean)
    rating_matrix = ratings.replace(0, np.NaN).to_numpy()
    overall_mean_rating = np.nanmean(rating_matrix.flatten())
    movie_means = np.nanmean(rating_matrix, axis=0)

    # Set NaN means to 0 since not all movies have a rating
    movie_means[np.isnan(movie_means)] = 0
    movie_deviations = movie_means - overall_mean_rating

    # Calculate similarity (pearson correlation) scores of movies.
    rating_matrix = ratings.T
    utility_matrix = ratings.corr()
    utility_matrix.fillna(0)

    for row in predictions.to_numpy():
        # Get indices of userID en movieID to predict rating for.
        user_index = row[0]
        movie_index = row[1]

        # Retrieve the similarity weights as vector and also the ratings for the movies from a particular user.
        similarity_weights = utility_matrix[movie_index]
        rating_movies = rating_matrix[user_index]
        movie_indices = np.arange(0, len(rating_movies))

        # Only keep weights where a rating exists (also removes the current rating to predict for).
        similarity_weights = similarity_weights[rating_movies > 0].to_numpy()
        movie_indices = movie_indices[rating_movies > 0]
        rating_movies = rating_movies[rating_movies > 0].to_numpy()

        # Sorts the arrays on similarity scores and takes the last k values (highest k similarities)
        sort_indices = similarity_weights.argsort()
        sorted_similarity_weights = similarity_weights[sort_indices]
        sorted_rating_movies = rating_movies[sort_indices]
        sorted_movie_indices = movie_indices[sort_indices]
        sorted_similarity_weights = sorted_similarity_weights[-k:]
        sorted_rating_movies = sorted_rating_movies[-k:]
        sorted_movie_indices = sorted_movie_indices[-k:]

        # Set to [] in case sorted_movie_indices results into a NaN
        if np.isnan(sorted_movie_indices).any():
            sorted_movie_indices = []

        # Calculate baseline estimate for the to be predicted rating
        user_deviation = sorted_rating_movies.mean() - overall_mean_rating

        # Set to 0 in case user_deviation results into a NaN in order for it not to be counted
        if np.isnan(user_deviation):
            user_deviation = 0

        overall_baseline_estimate = overall_mean_rating + user_deviation + movie_deviations[movie_index-1]
        baselines_estimate_movies = overall_mean_rating + user_deviation + movie_deviations[sorted_movie_indices]

        # Check if sum of weights isn't 0 to avoid division by 0 and calculate the weighted average.
        total_weight = sum(sorted_similarity_weights)
        if total_weight != 0:
            weighted_average = sum(sorted_similarity_weights * (sorted_rating_movies - baselines_estimate_movies))/total_weight
            predicted_rating = overall_baseline_estimate + weighted_average
            if predicted_rating <= 1:
                ratings[movie_index][user_index] = 1
            elif predicted_rating >= 5:
                ratings[movie_index][user_index] = 5
            elif 1 < predicted_rating < 5:
                ratings[movie_index][user_index] = predicted_rating
            else:
                ratings[movie_index][user_index] = overall_baseline_estimate
        else:
            ratings[movie_index][user_index] = overall_baseline_estimate

    return ratings

#####
##
## COLLABORATIVE FILTERING: Item-Item approach with Cosine Similarity.
##
#####

def predict_cf_item_cosine(ratings, predictions, k=25):
    """
    Calculates the ratings for user/movie index listed in predictions through Item-Item Collaborative Filtering with cosine similarity instead of pearson.
    :param ratings Movies x Users Matrix containing all ratings (previous ratings and NaN)
    :param predictions List of tuples (User Id, Movie Id) indicating what ratings should be predicted.
    :param k Number of neighbours to use values from.
    :return New Movies x User Matrix containing the previous ratings and the predicted ratings
    """
    # Calculate similarity (cosine correlation) scores of movies.
    d = ratings.T @ ratings
    norm = (ratings.T.dot(ratings)).sum(0) ** .5
    utility_matrix = d / norm / norm.T
    utility_matrix.fillna(0)
    rating_matrix = ratings.T

    # Iterate over all values that need to be predicted.
    for row in predictions.to_numpy():
        # Get indices of userID en movieID to predict rating for.
        user_index = row[0]
        movie_index = row[1]

        # Retrieve the similarity weights as vector and also the ratings for the movies from a particular user.
        similarity_weights = utility_matrix[movie_index]
        rating_movies = rating_matrix[user_index]

        # Only keep weights where a rating exists.
        similarity_weights = similarity_weights[rating_movies > 0].to_numpy()
        rating_movies = rating_movies[rating_movies > 0].to_numpy()

        # Sorts the arrays on similarity scores and takes the last k values (highest k similarities)
        sort_indices = similarity_weights.argsort()
        sorted_similarity_weights = similarity_weights[sort_indices]
        sorted_rating_movies = rating_movies[sort_indices]
        sorted_similarity_weights = sorted_similarity_weights[-k:]
        sorted_rating_movies = sorted_rating_movies[-k:]

        # Check if sum of weights isn't 0 to avoid division by 0 and Calculate the weighted average.
        total_weight = sum(sorted_similarity_weights)
        if total_weight != 0:
            weighted_average = sum(sorted_similarity_weights * sorted_rating_movies)/total_weight
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
## COLLABORATIVE FILTERING: Item-Item approach with Cosine Similarity and Baseline modelling.
##
#####

def predict_cf_item_cosine_deviations(ratings, predictions, k=25):
    """
    Calculates the ratings for user/movie index listed in predictions through Item-Item Collaborative Filtering with modelling the baseline.
    :param ratings Movies x Users Matrix containing all ratings (previous ratings and NaN)
    :param predictions List of tuples (User Id, Movie Id) indicating what ratings should be predicted.
    :param k Number of neighbours to use values from.
    :return New Movies x User Matrix containing the previous ratings and the predicted ratings
    """

    # Calculate overall mean rating and deviation of all movies (mean score - overall mean)
    rating_matrix = ratings.replace(0, np.NaN).to_numpy()
    overall_mean_rating = np.nanmean(rating_matrix.flatten())
    movie_means = np.nanmean(rating_matrix, axis=0)

    # Set means to 0 since not all movies have a rating
    movie_means[np.isnan(movie_means)] = 0
    movie_deviations = movie_means - overall_mean_rating

    # Calculate similarity (cosine correlation) scores of movies.
    rating_matrix = ratings.T
    d = ratings.T @ ratings
    norm = (ratings.T.dot(ratings))
    norm = norm.to_numpy().sum(0, keepdims=True) ** 0.5
    utility_matrix = d / norm / norm.T
    utility_matrix.fillna(0)

    for row in predictions.to_numpy():
        # Get indices of userID en movieID to predict rating for.
        user_index = row[0]
        movie_index = row[1]

        # Retrieve the similarity weights as vector and also the ratings for the movies from a particular user.
        similarity_weights = utility_matrix[movie_index]
        rating_movies = rating_matrix[user_index]
        movie_indices = np.arange(0, len(rating_movies))

        # Only keep weights where a rating exists (also removes the current rating to predict for).
        similarity_weights = similarity_weights[rating_movies > 0].to_numpy()
        movie_indices = movie_indices[rating_movies > 0]
        rating_movies = rating_movies[rating_movies > 0].to_numpy()

        # Sorts the arrays on similarity scores and takes the last k values (highest k similarities)
        sort_indices = similarity_weights.argsort()
        sorted_similarity_weights = similarity_weights[sort_indices]
        sorted_rating_movies = rating_movies[sort_indices]
        sorted_movie_indices = movie_indices[sort_indices]
        sorted_similarity_weights = sorted_similarity_weights[-k:]
        sorted_rating_movies = sorted_rating_movies[-k:]
        sorted_movie_indices = sorted_movie_indices[-k:]

        # Set to [] in case sorted_movie_indices results into a NaN
        if np.isnan(sorted_movie_indices).any():
            sorted_movie_indices = []

        # Calculate baseline estimate for the to be predicted rating
        user_deviation = sorted_rating_movies.mean() - overall_mean_rating

        # Set to 0 in case user_deviation results into a NaN in order for it not to be counted
        if np.isnan(user_deviation):
            user_deviation = 0

        overall_baseline_estimate = overall_mean_rating + user_deviation + movie_deviations[movie_index-1]
        baselines_estimate_movies = overall_mean_rating + user_deviation + movie_deviations[sorted_movie_indices]

        # Check if sum of weights isn't 0 to avoid division by 0 and calculate the weighted average.
        total_weight = sum(sorted_similarity_weights)
        if total_weight != 0:
            weighted_average = sum(sorted_similarity_weights * (sorted_rating_movies - baselines_estimate_movies))/total_weight
            predicted_rating = overall_baseline_estimate + weighted_average
            if predicted_rating <= 1:
                ratings[movie_index][user_index] = 1
            elif predicted_rating >= 5:
                ratings[movie_index][user_index] = 5
            elif 1 < predicted_rating < 5:
                ratings[movie_index][user_index] = predicted_rating
            else:
                ratings[movie_index][user_index] = overall_baseline_estimate
        else:
            ratings[movie_index][user_index] = overall_baseline_estimate

    return ratings

#####
##
## LATENT FACTORS: Stochastic Gradient Descent
##
#####

def predict_lf_sgd(ratings, epochs=10, k=25, alpha=0.001, beta=0.05):
    """
    Calculates the ratings for user/movie index listed in predictions through Stochastic Gradient Descent of Matrix Factorization.
    :param ratings Movies x Users Matrix containing all ratings (previous ratings and NaN)
    :param epochs Number of iterations
    :param k Number of features to use for the dimensionality reduction.
    :param alpha Learning rate
    :param beta Regulation term
    :return New Movies x User Matrix containing the previous ratings and the predicted ratings
    """
    ratings = ratings.to_numpy()
    user_amount, movie_amount = ratings.shape

    # Initialize random matrices for Q, P and also for the biases
    Q = np.random.uniform(low=0, high=5/k, size=(user_amount, k))
    P = np.random.uniform(low=0, high=5/k, size=(k,movie_amount))
    user_bias = np.zeros(user_amount)
    movie_bias = np.zeros(movie_amount)
    overall_mean_rating = np.mean(ratings[np.where(ratings != 0)])

    for epoch in range(epochs):
        for user_idx in range(user_amount):
            for movie_idx in range(movie_amount):

                # Check if rating already exists and calculate error by subtracting prediction from true rating
                if ratings[user_idx][movie_idx] > 0:
                    predicted_rating = overall_mean_rating + user_bias[user_idx] + movie_bias[movie_idx] + np.dot(Q[user_idx,:], P[:, movie_idx])
                    error = ratings[user_idx][movie_idx] - predicted_rating

                    # Update user and movie biases
                    user_bias[user_idx] += alpha * (error - beta * user_bias[user_idx])
                    movie_bias[movie_idx] += alpha * (error - beta * movie_bias[movie_idx])

                    # Update the latent factor feature matrices Q and P
                    for k_idx in range(k):
                        Q[user_idx][k_idx] += alpha * (2 * error * P[k_idx][movie_idx] - beta * Q[user_idx][k_idx])
                        P[k_idx][movie_idx] += alpha * (2 * error * Q[user_idx][k_idx] - beta * P[k_idx][movie_idx])

        print('Iteration ', epoch, ' - ')

    # Calculate final rating matrix and add bias to it.
    R = overall_mean_rating + user_bias[:, np.newaxis] + movie_bias[np.newaxis,:] +  np.dot(Q, P)

    # Round values outside the range 1 <= rating <= 5 up to 1 or down to 5.
    R[R > 5] = 5
    R[R < 1] = 0
    return R



#####
##
## FINAL PREDICTORS
##
#####
def predict_combined(movies, users, ratings, predictions):
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
    # rating_matrix = predict_cf_user_pearson_deviations(ratings=rating_matrix, predictions=predictions, k=25)
    # rating_matrix = predict_cf_user_cosine(ratings=rating_matrix, predictions=predictions, k=25)
    # rating_matrix = predict_cf_user_cosine_deviations(ratings=rating_matrix, predictions=predictions, k=25)
    # rating_matrix = predict_cf_item_pearson(ratings=rating_matrix, predictions=predictions, k=25)
    # rating_matrix = predict_cf_item_pearson_deviations(ratings=rating_matrix, predictions=predictions, k=25)
    # rating_matrix = predict_cf_item_cosine(ratings=rating_matrix, predictions=predictions, k=25)
    rating_matrix = predict_cf_item_cosine_deviations(ratings=rating_matrix, predictions=predictions, k=25)
    print('-Finished executing cf nearest neighbour-')

    # Calls method to perform Matrix Factorization latent factor which returns a new rating matrix (numpy array)
    rating_matrix = predict_lf_sgd(rating_matrix, epochs=100, k=25, alpha=0.001, beta=0.01)
    print('-Finished executing cf matrix factorization-')

    # Enumerates the predicted ratings starting from 1 and returns a list of tuples in the format: (index, predicted rating)
    # In case latent factors is commented out and only collaborative filtering is run, add .to_numpy() to rating_matrix.
    final_rating = rating_matrix.to_numpy()
    prediction_ratings = []
    for row in predictions.to_numpy():
        prediction_ratings.append(final_rating[row[0]-1][row[1]-1])
    return list(enumerate(prediction_ratings, 1))

#####
##
## SAVE RESULTS
##
#####    

## //!!\\ TO CHANGE by your prediction function
predictions = predict_combined(movies_description, users_description, ratings_description, predictions_description)

#Save predictions, should be in the form 'list of tuples' or 'list of lists'
with open(submission_file, 'w') as submission_writer:
    #Formates data
    predictions = [map(str, row) for row in predictions]
    predictions = [','.join(row) for row in predictions]
    predictions = 'Id,Rating\n'+'\n'.join(predictions)

    #Writes it down
    submission_writer.write(predictions)