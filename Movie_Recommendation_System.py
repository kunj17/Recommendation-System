"""
Created on Wed Jan  3 08:15:43 2018

@author: KUNJ
"""

"""
1.Godfather-1
2.Ted
3.Straight outta Compton
4.Godfather-2
5.Notorious 
6.Get rich or die trying
7.Frozen
8.Tangled
9.Dunkirk
10.Interstellar
"""
from numpy import *

num_movies = 10
num_users = 5

ratings = random.randint(11, size = (num_movies, num_users))
print (ratings)
did_rate = (ratings != 0) * 1
print(did_rate)

ratings.shape
did_rate.shape
kunj_ratings = zeros((num_movies, 1))
print (kunj_ratings)
print (kunj_ratings[9]) 

kunj_ratings[0] = 8
kunj_ratings[4] = 7
kunj_ratings[7] = 3

print (kunj_ratings)
ratings = append(kunj_ratings, ratings, axis = 1)
did_rate = append(((kunj_ratings != 0) * 1), did_rate, axis = 1)
print (ratings)
ratings.shape
did_rate
print (did_rate)
did_rate.shape

def normalize_ratings(ratings, did_rate):
    num_movies = ratings.shape[0]
    
    ratings_mean = zeros(shape = (num_movies, 1))
    ratings_norm = zeros(shape = ratings.shape)
    
    for i in range(num_movies): 
        
        idx = where(did_rate[i] == 1)[0]
        ratings_mean[i] = mean(ratings[i, idx])
        ratings_norm[i, idx] = ratings[i, idx] - ratings_mean[i]
    
    return ratings_norm, ratings_mean

         
ratings, ratings_mean = normalize_ratings(ratings, did_rate)
print (ratings)

num_users = ratings.shape[1]
num_features = 3

movie_features = random.randn( num_movies, num_features )
user_prefs = random.randn( num_users, num_features )
initial_X_and_theta = r_[movie_features.T.flatten(), user_prefs.T.flatten()]


print(movie_features)


print (user_prefs)


print (initial_X_and_theta)


initial_X_and_theta.shape


movie_features.T.flatten().shape


user_prefs.T.flatten().shape


initial_X_and_theta

def unroll_params(X_and_theta, num_users, num_movies, num_features):
	first_30 = X_and_theta[:num_movies * num_features]
	
	X = first_30.reshape((num_features, num_movies)).transpose()
	
	last_18 = X_and_theta[num_movies * num_features:]
	
	theta = last_18.reshape(num_features, num_users ).transpose()
	return X, theta

def calculate_gradient(X_and_theta, ratings, did_rate, num_users, num_movies, num_features, reg_param):
	X, theta = unroll_params(X_and_theta, num_users, num_movies, num_features)
	
	# obs for which a rating was given
	difference = X.dot( theta.T ) * did_rate - ratings
	X_grad = difference.dot( theta ) + reg_param * X
	theta_grad = difference.T.dot( X ) + reg_param * theta
	
	return r_[X_grad.T.flatten(), theta_grad.T.flatten()]

def calculate_cost(X_and_theta, ratings, did_rate, num_users, num_movies, num_features, reg_param):
	X, theta = unroll_params(X_and_theta, num_users, num_movies, num_features)
	cost = sum( (X.dot( theta.T ) * did_rate - ratings) ** 2 ) / 2
	
	regularization = (reg_param / 2) * (sum( theta**2 ) + sum(X**2))
	return cost + regularization

from scipy import optimize

reg_param = 30

#scipy fmin
minimized_cost_and_optimal_params = optimize.fmin_cg(calculate_cost, fprime=calculate_gradient, x0=initial_X_and_theta,args=(ratings, did_rate, num_users, num_movies, num_features, reg_param),maxiter=100, disp=True, full_output=True ) 

cost, optimal_movie_features_and_user_prefs = minimized_cost_and_optimal_params[1], minimized_cost_and_optimal_params[0]

movie_features, user_prefs = unroll_params(optimal_movie_features_and_user_prefs, num_users, num_movies, num_features)

print(movie_features)


print(user_prefs)

all_predictions = movie_features.dot( user_prefs.T )


print(all_predictions)

predictions_for_kunj = all_predictions[:, 0:1] + ratings_mean
print (predictions_for_kunj)
print (kunj_ratings)
