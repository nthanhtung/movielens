library(caret)
library(dplyr)
library(dslabs)
library(ggplot2)

#create dataset


# Note: this process could take a couple of minutes

if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")

# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip

dl <- tempfile()
download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings <- read.table(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                      col.names = c("userId", "movieId", "rating", "timestamp"))

movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
colnames(movies) <- c("movieId", "title", "genres")
movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(levels(movieId))[movieId],
                                           title = as.character(title),
                                           genres = as.character(genres))

movielens <- left_join(ratings, movies, by = "movieId")

# Validation set will be 10% of MovieLens data

set.seed(1)
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Make sure userId and movieId in validation set are also in edx set

validation <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add rows removed from validation set back into edx set

removed <- anti_join(temp, validation)
edx <- rbind(edx, removed)

rm(dl, ratings, movies, test_index, temp, movielens, removed)

#RMSE function

RMSE <- function(predicted_value, actual_value){
  sqrt(mean((predicted_value - actual_value)^2))
}

#Data exploration

nrow(edx)
nrow(validation)
ncol(edx)
ncol(validation)
summary(edx)
#Distribution of rating
edx %>% 
  ggplot(aes(x = rating)) +
  geom_histogram(binwidth = 0.25, fill = "pink", color = "green") +
  scale_x_discrete(limits = c(seq(0.5,5,0.5))) + 
  scale_y_continuous(name = "number of rating", labels = scales::comma) +
  ggtitle("Rating distribution")

n_rating_by_movie <- edx %>% 
  group_by(movieId, title) %>%
  summarise(avg_rating = mean(rating), n_rating = n()) %>%
  arrange(desc(n_rating))

#average number of rating of all movie
mean(n_rating_by_movie$n_rating)
#distribution of number of rating by movie
n_rating_by_movie %>%
  ggplot(aes(x = n_rating)) +
  geom_histogram(fill = "pink", color = "green") +
  scale_y_continuous(name = "number of movie", labels = scales :: comma) +
  ggtitle("distribution of number of rating by movie")

#correlation of number of rating and rating

n_rating_by_movie %>%
  ggplot(aes(x = n_rating, y = avg_rating)) +
  geom_line() +
  geom_smooth()

#average method

mu <- mean(edx$rating)
mu_RMSE <- RMSE(validation$rating, mu)
naive_rmse <- RMSE(validation$rating, mu) 
naive_rmse
rmse_results <- data_frame(method = "Naive model", RMSE = naive_rmse) 
rmse_results %>% knitr::kable()

#movie effect method

#data
movie_bias_data <- edx %>% 
  group_by(movieId) %>%
  summarise(movie_bias = mean(rating - mu)) %>%
  mutate(mu = mu)
#model
movie_bias_model <- left_join(x = validation, y = movie_bias_data, by = "movieId") %>%
  mutate(y_hat_movie_bias = mu + movie_bias)
#movie bias RMSE
movie_bias_RMSE <- RMSE(movie_bias_model$rating, movie_bias_model$y_hat_movie_bias)
rmse_results <- rbind(rmse_results, 
                      data.frame(method = "Movie rating model", RMSE = movie_bias_RMSE))
rmse_results

#movie & user effect method

#user & movie bias data
user_bias_data <- edx %>%
  left_join(y = movie_bias_data, by = "movieId") %>%
  group_by(userId) %>% 
  summarise(user_bias = mean(rating - mu - movie_bias))
#model
user_movie_bias_model <- movie_bias_model %>% 
  left_join(y = user_bias_data, by = "userId") %>%
  mutate(y_hat = mu + user_bias + movie_bias)

#user & movie bias RMSE
user_movie_bias_RMSE <- RMSE(user_movie_bias_model$rating, user_movie_bias_model$y_hat)

rmse_results <- rbind(rmse_results, 
                      data.frame(method = "Movie & User rating model", RMSE = user_movie_bias_RMSE))
rmse_results

#regularization method

#some movie has many few rating -> untrustworthy
edx %>% group_by(movieId) %>% summarise(n =n()) %>% arrange(n) %>% filter(n < 10)

#regularized parameter
lambdas <- seq(0, 10, 0.25)

RMSE_lambdas <- sapply(lambdas, function(l){
  
  mu <- mean(edx$rating)
  
  b_i <- edx %>% 
    group_by(movieId) %>%
    summarize(b_i = sum(rating - mu)/(n()+l))
  
  b_u <- edx %>% 
    left_join(b_i, by="movieId") %>%
    group_by(userId) %>%
    summarize(b_u = sum(rating - b_i - mu)/(n()+l))
  
  predicted_ratings <- 
    validation %>% 
    left_join(b_i, by = "movieId") %>%
    left_join(b_u, by = "userId") %>%
    mutate(pred = mu + b_i + b_u) %>%
    pull(pred)
  
  return(RMSE(predicted_ratings, validation$rating))
})

result_lambdas <- data.frame(lambdas = lambdas, RMSE_lambdas = RMSE_lambdas)
result_lambdas %>% ggplot(aes( x = lambdas, y = RMSE_lambdas)) + geom_point()
lambdas[which(RMSE_lambdas == min(RMSE_lambdas))]
rl_user_movie_bias_RMSE <- min(RMSE_lambdas)

rmse_results <- rbind(rmse_results, 
                      data.frame(method = "RL Movie & User rating model", RMSE = rl_user_movie_bias_RMSE))
rmse_results