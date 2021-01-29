# Building-Recommendation-Engines
What are recommendation Engines? 
Recommendation engines use the feedback of users to find relevant items for them or for others with the assumption that users who has similar preference in the past are likely to have similar preference in the future. Recommendation engines benefit for having many to many match between users giving the feedback and the items receiving the feedback. In other words, better recommendation can be made for item that has been giving a lots of feedback and more personalized recommendations can be given for user for given a lot of feedback. 
ref: datacamp
# Inspect the listening_history_df DataFrame
print(listening_history_df.head())

      User            Song Title  Skipped Track  Rating
    0  User_001  Like a Rolling Stone           True       6
    1  User_001               Imagine          False       2
    2  User_001       What's Going On          False       9
    3  User_002               Respect          False       6
    4  User_003       Good Vibrations           True       0
    
   

# Calculate the number of unique values
print(listening_history_df[['Rating', 'Skipped Track']].nunique())
Rating           11
Skipped Track     2
    dtype: int64

# Display a histogram of the values in the Rating column
listening_history_df['Rating'].hist()
plt.show()

# NON_PERSONALIZED_RECOMMENDATION
They call this because they are made to all users without taking preference into account.

# Get the counts of occurrences of each movie title
movie_popularity = user_ratings_df["title"].value_counts()

#Print the titles of the top five most frequently seen movies.
# Inspect the most common values
print(movie_popularity.head().index)

# Find the mean of the ratings given to each title
average_rating_df = user_ratings_df[["title", "rating"]].groupby('title').mean()

# Order the entries by highest average rating to lowest
sorted_average_ratings = average_rating_df.sort_values(by='rating', ascending=False)

# Inspect the top movies
print(sorted_average_ratings.head())

