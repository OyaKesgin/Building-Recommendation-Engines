# Building-Recommendation-Engines
What are recommendation Engines? 
Recommendation engines use the feedback of users to find relevant items for them or for others with the assumption that users who has similar preference in the past are likely to have similar preference in the future. Recommendation engines benefit for having many to many match between users giving the feedback and the items receiving the feedback. In other words, better recommendation can be made for item that has been giving a lots of feedback and more personalized recommendations can be given for user for given a lot of feedback. 
ref: datacamp
# Implicit vs Explicit Data
You are beginning to see recommendation engines rely on data that records the preference of users.How these preferences are measured falling into two main groups.Implicit and explicit..
Explicit data contains direct feedback from a user how they feel about the item such as numerical rating or upvoting downvoting.Take a dataset for users rate a restaurant out of 5 stars. the feedback from user explicitly recorded implicit data relies on not users direct rating but instead users  actions summarise to preferences. such users watch the certain type of programmes or having specific buying history. Users historic choise of Spoitfy.

# Non_Personalized Recommendation
They call this as they all made to all users without taking the preferences into account. For instance, Frequenlty bought toegther items on Amazon .
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

# Permutations ( list, lenght_of_permutation) generate iterable object containing all permutation
#list- converts this object to a usable list.
# pd.DataFrame() - converts this list to a DataFrame containing the columns movie a and movie b

def create_pairs(x):
    pairs = pd.DataFrame(list(permutations(x.values,2)),columns=[movie_a,movie_b])
return pairs

# 2.1 Intro to Content Based Recommendation

In this chapter, I will move to more targeted models by recommending items based on similarities to items for users has likely it in the past.For example, if user likes A,we calculate book a and b similar. We belive users will like book b.We will address how to calculate what items are similar and which ones are not.We can do so by comparing attribute of items. Recommendations are made with similar attributes are called content based recommendations. for examples: in book, attributes= genre, language, page number, author, type and publishing date. Content based models requires to use any available attributes to build profiles of items the way allow us mathematically pair between them. This allow us for instance to find most similar items and recommend them. This is based on encoding as a vector. One to Many relationships is very common. We want to create a new table contains single  row for per item encoding whether or not has the attribute . to transform this data,we can use pandas pd.crosstab().This function generates cross tab relation of two or more factors. Here we want to used to find crosstab relation of the book titles and genres they have been labelled with.

print(movie_genre_df.head())
       name genre_list
0  Toy Story  Adventure
1  Toy Story  Animation
2  Toy Story   Children
3  Toy Story     Comedy
4  Toy Story    Fantasy

# Select only the rows with values in the name column equal to Toy Story
toy_story_genres = movie_genre_df[movie_genre_df['name'] == 'Toy Story']

# Inspect the subset
print(toy_story_genres)

# Select only the rows with values in the name column equal to Toy Story
toy_story_genres = movie_genre_df[movie_genre_df['name'] == 'Toy Story']

# Create cross-tabulated DataFrame from name and genre_list columns
movie_cross_table = pd.crosstab(movie_genre_df['name'], movie_genre_df['genre_list'])

# Select only the rows with Toy Story as the index
toy_story_genres_ct = movie_cross_table[movie_cross_table.index == 'Toy Story']
print(toy_story_genres_ct)

genre_list  Action  Adventure  Animation  Children  Comedy  ...  Drama  Fantasy  Horror  Romance  Thriller
name                                                        ...                                           
Toy Story        0          1          1         1       1  ...      0        1       0        0         0

[1 rows x 11 columns]

# Yogi Bear film
yogi_story_genres_ct = movie_cross_table[movie_cross_table.index == 'Yogi Bear']

print(yogi_story_genres_ct)
genre_list  Action  Adventure  Animation  Children  Comedy  Crime
name                                                             
Yogi Bear        0          0          0         1       1      0

# Dealing with Sparsity

What if data is less full. This actually common concern in world rating data. As a number of users and items are genereally quite high and the number of reviews are quite low. We call the percentage of data frame that is empty, data frame sparsity. another word the number of empty cells over the number of cells with data.
why sparsity is a problem. this creates a problem when we use KNN with sparse data because KNN requires you to find K near users that rate the item. In data set, the large numbers of books have received 1 or 2 reviews.
We can leverage Matrix factorization to deal with this problem remarkably well and create quite interesting features while doing so. Matrix factorizations When we decompose the user rating matrix into product two lower dimensionality matrises. These matrises shown here are factors of the original matrix on the left. if you are here to find product on two of them, this would be  disoriginal matrix. By finding factors of sparse matrix then multiplying together we can be left with fully filled matrix. We would dig into matrix factorization later but first we should review how matrix multiplication works. To multiply, two regtangle matrises, the number of rows in the first matrix(M) and the number of columns(N) on second matrix do not have to match but the number of columns on both matrixes must match the number of rows on the second.This results M x N matrix. The same multiplication can be performed in python using numpy dot product function. (np.dot).

# Matrix Factorization

it can be performed what values brings. just multiplices matrises together and they can be broken into factors. The huge benefit of these performed in conjuction with recommendation systems is that factors can be found as long as there is at least one value in every row and column. Or another words every user given at least one rating and every item has been rated at least once why this is valueable becasue we can multiply these factors together to create fully filled matrix . It calculates what values should be in these gaps base of incomplete matrixes factors. We will go into further depth about how we will do this but first lets run through how will factor the matrises. 

The matrix factorisation brings the matrix into two compound matrixes. Take a rating matrix with M users as rows and n items are rated as the columns. Matrix factorization will break this down into one matrix with depth is equal to the number of users. One matrix with a width equal to the number of items. The number of values in newly created dimension is here called the rank of the matrix and must be equal to each other and can be decided by us. What is this unlabel columns and rows represent.They are called latent features. These are the features matrix factorization view mathematically the best way to describe are sum up these dataset in least number of features. To explain , what is in the tails


SINGULAR VALUE DECOMPOSITION

There are many ways to find factors of matrix. We will use a technique called Singular Value Decomposition like any matrix factorization approach SVD finds factors for the matrix U is the user matrix V transpose is the feature matrix , transpose in this case v flipped over diagonal. we here dont need to worry about here. but it also generates sigma as seen here which is a diagonal matrix which can be thought of as the weigth of latent feautures or how large impact they are calculated to have.




