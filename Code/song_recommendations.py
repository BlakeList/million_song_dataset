############################
# Blake List

############################
# SONG RECOMMENDATIONS

# hdfs dfs -du -h /data/msd/tasteprofile

# Load necessary imports.
from pyspark.sql.types import *
from pyspark.sql import functions as F

# From processing.

# Define the tasteprofile schema.
tasteprofile_schema = StructType([
    StructField('User_ID', StringType()),
    StructField('Song_ID', StringType()),
    StructField('Play_Count', IntegerType())
])

# Load the tasteprofile data.
tasteprofile = (
    spark.read.format("com.databricks.spark.csv")
    .option("delimiter", "\t")
    .option("header", "false")
    .option("inferSchema", "false")
    .schema(tasteprofile_schema)
    .load("hdfs:///data/msd/tasteprofile/triplets.tsv")
)

tasteprofile.show(20, False)

# Load in the mismatched data.
mismatches_text = (
    spark.read.format("text")
    .load('hdfs:///data/msd/tasteprofile/mismatches/sid_mismatches.txt')
)

# Parse the fixed width text data to format it.
mismatches = mismatches_text.select(
    F.trim(F.col('value').substr(9, 18)).alias('Song_ID').cast(StringType()),
    F.trim(F.col('value').substr(28, 18)).alias('Track_ID').cast(StringType())
)

mismatches.show(20, False)

# Load in the accepted mismatched data.
mismatches_accepted_text = (
    spark.read.format("text")
    .load('hdfs:///data/msd/tasteprofile/mismatches/sid_matches_manually_accepted.txt')
)

# Parse the fixed width text data to format it.
mismatches_accepted = mismatches_accepted_text.select(
    F.trim(F.col('value').substr(11, 18)).alias('Song_ID').cast(StringType()),
    F.trim(F.col('value').substr(30, 18)).alias('Track_ID').cast(StringType())
)

mismatches_accepted.show(20, False)

# Left anti join the tasteprofile data with the mismatches.
matches = (
    tasteprofile
    .join(
        mismatches
        .join(
            mismatches_accepted,
            on='Song_ID',
            how='left_anti'),
        on='Song_ID',
        how='left_anti')
    )

matches.show(20, False)

# Number of unique songs in the dataset. 378310 songs.
matches.select('Song_ID').distinct().count()

# Number of unique users in the dataset. 1019318 users.
matches.select('User_ID').distinct().count()

##########################

# Number of different songs the most active user has played. 195 unique songs with 13,074 total plays.
(
    matches
    .where(F.col('User_ID') == matches
        .groupBy('User_ID')
        .agg(F.sum('Play_Count').alias('User_Plays'))
        .orderBy('User_Plays', ascending=False)
        .collect()[0][0]
        )
    ).count()

# As a pecentage of the total number of unique songs, it is 195/378310 or ~ 0.05% of the total number of 
# songs.

##########################

# Distribution of song popularity and user activity.

# We can define song popularity as the number of unique users per song. We could say that song popularity is 
# the number of plays per song, but one person could have listened to a song one million times, that does
# not make it a popular song. Furthermore, we can define the user activity as the sum of the number of plays
# for each unique user. So, a user could listen to one song 100 times, or 100 songs once and it would give
# the same result.

# Define the song popularity dataframe.
song_popularity = (
    matches
    .groupBy('Song_ID')
    .agg(F.countDistinct('User_ID').alias('N_Users'))
    .orderBy('N_Users', ascending=False)
    )

song_popularity.show()

# Convert the spark RDD to a pandas dataframe for loading into Python.
song_popularity_pd = song_popularity.toPandas().to_csv('/users/home/bwl25/song_popularity.csv')

# Define the user activity dataframe.
user_activity = (
    matches
    .groupBy('User_ID')
    .agg(F.sum('Play_Count').alias('N_Plays'))
    .orderBy('N_Plays', ascending=False)
    )

user_activity.show()

# Convert the spark RDD to a pandas dataframe for loading into Python.
user_activity_pd = user_activity.toPandas().to_csv('/users/home/bwl25/user_activity.csv')

# See Python script "Song_distributions.ipynb" for the visualisations of the song popularity and user 
# activity.

##########################

# Create a clean dataset of user-song plays by removing songs which have been played less than N times 
# and users who have listened to fewer than M songs in total.
# We will define the region by which to remove user-song plays by the mean of the number of unique songs
# played by each user and the mean of the total plays for each song. 

# Filter the taste profile data removing the songs with less than 347 plays.
wanted_songs = (
    matches.groupBy('Song_ID')
    .agg(F.sum('Play_Count').alias('Total_Plays'))
    .filter(F.col('Total_Plays') >= round(matches
        .groupBy('Song_ID')
        .agg(F.sum('Play_Count').alias('Total_Plays'))
        .agg(F.mean('Total_Plays'))
        .collect()[0][0]) # 347
        )
    .orderBy('Total_Plays', ascending=False)
    )

wanted_songs.show()

# Left semi-join with the taste profile data.
matches = (
    matches
    .join(
        wanted_songs,
        on='Song_ID',
        how='left_semi')
    )

# Filter the taste profile data removing the users who have played less than 38 songs.
wanted_users = (
    matches.groupBy('User_ID')
    .agg(F.countDistinct('Song_ID').alias('N_Songs'))
    .filter(F.col('N_Songs') >= round(matches
        .groupBy('User_ID')
        .agg(F.countDistinct('Song_ID').alias('N_Songs'))
        .agg(F.mean('N_Songs'))
        .collect()[0][0]) # 38
        )
    .orderBy('N_Songs', ascending=False)
    )

wanted_users.show()

# Left semi-join with the taste profile data again.
matches_clean = (
    matches
    .join(
        wanted_users,
        on='User_ID',
        how='left_semi')
    )

# We are filtering out songs that have less than 347 total plays and users who have listened to less than
# 38 songs.

# These datasets are then left-semi joined back onto the taste profile dataset.
# Left-semi join only returns observations from the left-hand dataframe that match with the right-hand 
# dataframe.

matches_clean.cache()
matches_clean.show()

##########################

# Looking ahead, in order to train the alternating least squares model both the user ID and the song ID 
# need to be input as numerics. So these must be encoded using the string indexer.
from pyspark.ml.feature import StringIndexer

# Encode the column of user IDs to a column of indices.
user_indexer = StringIndexer(
    inputCol='User_ID', 
    outputCol='User_ID_encoded')

# Fit and transform the user ID indexer to the data.
user_indexer_model = user_indexer.fit(matches_clean)
matches_clean = user_indexer_model.transform(matches_clean)

# Encode the column of song IDs to a column of indices.
song_indexer = StringIndexer(
    inputCol='Song_ID', 
    outputCol='Song_ID_encoded')

# Fit and transform the song ID indexer to the data.
song_indexer_model = song_indexer.fit(matches_clean)
matches_clean = song_indexer_model.transform(matches_clean)

matches_clean.cache()
matches_clean.show()

##########################

# Remove users that have less that two unique songs
matches_clean.groupBy('User_ID').agg(F.countDistinct('Song_ID').alias('N_Songs')) \
.orderBy('N_Songs').show()

matches_clean.count() # 25887959.
matches_clean.select('User_ID').distinct().count() # 308225 unique users.

# First define the train data as having one observation per unique user ID.
train_collab = (
    matches_clean
    .dropDuplicates(['User_ID'])
    ).orderBy('User_ID', ascending=False)

train_collab.show()
train_collab.count()  # Should be 308225.

# Remove the train observations from the data and drop duplicate user IDs to ensure the test set
# has one observation per unique user ID.
test_collab = (
    matches_clean
    .subtract(train_collab)
    .dropDuplicates(['User_ID'])
    )

test_collab.show()
test_collab.count() # Should also be 308225.

# Ensure we have the same User IDs in both train and test by left anti-joining the initial train and
# test sets
train_collab.join(test_collab, how='left_anti', on='User_ID').count() # Should be 0.

# Remove the observations in the train and test sets from the original data.
matches_clean_split = (
    matches_clean
    .subtract(train_collab)
    .subtract(test_collab)
    )

matches_clean_split.show()
matches_clean_split.count() # Should be 25887959 - 308225 - 308225 = 25271509

# Do a random train test split on the reduced data.
matches_clean_split_train, matches_clean_split_test = matches_clean_split.randomSplit([0.8, 0.2], 10)

# Stack the initial train test splits with the newly created random splits.
train_collab = train_collab.union(matches_clean_split_train)
test_collab = test_collab.union(matches_clean_split_test)

train_collab.cache()
test_collab.cache()

train_collab.show()
train_collab.count() # 20523824 / 25887959 = 0.79 = 79% of data in train set.

test_collab.show()
test_collab.count() # 5363990 / 25887959 = 0.21 = 21% of data in the test set.

# We can confirm we have each unique user in both splits. 308225 unique users in each set.
train_collab.select('User_ID').distinct().count() 
test_collab.select('User_ID').distinct().count()

# Confirm that the encoded IDs matched the original IDs.
train_collab.orderBy('User_ID', ascending=True).show()
test_collab.orderBy('User_ID', ascending=True).show()
train_collab.orderBy('Song_ID').show()
test_collab.orderBy('Song_ID').show()

##########################

# Train an implicit matrix factorization model using Alternating Least Squares (ALS).
import numpy as np
from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.mllib.evaluation import RankingMetrics
from pyspark.sql import Window

# Define the alternating least squares model and set the input columns.
als = ALS(maxIter=5, seed=10, 
    userCol='User_ID_encoded', 
    itemCol='Song_ID_encoded', 
    ratingCol='Play_Count')

# Fit the ALS model to the collaborative filtering train data.
als_model = als.fit(train_collab)

# Predict on the collaborative filtering test data.
als_model_predictions = als_model.transform(test_collab)
als_model_predictions.orderBy('prediction', ascending=False).show()

# Evaluate the collaborative filtering model using the root mean squared error. RMSE = 5.629 (3dp)
evaluator = RegressionEvaluator(metricName="rmse", labelCol="Play_Count", predictionCol="prediction")
evaluator.evaluate(als_model_predictions.filter(F.col('prediction') != np.NaN))

##########################

# Generate some recommendations for some users from the test set.

# Define the rank. 
k = 10 

# Create a dataframe of recommended songs for each user from the predictions data.
windowSpec = Window.partitionBy('User_ID_encoded').orderBy(F.col('prediction').desc())
predicted_songs_per_user = (
    als_model_predictions
    .select(
        F.col('User_ID_encoded'), 
        F.col('Song_ID_encoded'), 
        F.col('prediction'), 
        F.rank().over(windowSpec).alias('rank'))
    .where('rank <= {0}'.format(k))
    .groupBy('User_ID_encoded')
    .agg(F.expr('collect_list(Song_ID_encoded) as recommended_songs'))
    )

predicted_songs_per_user.cache()
predicted_songs_per_user.show(10, 100)

# Create a dataframe of actual songs played by each user from the predictions data.
windowSpec = Window.partitionBy('User_ID_encoded').orderBy(F.col('Play_Count').desc())
actual_songs_per_user = (
    als_model_predictions
    .select(
        F.col('User_ID_encoded'), 
        F.col('Song_ID_encoded'),
        F.col('Play_Count'), 
        F.rank().over(windowSpec).alias('rank'))
    .where('rank <= {0}'.format(k))
    .groupBy('User_ID_encoded')
    .agg(F.expr('collect_list(Song_ID_encoded) as relevant_songs'))
    )

actual_songs_per_user.cache()
actual_songs_per_user.show(10, 100)

# Join the song recommendations and actual songs played together for each user.
user_songs = (
    predicted_songs_per_user
    .join(
        actual_songs_per_user,
        on='User_ID_encoded',
        how='inner'
        )
    )

user_songs.show(20, 100)

# Select only the recommended and actual song columns and convert the dataframe to an rdd. 
user_songs_rdd = user_songs.select(F.col('recommended_songs'), F.col('relevant_songs')).rdd
user_songs_rdd.cache()

# Compute the ranking metrics for the collaborative filtering model.
rank_metrics = RankingMetrics(user_songs_rdd)

# Precision @ 5: 0.907 (3dp)
precision_at_5 = rank_metrics.precisionAt(5)
print(precision_at_5)

# NDCG @ 10: 0.906 (3dp)
ndcg_at_10 = rank_metrics.ndcgAt(10)
print(ndcg_at_10)

# MAP: 0.699 (3dp)
mean_average_precision = rank_metrics.meanAveragePrecision
print(mean_average_precision)
