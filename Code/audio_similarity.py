############################
# Blake List

############################
# AUDIO SIMILARITY

# Choosing one of the audio feature datasets to analyse and predict genre.

# hdfs dfs -du -h /data/msd/audio/features

# Load necessary imports.
from pyspark.sql.types import *
from pyspark.sql import functions as F
import json
import pandas as pd
from pyspark.ml.stat import Correlation
from pyspark.ml.feature import VectorAssembler, StandardScaler, PCA, StringIndexer

# Define the attributes schema.
attribute_schema = StructType([
    StructField('Name', StringType()),
    StructField('Type', StringType())
    ])

# Create a dictionary for the data types we want to map to.
type_mapping = {
    'real' : DoubleType(),
    'NUMERIC' : DoubleType(),
    'string' : StringType(),
    'STRING' : StringType()
    }

# Define the name of the dataset we will use for the attributes and features.
data_name = "msd-jmir-area-of-moments-all-v1.0"

# Load the attributes data and apply the default schema.
attributes = (
    spark.read.format("com.databricks.spark.csv")
    .option("header", "false")
    .option("inferSchema", "false")
    .schema(attribute_schema)
    .load("hdfs:///data/msd/audio/attributes/%s.attributes.csv" % data_name)
    ) 

# Collect each row and type as a list of dictionaries.
results = attributes.toJSON().collect()

# Define an empty list to append the mapped data types.
schema_list = []

# Iterate over each dictionary and get the new data type mappings.
for row in results:
    data = json.loads(row)
    schema_list.append(StructField(data.get('Name'), type_mapping.get(data.get('Type')), True))

# Create a new schema structure using the dictionary of new mappings.
feature_schema = StructType(schema_list)

# Load in the features dataset with the proper schema.
features_jmir_area_moments= (
    spark.read.format("com.databricks.spark.csv")
    .option("header", "false")
    .option("inferSchema", "false")
    .schema(feature_schema)
    .load("hdfs:///data/msd/audio/features/%s.csv" % data_name)
    ) 

features_jmir_area_moments.show()

# Remove the quotations in the track ID column.
features_jmir_area_moments = features_jmir_area_moments.withColumn('MSD_TRACKID', F.col('MSD_TRACKID').substr(2, 18))
features_jmir_area_moments.show()

# Create a subset of the features dataset for describing.
features_subset = features_jmir_area_moments.select(features_jmir_area_moments.columns[:5])
features_subset.describe().show()

# Create a correlation matrix of the data.
# Convert to column of feature vectors.
vector_col = "corr_features"

assembler = VectorAssembler(
    inputCols=features_subset.columns, 
    outputCol=vector_col
    )

# Vectorise the data and drop na's.
features_vector = assembler.transform(features_subset.na.drop()).select(vector_col)

# Get correlation matrix
corr_matrix = Correlation.corr(features_vector, vector_col)

# Output the matrix.
corr_matrix.collect()[0]["pearson({})".format(vector_col)].values

# Other correlation implementation.
from pyspark.mllib.stat import Statistics
col_names = features_subset.columns
features = features_subset.na.drop().rdd.map(lambda row: row[0:])
corr_mat = Statistics.corr(features, method="pearson")
corr_df = pd.DataFrame(corr_mat)
corr_df.index, corr_df.columns = col_names, col_names
print(corr_df.to_string())

# For a correlation plot of all variables in the features dataset, see the Python script "Corr_plot.ipynb".

# For the first 5 columns, we see that Area_Method_of_Moments_Overall_Standard_Deviation_4 is highly 
# correlated with Area_Method_of_Moments_Overall_Standard_Deviation_5 with a correlation coefficient of
# 0.946 (3dp), and Area_Method_of_Moments_Overall_Standard_Deviation_2 is highly correlated with 
# Area_Method_of_Moments_Overall_Standard_Deviation_4 with a correlation coefficient of 0.849 (3dp).

##########################

# Define the schema of the MAGD dataset.
MAGD_schema = StructType([
    StructField('Track_ID', StringType()),
    StructField('Genre', StringType())
    ])

# Load each of the MAGD tsv's.
MAGD_data = (
    spark.read.format("com.databricks.spark.csv")
    .option("delimiter", "\t")
    .option("header", "false")
    .option("inferSchema", "false")
    .schema(MAGD_schema)
    .load("hdfs:///data/msd/genre/msd-MAGD-genreAssignment.tsv")
    )

# Get the counts of each genre in the MAGD dataset.
Genre_counts = (
    MAGD_data
    .groupBy('Genre')
    .agg(
        F.count(F.col('Genre'))
        )
    .select(
        F.col('Genre'),
        F.col('count(Genre)').alias('Count')
        )
    .orderBy('Count', ascending=False)
    )

Genre_counts.show(21)


# Convert the spark RDD to a pandas dataframe for loading into Python.
Genre_counts_pd = Genre_counts.toPandas().to_csv('/users/home/bwl25/Genre_counts.csv')

# Visualise the distribution of the genres for the songs that were matched.
# See Python script Genre_plot.ipynb.

##########################

# Merge the genres dataset and the audio features dataset.
genres_features = (
    features_jmir_area_moments
    .withColumnRenamed('MSD_TRACKID', 'Track_ID')
    .join(
        MAGD_data,
        on='Track_ID',
        how='left'
        )
    .filter(
        F.col('Genre').isNotNull()
        )
    )

genres_features.show(20, False)

############################

# Research and choose three classification algorithms from the spark.ml library.

# Logistic regression
# Random forest
# Linear SVM

# Get the numeric columns from the data.
num_cols = [item[0] for item in genres_features.dtypes if item[1].startswith('double')]
genres_features_nums = genres_features.select(*num_cols)

# Vectorise the numeric columns dataset.
assembler = VectorAssembler(
    inputCols=genres_features_nums.columns, 
    outputCol="features"
    )

# Apply the vectoriser.
transformed = assembler.transform(genres_features)

# Define the standard scaler.
scaler = StandardScaler(
    inputCol="features",
    outputCol="scaledFeatures",
    withMean=True,
    withStd=True 
    )

# Scale the numeric columns of the data.
scalerModel =  scaler.fit(transformed)
scaled_genres_features = scalerModel.transform(transformed).distinct()

# Select the desired columns from the data.
scaled_genres_features = (
    scaled_genres_features
    .select(
        F.col('Track_ID'), 
        F.col('scaledFeatures').alias('features'), 
        F.col('Genre')
        )
    )

scaled_genres_features.show()

# Define the principle component analysis object. We only want the top 10 features.
pca = PCA(
    k = 10, 
    inputCol="features",
    outputCol="pca_features"
    )

# Fit and transform the data into PCA features.
pca_model = pca.fit(scaled_genres_features)
scaled_genres_features = pca_model.transform(scaled_genres_features)

##########################

# Convert the genre column into a column representing if the song is "Electronic" or some other genre 
# as a binary label.
scaled_genres_features = (
    scaled_genres_features
    .withColumn(
        'label',
        F.when((F.col('Genre') == 'Electronic'), 1)
        .otherwise(0)
        )
    )

scaled_genres_features.show(20)

# Show the class balance of the binary label.
(
    scaled_genres_features
    .groupBy('label')
    .agg(
        F.count(F.col('label'))
        )
    .select(
        F.col('label'),
        F.col('count(label)').alias('Count')
        )
    .show()
    )

# +-----+------+
# |label| Count|
# +-----+------+
# |    1| 40662|
# |    0|379942|
# +-----+------+

# About 11% of the data has the class 'Electronic' as its genre. The dataset is quite imbalanced so we will
# need to use some resampling methods in order to balance the data.

##########################

# Split the data into train and test sets before resampling to deal with the class imbalance.
#train = scaled_genres_features.sampleBy('label', fractions={0: 0.7, 1: 0.7}, seed=10)
#test = scaled_genres_features.subtract(train)
#test = scaled_genres_features.join(train, on='Track_ID', how='left_anti')
train, test = scaled_genres_features.randomSplit([0.7, 0.3], 10)

# Show the class counts of the train and test sets before oversampling.
train.groupBy('label').count().show()
test.groupBy('label').count().show()

# Train set.
# +-----+------+
# |label| count|
# +-----+------+
# |    1| 28430|
# |    0|266059|
# +-----+------+

# Test set.
# +-----+------+
# |label| count|
# +-----+------+
# |    1| 12232|
# |    0|113883|
# +-----+------+

# Oversample the minority class by the ratio of the classes.
train_class_0 = train[train['label'] == 0]
train_class_1 = train[train['label'] == 1]
fraction = train_class_0.count()/train_class_1.count()
train_class_1_over_sample = train_class_1.sample(True, fraction, 10)
train = train_class_0.union(train_class_1_over_sample)

# Show the class counts of the train and test sets after oversampling.
train.groupBy('label').count().show()
test.groupBy('label').count().show()

# Train set.
# +-----+------+
# |label| count|
# +-----+------+
# |    1|266585|
# |    0|266059|
# +-----+------+

# Test set.
# +-----+------+
# |label| count|
# +-----+------+
# |    1| 12232|
# |    0|113883|
# +-----+------+

# There are various ways to deal with class imbalanced data. These include: oversampling - sampling more of 
# the minority class, undersampling - sampling fewer of the majority class, synthetic oversampling 
# (SMOTE/ROSE) - randomly generating synthetic samples from the minority class, systematic sampling - 
# choosing a proportion of each class in the train and test sets, cost sensitive learning - weighting the 
# metrics of the model to account for the class imbalance. Although there is no definitive winner between 
# the different sampling methods, oversampling has been performed to randomly sample observations from the 
# minority class with replacement. 

##########################

# Train the three classification algorithms.
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.classification import LinearSVC
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator

# Logistic Regression.
lr = LogisticRegression(
    maxIter=20, 
    regParam=0.3, 
    elasticNetParam=0,
    family='binomial', 
    featuresCol='pca_features', 
    labelCol='label')

# Fit the model to the train data.
lr_model = lr.fit(train)

# Predict on the test data.
lr_predictions = lr_model.transform(test)

# Show the prediction dataframe.
lr_predictions.select(
    'Genre',
    'pca_features', 
    'rawPrediction', 
    'probability',
    'prediction', 
    'label'
    ).show(10)

# Evaluate the prediction accuracy of the model.
multi_evaluator = MulticlassClassificationEvaluator(predictionCol="prediction")
print('Prediction Accuracy:', multi_evaluator.evaluate(lr_predictions))

# Compute the area under the roc curve.
binary_evaluator = BinaryClassificationEvaluator()
print('Area Under ROC:', binary_evaluator.evaluate(lr_predictions))

# Prediction Accuracy: 0.6654921584813003
# Area Under ROC: 0.6317063380889896

##########################

# Random Forest.
rf = RandomForestClassifier(
    numTrees=100,
    maxDepth=4,
    maxBins=32, 
    featuresCol='pca_features', 
    labelCol='label')

# Fit the model to the train data.
rf_model = rf.fit(train)

# Predict on the test data.
rf_predictions = rf_model.transform(test)

# Show the prediction dataframe.
rf_predictions.select(
    'Genre',
    'pca_features', 
    'rawPrediction', 
    'probability',
    'prediction', 
    'label'
    ).show(10)

# Evaluate the prediction accuracy of the model.
multi_evaluator = MulticlassClassificationEvaluator(predictionCol="prediction")
print('Prediction Accuracy:', multi_evaluator.evaluate(rf_predictions))

# Compute the area under the roc curve.
binary_evaluator = BinaryClassificationEvaluator()
print('Area Under ROC:', binary_evaluator.evaluate(rf_predictions))

# Prediction Accuracy: 0.5996065814745933
# Area Under ROC: 0.6727650907190462

##########################

# Linear Support Vector Machine.
svm = LinearSVC(
    maxIter=5, 
    regParam=0.3, 
    featuresCol='pca_features', 
    labelCol='label') 

# Fit the model to the train data.
svm_model = svm.fit(train)

# Predict on the test data.
svm_predictions = svm_model.transform(test)

# Output the results of the prediction.
svm_predictions.select(
    'Genre',
    'pca_features', 
    'rawPrediction', 
    'prediction', 
    'label'
    ).show(10)

# Evaluate the prediction accuracy of the model.
multi_evaluator = MulticlassClassificationEvaluator(predictionCol="prediction")
print('Prediction Accuracy:', multi_evaluator.evaluate(svm_predictions))

# Compute the area under the roc curve.
binary_evaluator = BinaryClassificationEvaluator()
print('Area Under ROC:', binary_evaluator.evaluate(svm_predictions))

# Prediction Accuracy: 0.632165944263369
# Area Under ROC: 0.6266893155239741

##########################

# Define a function to calculate the metrics of each model. 
# Although the multiclass classification evaluator has a form of weighted precision and recall, it is not 
# appropriate nor well defined for this case. Documentation and formulae can be found on the older
# Pyspark MLLib docs, however, it is best to manually define and calculate the metrics for each label.

def get_metrics(df, label):
    TP = df.where((F.col('prediction') == label) & (F.col('label') == label)).count()
    TN = df.where((F.col('prediction') != label) & (F.col('label') != label)).count()
    FP = df.where((F.col('prediction') == label) & (F.col('label') != label)).count()
    FN = df.where((F.col('prediction') != label) & (F.col('label') == label)).count()

    accuracy = (TP + TN) / df.count()
    
    if (TP == 0 and FP == 0):
        precision = 0.0
    else:
        precision = TP / (TP + FP)
    
    recall =  TP / (TP + FN)

    if (precision == 0.0 and recall == 0.0):
        f1_score = 0.0
    else:
        f1_score = (2 * precision * recall) / (precision + recall)

    print('Label: {0}\n accuracy: {1}\n f1_score: {2}\n precision: {3}\n recall: {4}\n'.format(
        label, accuracy, f1_score, precision, recall))
    
print('Logistic Regression:')
get_metrics(lr_predictions, label=1)

# Logistic Regression:
# Label: 1
#  accuracy: 0.580882527851564
#  f1_score: 0.22404909056210454
#  precision: 0.13654338218190276
#  recall: 0.6238554610856769

print('Random Forest:')
get_metrics(rf_predictions, label=1)

# Random Forest:
# Label: 1
#  accuracy: 0.508591365023986
#  f1_score: 0.2286034353995519
#  precision: 0.1348299759205967
#  recall: 0.7507357750163506

print('Linear SVM:')
get_metrics(svm_predictions, label=1)

# Linear SVM:
# Label: 1
#  accuracy: 0.5427982397018594
#  f1_score: 0.22289010485457827
#  precision: 0.13344414679017524
#  recall: 0.6760137344669719

# Use cross-validation to tune the hyperparameters of each model.

# Logistic Regression.
lr = LogisticRegression(
    maxIter=20, 
    regParam=0.3, 
    elasticNetParam=0,
    family='binomial', 
    featuresCol='pca_features', 
    labelCol='label')

# Create the parameter grid for cross-validation.
lr_param_grid = (ParamGridBuilder()
            .addGrid(lr.maxIter, [10, 20, 50])
            .addGrid(lr.regParam, [0.0, 0.1, 0.3, 0.5]) 
            .addGrid(lr.elasticNetParam, [0.0, 0.1, 0.2, 0.3]) 
            .build())

# Create a 5-fold cross-validator.
lr_cross_val = CrossValidator(
    estimator=lr,
    estimatorParamMaps=lr_param_grid,
    evaluator=binary_evaluator,
    numFolds=5)

# Fit the cross-validation model to the training data.
lr_cross_val_model = lr_cross_val.fit(train)

# Predict on the test data.
lr_cross_val_predictions = lr_cross_val_model.transform(test)

# Compute the metrics for the cross-validated model.
print('Logistic Regression with Cross Validation:')
get_metrics(lr_cross_val_predictions, label=1)

# Logistic Regression with Cross Validation:
# Label: 1
#  accuracy: 0.5773143559449708
#  f1_score: 0.22560541569214235
#  precision: 0.13717869446162
#  recall: 0.6348103335513408

##########################

# Random Forest.
rf = RandomForestClassifier(
    numTrees=100,
    maxDepth=4,
    maxBins=32, 
    featuresCol='pca_features', 
    labelCol='label')

# Create the parameter grid for cross-validation.
rf_param_grid = (ParamGridBuilder()
            .addGrid(rf.numTrees, [100, 250, 500])
            .addGrid(rf.maxDepth, [4, 5, 8, 10]) 
            .addGrid(rf.maxBins, [8, 16, 24, 32]) 
            .build())

# Create a 5-fold cross-validator.
rf_cross_val = CrossValidator(
    estimator=rf,
    estimatorParamMaps=rf_param_grid,
    evaluator=binary_evaluator,
    numFolds=5)

# Fit the cross-validation model to the training data.
rf_cross_val_model = rf_cross_val.fit(train)

# Predict on the test data.
rf_cross_val_predictions = rf_cross_val_model.transform(test)

# Compute the metrics for the cross-validated model.
print('Random Forest with Cross Validation:')
get_metrics(rf_cross_val_predictions, label=1)

# Random Forest with Cross Validation:
# Label: 1
#  accuracy: 0.6830829005272965
#  f1_score: 0.2633169904523169
#  precision: 0.16998239017657418
#  recall: 0.5839601046435579

##########################

# Linear Support Vector Machine.
svm = LinearSVC(
    maxIter=5, 
    regParam=0.3, 
    featuresCol='pca_features', 
    labelCol='label') 

# Create the parameter grid for cross-validation.
svm_param_grid = (ParamGridBuilder()
            .addGrid(svm.maxIter, [5, 7, 10, 15])
            .addGrid(svm.regParam, [0.0, 0.1, 0.2, 0.3]) 
            .build())

# Create a 5-fold cross-validator.
svm_cross_val = CrossValidator(
    estimator=svm,
    estimatorParamMaps=svm_param_grid,
    evaluator=binary_evaluator,
    numFolds=5)

# Fit the cross-validation model to the training data.
svm_cross_val_model = svm_cross_val.fit(train)

# Predict on the test data.
svm_cross_val_predictions = svm_cross_val_model.transform(test)

# Compute the metrics for the cross-validated model.
print('Linear SVM with Cross Validation:')
get_metrics(svm_cross_val_predictions, label=1)

# Linear SVM with Cross Validation:
# Label: 1
#  accuracy: 0.4737105023193117
#  f1_score: 0.21238623013848182
#  precision: 0.1242243784616666
#  recall: 0.7316056245912361

############################

# One-versus-Rest and One-versus-One

############################

# Convert the genre column into an integer index.
scaled_genres_features = scaled_genres_features.drop('label')

# Encode the column of labels to a column of indices, in order of frequency.
label_indexer = StringIndexer(
    inputCol='Genre', 
    outputCol='label')

# Fit and transform the label indexer to the data.
multi_class_model = label_indexer.fit(scaled_genres_features)
genres_features_multi_class = multi_class_model.transform(scaled_genres_features)

genres_features_multi_class.show()

# Show the distribution of classes for each label.
genres_features_multi_class.groupBy('Genre', 'label').count().orderBy('count', ascending=False).show(21)

############################

# Split the data into train and test sets before resampling to deal with the class imbalance.
train_multi, test_multi = genres_features_multi_class.randomSplit([0.7, 0.3], 10)

# Show the class counts of the train and test sets.
train_multi.groupBy('label').count().show()
test_multi.groupBy('label').count().show()

# To resample each class for multiclass classification, first we will calculate the mean value of the
# class counts, then iterate through each class: 
# If the positive class count is larger, undersample. If the class count is smaller, oversample. The ratio
# by which to over or undersample is calculated by the ratio of the mean count to the class count.

# Find the mean of the class counts. 10725
average_count = round(train_multi.groupBy('label').count().agg(F.mean('count')).collect()[0][0], 1)
 
# Iterate through each class and upsample or oversample.
for i in range(21): 
    print(i)
    train_class_0 = train_multi[train_multi['label'] != i]
    train_class_1 = train_multi[train_multi['label'] == i]
    fraction = average_count/train_class_1.count()
    train_class_1_sample = train_class_1.sample(True, fraction, 10)

    if(i == 0):
        train_multi_sampled = train_class_1_sample
    else:
        train_multi_sampled = train_multi_sampled.union(train_class_1_sample)

# Show the class counts of the train and test sets.
train_multi.groupBy('label').count().orderBy('label').show(21)
train_multi_sampled.groupBy('label').count().orderBy('label').show(21)
test_multi.groupBy('label').count().orderBy('label').show(21)

train_multi_sampled.cache()
test_multi.cache()

# Import the One-vs-Rest function from Pyspark.
from pyspark.ml.classification import OneVsRest

# Multiclass classification using random forest without resampling.
rf = RandomForestClassifier(
    numTrees=100,
    maxDepth=4,
    maxBins=32, 
    featuresCol='pca_features', 
    labelCol='label')

# Define the one-vs-rest model.
one_vs_rest_model = OneVsRest(classifier=rf)

# Fit the one-vs-rest model to the train data without resampling.
rf_ovr_model = one_vs_rest_model.fit(train_multi)

# Predict on the test data.
rf_ovr_predictions = rf_ovr_model.transform(test_multi)

# Compute the metrics for the one-vs-rest model.
print('Random Forest with OneVsRest:')
for i in range(21):
    get_metrics(rf_ovr_predictions, label=i)

# Random Forest with OneVsRest:
# Label: 0
#  accuracy: 0.566459184078024
#  f1_score: 0.7225638839838437
#  precision: 0.5658787811352546
#  recall: 0.9992421478092458

# Label: 1
#  accuracy: 0.90338183404036
#  f1_score: 0.027145708582834327
#  precision: 0.5802047781569966
#  recall: 0.013897972531066055

# Label: 2
#  accuracy: 0.9514570035285256
#  f1_score: 0.0
#  precision: 0.0
#  recall: 0.0

# Label: 3
#  accuracy: 0.9584744082781588
#  f1_score: 0.0
#  precision: 0.0
#  recall: 0.0

# Label: 4
#  accuracy: 0.9577449153550331
#  f1_score: 0.0
#  precision: 0.0
#  recall: 0.0

# Label: 5
#  accuracy: 0.9655393886532133
#  f1_score: 0.0
#  precision: 0.0
#  recall: 0.0

# Label: 6
#  accuracy: 0.9663719620980851
#  f1_score: 0.0
#  precision: 0.0
#  recall: 0.0

# Label: 7
#  accuracy: 0.9718669468342386
#  f1_score: 0.0
#  precision: 0.0
#  recall: 0.0

# Label: 8
#  accuracy: 0.9790587955437497
#  f1_score: 0.0
#  precision: 0.0
#  recall: 0.0

# Label: 9
#  accuracy: 0.9836498433968996
#  f1_score: 0.0
#  precision: 0.0
#  recall: 0.0

# Label: 10
#  accuracy: 0.9834040359988899
#  f1_score: 0.0
#  precision: 0.0
#  recall: 0.0

# Label: 11
#  accuracy: 0.9851564048685724
#  f1_score: 0.0
#  precision: 0.0
#  recall: 0.0

# Label: 12
#  accuracy: 0.986076200293383
#  f1_score: 0.0
#  precision: 0.0
#  recall: 0.0

# Label: 13
#  accuracy: 0.9907148237719542
#  f1_score: 0.0
#  precision: 0.0
#  recall: 0.0

# Label: 14
#  accuracy: 0.9950600642270944
#  f1_score: 0.0
#  precision: 0.0
#  recall: 0.0

# Label: 15
#  accuracy: 0.9964238988225033
#  f1_score: 0.0
#  precision: 0.0
#  recall: 0.0

# Label: 16
#  accuracy: 0.9963446061134679
#  f1_score: 0.0
#  precision: 0.0
#  recall: 0.0

# Label: 17
#  accuracy: 0.997565713832613
#  f1_score: 0.0
#  precision: 0.0
#  recall: 0.0

# Label: 18
#  accuracy: 0.9985806605082662
#  f1_score: 0.0
#  precision: 0.0
#  recall: 0.0

# Label: 19
#  accuracy: 0.9989612655116362
#  f1_score: 0.0
#  precision: 0.0
#  recall: 0.0

# Label: 20
#  accuracy: 0.9995321730166911
#  f1_score: 0.0
#  precision: 0.0
#  recall: 0.0

############################

# Multiclass classification using random forest with resampling.
# Fit the one-vs-rest model to the train data with resampling.
rf_ovr_model_sampled = one_vs_rest_model.fit(train_multi_sampled)

# Predict on the test data.
rf_ovr_sampled_predictions = rf_ovr_model_sampled.transform(test_multi)

# Compute the metrics for the one-vs-rest model.
print('Random Forest with resampling and OneVsRest:')
for i in range(21):
    get_metrics(rf_ovr_sampled_predictions, label=i)

# Random Forest with resampling and OneVsRest:
# Label: 0
#  accuracy: 0.5140387741347183
#  f1_score: 0.29481411591434714
#  precision: 0.8183328010220376
#  recall: 0.17979341510652033

# Label: 1
#  accuracy: 0.7592118304721881
#  f1_score: 0.192817841099386
#  precision: 0.14285714285714285
#  recall: 0.296517331589274

# Label: 2
#  accuracy: 0.8222098878008167
#  f1_score: 0.12741282689912828
#  precision: 0.0836313477061408
#  recall: 0.2673962757268866

# Label: 3
#  accuracy: 0.9300796891725805
#  f1_score: 0.13123152709359606
#  precision: 0.13555872175859962
#  recall: 0.12717204506396793

# Label: 4
#  accuracy: 0.9574911786861198
#  f1_score: 0.0007455731593662628
#  precision: 0.05555555555555555
#  recall: 0.0003753049352598987

# Label: 5
#  accuracy: 0.9652063592752647
#  f1_score: 0.0022737608003638014
#  precision: 0.09615384615384616
#  recall: 0.001150483202945237

# Label: 6
#  accuracy: 0.964889188439123
#  f1_score: 0.004496402877697842
#  precision: 0.04830917874396135
#  recall: 0.002357934449422306

# Label: 7
#  accuracy: 0.9718114419379138
#  f1_score: 0.0
#  precision: 0.0
#  recall: 0.0

# Label: 8
#  accuracy: 0.919565475954486
#  f1_score: 0.06541367237884652
#  precision: 0.04322415682454645
#  recall: 0.1344187807648618

# Label: 9
#  accuracy: 0.9786147563731515
#  f1_score: 0.042598509052183174
#  precision: 0.07947019867549669
#  recall: 0.029097963142580018

# Label: 10
#  accuracy: 0.9833961067279864
#  f1_score: 0.0
#  precision: 0.0
#  recall: 0.0

# Label: 11
#  accuracy: 0.9851564048685724
#  f1_score: 0.0
#  precision: 0.0
#  recall: 0.0

# Label: 12
#  accuracy: 0.9858700392498909
#  f1_score: 0.0
#  precision: 0.0
#  recall: 0.0

# Label: 13
#  accuracy: 0.9484200927724695
#  f1_score: 0.06791803983378708
#  precision: 0.040805785123966945
#  recall: 0.20239111870196413

# Label: 14
#  accuracy: 0.8920112595646831
#  f1_score: 0.03500318854956423
#  precision: 0.018309859154929577
#  recall: 0.39646869983948635

# Label: 15
#  accuracy: 0.9847678705942988
#  f1_score: 0.03321590337191746
#  precision: 0.021484375
#  recall: 0.07317073170731707

# Label: 16
#  accuracy: 0.9854339293501962
#  f1_score: 0.011834319526627219
#  precision: 0.007868383404864092
#  recall: 0.02386117136659436

# Label: 17
#  accuracy: 0.9582603179637632
#  f1_score: 0.036603221083455345
#  precision: 0.01939111886755866
#  recall: 0.3257328990228013

# Label: 18
#  accuracy: 0.9517503865519565
#  f1_score: 0.013296578563320904
#  precision: 0.006847027388109553
#  recall: 0.22905027932960895

# Label: 19
#  accuracy: 0.9674899892954842
#  f1_score: 0.0034030140982012637
#  precision: 0.0017574692442882249
#  recall: 0.05343511450381679

# Label: 20
#  accuracy: 0.889354953811997
#  f1_score: 0.002288002288002288
#  precision: 0.0011488475622890787
#  recall: 0.2711864406779661

############################

# Multiclass classification using random forest with resampling and cross-validation.
# Create the parameter grid for cross-validation.
rf_param_grid = (ParamGridBuilder()
            .addGrid(rf.numTrees, [100, 250, 500])
            .addGrid(rf.maxDepth, [4, 5, 8, 10]) 
            .addGrid(rf.maxBins, [8, 16, 24, 32]) 
            .build())

# Create a 5-fold cross-validator.
rf_cross_val = CrossValidator(
    estimator=rf,
    estimatorParamMaps=rf_param_grid,
    evaluator=binary_evaluator,
    numFolds=5)

# Fit the cross-validation model to the training data.
rf_cross_val_sampled_model = rf_cross_val.fit(train_multi_sampled)

# Predict on the test data.
rf_cross_val_sampled_predictions = rf_cross_val_sampled_model.transform(test_multi)

# Compute the metrics for the cross-validated model.
print('Random Forest with resampling, cross-validation and OneVsRest:')
for i in range(21):
    get_metrics(rf_cross_val_sampled_predictions, label=i)

# Random Forest with resampling, cross-validation and OneVsRest:
# Label: 0
#  accuracy: 0.5150854378939856
#  f1_score: 0.2845778594073537
#  precision: 0.8549237365572503
#  recall: 0.1706991888174699

# Label: 1
#  accuracy: 0.8929231257185902
#  f1_score: 0.10200824577736402
#  precision: 0.27334283677833215
#  recall: 0.06270438194898627

# Label: 2
#  accuracy: 0.8134321849105974
#  f1_score: 0.167910315804364
#  precision: 0.10715414127736403
#  recall: 0.387781770663182

# Label: 3
#  accuracy: 0.9584744082781588
#  f1_score: 0.0
#  precision: 0.0
#  recall: 0.0

# Label: 4
#  accuracy: 0.9563097173214923
#  f1_score: 0.007564841498559079
#  precision: 0.09417040358744394
#  recall: 0.003940701820228936

# Label: 5
#  accuracy: 0.9267018197676724
#  f1_score: 0.10495739736638264
#  precision: 0.09060514877967235
#  recall: 0.12471237919926369

# Label: 6
#  accuracy: 0.9663719620980851
#  f1_score: 0.0
#  precision: 0.0
#  recall: 0.0

# Label: 7
#  accuracy: 0.9315624628315426
#  f1_score: 0.07640449438202247
#  precision: 0.06158357771260997
#  recall: 0.10062006764374296

# Label: 8
#  accuracy: 0.8298854220354438
#  f1_score: 0.05397301349325337
#  precision: 0.030543494535110047
#  recall: 0.23173040514956456

# Label: 9
#  accuracy: 0.9811441937913808
#  f1_score: 0.009166666666666667
#  precision: 0.03254437869822485
#  recall: 0.00533462657613967

# Label: 10
#  accuracy: 0.9352971494271102
#  f1_score: 0.04806346243583762
#  precision: 0.03179503009723723
#  recall: 0.09842331581462016

# Label: 11
#  accuracy: 0.9691868532688419
#  f1_score: 0.08907641819034223
#  precision: 0.07936507936507936
#  recall: 0.1014957264957265

# Label: 12
#  accuracy: 0.9855290806010387
#  f1_score: 0.0010946907498631637
#  precision: 0.014084507042253521
#  recall: 0.0005694760820045558

# Label: 13
#  accuracy: 0.9629703048804662
#  f1_score: 0.14122839279146748
#  precision: 0.0899929692992735
#  recall: 0.32792485055508114

# Label: 14
#  accuracy: 0.8940887285414106
#  f1_score: 0.05544162364754969
#  precision: 0.028998372540316616
#  recall: 0.6292134831460674

# Label: 15
#  accuracy: 0.986662966340245
#  f1_score: 0.036655211912943866
#  precision: 0.02471042471042471
#  recall: 0.07095343680709534

# Label: 16
#  accuracy: 0.9740395670618087
#  f1_score: 0.026753864447086804
#  precision: 0.015501205649328281
#  recall: 0.09761388286334056

# Label: 17
#  accuracy: 0.9784006660587559
#  f1_score: 0.04488078541374474
#  precision: 0.025147347740667975
#  recall: 0.20846905537459284

# Label: 18
#  accuracy: 0.9714229076636404
#  f1_score: 0.029094827586206896
#  precision: 0.015284460798188508
#  recall: 0.3016759776536313

# Label: 19
#  accuracy: 0.9894699282400983
#  f1_score: 0.007473841554559044
#  precision: 0.004142502071251036
#  recall: 0.03816793893129771

# Label: 20
#  accuracy: 0.8701581889545257
#  f1_score: 0.001341708849179728
#  precision: 0.0006732770228914188
#  recall: 0.1864406779661017
