############################
# Blake List

############################
# PROCESSING

# Filter the tasteprofile data to remove mimatched songs.

# Load necessary imports.
from pyspark.sql.types import *
from pyspark.sql import functions as F

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

# Remove the manually accepted mismatches from the mismatches dataset then left anti-join the 
# tasteprofile data with the mismatches.
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
matches.count()
# Total number of matches is 45795111 / 48373586

##########################

# We need to convert each row in the attributes dataframe into json format so that we can create
# a structtype from the json data. We can initialise a default schema to load the attributes data in 
# then use json to extract the structure for each row.

import json

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

# Create a list of each feature and attribute dataset.
data_names = [
"msd-jmir-area-of-moments-all-v1.0",
"msd-jmir-lpc-all-v1.0",
"msd-jmir-methods-of-moments-all-v1.0",
"msd-jmir-mfcc-all-v1.0",
"msd-jmir-spectral-all-all-v1.0",
"msd-jmir-spectral-derivatives-all-all-v1.0",
"msd-marsyas-timbral-v1.0",
"msd-mvd-v1.0",
"msd-rh-v1.0",
"msd-rp-v1.0",
"msd-ssd-v1.0",
"msd-trh-v1.0",
"msd-tssd-v1.0"
]

# Define a function to extract the correct schema for the attributes and features.
def get_schema(i):

    # Load the attributes data.
    attributes = (
        spark.read.format("com.databricks.spark.csv")
        .option("header", "false")
        .option("inferSchema", "false")
        .schema(attribute_schema)
        .load("hdfs:///data/msd/audio/attributes/%s.attributes.csv" % i)
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
    features = (
        spark.read.format("com.databricks.spark.csv")
        .option("header", "false")
        .option("inferSchema", "false")
        .schema(feature_schema)
        .load("hdfs:///data/msd/audio/features/%s.csv" % i)
    )

    return features

# Iterate through each dataset.
for i in data_names:
    get_schema(i).show()