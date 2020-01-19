############################
# Blake List

############################
# PROCESSING

# Overview of the structure of the datasets, including file formats, data types, and the expected level of 
# parallelism that can be achieved from HDFS. ########## data types and parallelism.

hdfs dfs -ls -h /data/msd
hdfs dfs -du -h /data/msd

# We have four datasets: audio (12.3 GB), genre (30.1 MB), main (174.4 MB), and tasteprofile (490.4 MB) 
# replicated over HDFS four times. We will explore the structure of each one in more detail.

hdfs dfs -ls -h /data/msd/audio
hdfs dfs -du -h /data/msd/audio
hdfs dfs -head /data/msd/audio/attributes/msd-jmir-mfcc-all-v1.0.attributes.csv

# The audio dataset contains three directories: attributes (103.0 MB), features (12.2 GB), and statistics
# (40.3 MB). The attributes contain 3 csv files pertaining to each of the audio features (Rhythm Patterns, 
# Marsyas, and jMir). The features folder contains 13 subfolders each containing some number of compressed 
# csv part files for each audio feature. The attributes csv files range from just under 1 KB to just under 
# 35 KB, while the features csv files range from 35 MB to nearly 4 GB in size. The statistics folder contains 
# just one gzipped csv file for the sample properties compressed to 40 MB. Looking at each file more closely, 
# each of the audio feature csv's in attribute contain various numerical string ID data pertaining to each 
# feature. The level of parallelism we can hope to achieve with loading in the data is dependent on the 
# number of replications. In this case we have four replications of the data on HDFS so we can parallelise 
# loading, however, as the statistics and features data is gzipped, we cannot parallelise applying 
# transformations as we will only ever have one core working on one partition. Once the data is read in, we 
# can parallelise actions on the data based on the number of partitions (from the number of compressed files) 
# and the number of workers. As the attribute data is not compressed, we do not have this limitation.

hdfs dfs -ls -h /data/msd/genre
hdfs dfs -du -h /data/msd/genre
hdfs dfs -head /data/msd/genre/msd-MAGD-genreAssignment.tsv

# The genre dataset contains three tab separated (tsv) files pertaining to the MSD Allmusic genre and style 
# assignments. The three files are between 8 and 11 MB in size. Each tsv contains a song ID and a genre or
# style category such as 'Pop_Rock' or 'Metal_alternative', respectively. As none of the files in the genre
# folder are compressed, we can expect to parallelise the loading of the data by the number of replications
# on HDFS, and the transforming of the data dependent on the number of workers.

hdfs dfs -ls -h /data/msd/main
hdfs dfs -du -h /data/msd/main
hdfs dfs -du -h /data/msd/main/summary

# The main dataset contains the song ID, the track ID, the artist ID, and 51 other fields, such as the year, 
# title, artist tags, and various audio properties such as loudness, beat, tempo, and time signature. The 
# main folder contains a summary subfolder which has two gzipped csv files - analysis (55.9 MB) and
# metadata (118.5 MB). As these files are both compressed, we will be able to parallelise the loading of the
# data up to the number of repetitions on HDFS, but we will only have one worker applying transformations 
# until they are unzipped.

hdfs dfs -ls -h /data/msd/tasteprofile
hdfs dfs -du -h /data/msd/tasteprofile
hdfs dfs -du -h /data/msd/tasteprofile/mismatches
hdfs dfs -du -h /data/msd/tasteprofile/triplets.tsv

# The tasteprofile contains two folders - mismatches (2.0 MB), and triplets.tsv (488.4 MB). The
# mismatches folder contains two txt files, one for the identifiers and names of the mismatched songs, and 
# the other for the manually added matches. The triplets.tsv folder contains eight gzipped part files for the
# implicit feedback user data. We would be able to parallelise the loading up to the number of replication,
# as previously mentioned, and the transformations would only be performed by one worker until the data in 
# unzipped.

##########################

# Usefulness of the repartition method.

# The repartition algorithm does a full shuffle and creates new partitions with data that is distributed 
# evenly. As Spark can only run one concurrent task for every partition of an RDD, up to the number of 
# cores used in the cluster, we would want to repartition the data to have about 2-4 times the number of 
# cores. Repartitioning the data can be useful if the goal is to create a larger number of partitions for 
# processing. As a full data shuffle is an expensive process, we must decide whether it is worth doing if 
# there are a large number of rows in a particular dataset.

##########################

# Number of rows in each dataset.

# /msd/audio/attributes
for filename in `hdfs dfs -ls -R /data/msd/audio/attributes | awk '{print $NF}' | grep 'csv' | tr -s ' '`;
    do 
      echo $filename `hdfs dfs -cat $filename | wc -l` 
    done;
#/data/msd/audio/attributes/msd-jmir-area-of-moments-all-v1.0.attributes.csv 21
#/data/msd/audio/attributes/msd-jmir-lpc-all-v1.0.attributes.csv 21
#/data/msd/audio/attributes/msd-jmir-methods-of-moments-all-v1.0.attributes.csv 11
#/data/msd/audio/attributes/msd-jmir-mfcc-all-v1.0.attributes.csv 27
#/data/msd/audio/attributes/msd-jmir-spectral-all-all-v1.0.attributes.csv 17
#/data/msd/audio/attributes/msd-jmir-spectral-derivatives-all-all-v1.0.attributes.csv 17
#/data/msd/audio/attributes/msd-marsyas-timbral-v1.0.attributes.csv 125
#/data/msd/audio/attributes/msd-mvd-v1.0.attributes.csv 421
#/data/msd/audio/attributes/msd-rh-v1.0.attributes.csv 61
##/data/msd/audio/attributes/msd-rp-v1.0.attributes.csv 1441
#/data/msd/audio/attributes/msd-ssd-v1.0.attributes.csv 169
#/data/msd/audio/attributes/msd-trh-v1.0.attributes.csv 421
#/data/msd/audio/attributes/msd-tssd-v1.0.attributes.csv 1177

# /msd/audio/features
for filename in `hdfs dfs -ls -R /data/msd/audio/features | awk '{print $NF}' | grep 'csv' | grep -v 'part' | tr -s ' '`;
    do 
      echo $filename `hdfs dfs -cat $filename/* | zcat | wc -l` 
    done;
#/data/msd/audio/features/msd-jmir-area-of-moments-all-v1.0.csv 994623
#/data/msd/audio/features/msd-jmir-lpc-all-v1.0.csv 994623
#/data/msd/audio/features/msd-jmir-methods-of-moments-all-v1.0.csv 994623
#/data/msd/audio/features/msd-jmir-mfcc-all-v1.0.csv 994623
#/data/msd/audio/features/msd-jmir-spectral-all-all-v1.0.csv 994623
#/data/msd/audio/features/msd-jmir-spectral-derivatives-all-all-v1.0.csv 994623
#/data/msd/audio/features/msd-marsyas-timbral-v1.0.csv 995001
#/data/msd/audio/features/msd-mvd-v1.0.csv 994188
#/data/msd/audio/features/msd-rh-v1.0.csv 994188
#/data/msd/audio/features/msd-rp-v1.0.csv 994188
#/data/msd/audio/features/msd-ssd-v1.0.csv 994188
#/data/msd/audio/features/msd-trh-v1.0.csv 994188
#/data/msd/audio/features/msd-tssd-v1.0.csv 994188

# /msd/audio/statistics
for filename in `hdfs dfs -ls -R /data/msd/audio/statistics | awk '{print $NF}' | grep 'csv' | grep -v 'part' | tr -s ' '`;
    do 
      echo $filename `hdfs dfs -cat $filename | zcat | wc -l` 
    done;
#/data/msd/audio/statistics/sample_properties.csv.gz 992866

# /msd/genre
for filename in `hdfs dfs -ls -R /data/msd/genre | awk '{print $NF}' | grep 'tsv' | tr -s ' '`;
   do
      echo $filename `hdfs dfs -cat $filename | wc -l` 
   done;
#/data/msd/genre/msd-MAGD-genreAssignment.tsv 422714
#/data/msd/genre/msd-MASD-styleAssignment.tsv 273936
#/data/msd/genre/msd-topMAGD-genreAssignment.tsv 406427

# /msd/main/summary
for filename in `hdfs dfs -ls -R /data/msd/main | awk '{print $NF}' | grep 'csv' | grep -v 'part' | tr -s ' '`;
   do
      echo $filename `hdfs dfs -cat $filename | zcat | wc -l` 
   done;
#/data/msd/main/summary/analysis.csv.gz 1000001
#/data/msd/main/summary/metadata.csv.gz 1000001

# /msd/tasteprofile
for filename in `hdfs dfs -ls -R /data/msd/tasteprofile | awk '{print $NF}' | grep 'tsv' | grep -v 'part' | tr -s ' '`;
   do
      echo $filename `hdfs dfs -cat $filename/* | zcat | wc -l` 
   done;
#/data/msd/tasteprofile/triplets.tsv 48373586

# Within the audio dataset, the attributes files small numbers of rows ranging from 11 to 1177 depending on 
# the type of attribute. The audio features and statistics files have around 994188 and 992866 rows, 
# respectively, just less than the total number of unique songs in the dataset. The three genre datasets, 
# MAGD genre, MASD style, and topMAGD  genre each have 422714, 273936, and 406427 rows, respectively. Lastly, 
# the Taste Profile triplets dataset has 48373586 lines.

############################

# See 'processing.py'

##########################

# Show the column names and types of each attribute dataset.
for filename in `hdfs dfs -ls -R /data/msd/audio/attributes | awk '{print $NF}' | grep 'csv' | tr -s ' '`;
   do
      echo $filename `hdfs dfs -head $filename` 
   done;
   