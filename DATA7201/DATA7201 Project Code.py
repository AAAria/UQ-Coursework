"""
DATA7201 Project
Ruyun Qi (44506065)

"""

import pyspark
from pyspark.sql import SparkSession, SQLContext, DataFrame
from pyspark.sql.types import *
from pyspark.sql.functions import mean, min, max, lit, col, first, last, lower, explode, szie, split, sum, count, countDistinct, desc
import pyspark.sql.functions as f
import pandas as pd

sc = pyspark.SparkContext("local") 
sqlContext = SQLContext(sc)

# Load data files in Oct 2020
data = spark.read.json("/data/ProjectDatasetFacebook/FBads-US-202010*")
data.printSchema()

# Select columns & Filter out NULL values
df = data.select("id", 
                 "ad_creative_body", 
                 "funding_entity",
                 col("spend.lower_bound").alias("lower_bound")),
         .filter('ad_creative_body IS NOT NULL')\
         .filter('funding_entity IS NOT NULL')\
         .dropDuplicates()


### Ad text analysis
# Lowercase ad text
df1 = df.select(lower(col("ad_creative_body")).alias("ad_creative_body"))\
        .dropDuplicates()

# Word frequency
word_freq = df1.withColumn('word', explode(split(col("ad_creative_body"), ' ')))\
               .groupBy('word')\
               .count() \
               .sort('count', ascending=False)\
               .toPandas()

# Install stop-words package
import sys
print(sys.executable)
!{sys.executable} -m pip install stop-words

# Filter out stop words
from stop_words import get_stop_words
stop_words = get_stop_words('en')

# Top 100 words
word_freq = word_freq[~word_freq['word'].isin(stop_words)].head(100)

# Export CSV file for making a word cloud
word_freq.to_csv('word_frequency.csv', encoding='utf-8', index=False)


### Ad ($5k+) analysis
# Group by funding entity
df2 = df.filter("lower_bound > 5000")\
        .groupBy('funding_entity')\
        .count() \
        .sort('count', ascending=False)\
        .toPandas()

# Export CSV file for data visualization in Tableau
df2.to_csv('lower_bound_5k.csv', encoding='utf-8', index=False)
