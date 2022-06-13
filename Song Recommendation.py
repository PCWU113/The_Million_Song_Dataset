"""
Assignment 2 Song Recommendation

@author: Pengcheng Wu

Reference : some codes changed from the Credit Fraud, MovieLens Logistic Regression, SparkLMlib and Assignment Examples
"""

------------------------------------------------------------------------------------------------------------------------------------
####################################################### Songs Recommendation #######################################################
------------------------------------------------------------------------------------------------------------------------------------

# Python and pyspark modules required

import sys

from pyspark import SparkContext
from pyspark.sql import SparkSession
from pyspark.sql.types import *

from pyspark.sql import functions as F

from pyspark.ml.feature import StringIndexer
from pyspark.sql.window import Window

# Required to allow the file to be submitted and run using spark-submit instead
# of using pyspark interactively

spark = SparkSession.builder.getOrCreate()
sc = SparkContext.getOrCreate()

# Compute suitable number of partitions

conf = sc.getConf()

N = int(conf.get("spark.executor.instances"))
M = int(conf.get("spark.executor.cores"))
partitions = 4 * N * M

# Load
#define mismatch
mismatches_schema = StructType([
  StructField("song_id", StringType(), True),
  StructField("song_artist", StringType(), True),
  StructField("song_title", StringType(), True),
  StructField("track_id", StringType(), True),
  StructField("track_artist", StringType(), True),
  StructField("track_title", StringType(), True)
])

#load the text 
with open("/scratch-network/courses/2022/DATA420-22S1/data/msd/tasteprofile/mismatches/sid_matches_manually_accepted.txt", "r") as f:
  lines = f.readlines()
  sid_matches_manually_accepted = []
  for line in lines:
    if line.startswith("< ERROR: "):
      a = line[10:28]
      b = line[29:47]
      c, d = line[49:-1].split("  !=  ")
      e, f = c.split("  -  ")
      g, h = d.split("  -  ")
      sid_matches_manually_accepted.append((a, e, f, b, g, h))

matches_manually_accepted = spark.createDataFrame(sc.parallelize(sid_matches_manually_accepted, 8), schema=mismatches_schema)


with open("/scratch-network/courses/2022/DATA420-22S1/data/msd/tasteprofile/mismatches/sid_mismatches.txt", "r") as f:
  lines = f.readlines()
  sid_mismatches = []
  for line in lines:
    if line.startswith("ERROR: "):
      a = line[8:26]
      b = line[27:45]
      c, d = line[47:-1].split("  !=  ")
      e, f = c.split("  -  ")
      g, h = d.split("  -  ")
      sid_mismatches.append((a, e, f, b, g, h))

mismatches = spark.createDataFrame(sc.parallelize(sid_mismatches, 64), schema=mismatches_schema)

#define triplets schema
triplets_schema = StructType([
  StructField("user_id", StringType(), True),
  StructField("song_id", StringType(), True),
  StructField("plays", IntegerType(), True)
])
triplets = (
  spark.read.format("csv")
  .option("header", "false")
  .option("delimiter", "\t")
  .option("codec", "gzip")
  .schema(triplets_schema)
  .load("hdfs:///data/msd/tasteprofile/triplets.tsv/")
  .cache()
)

# filter the triplets
mismatches_not_accepted = mismatches.join(matches_manually_accepted, on="song_id", how="left_anti")
triplets_not_mismatched = triplets.join(mismatches_not_accepted, on="song_id", how="left_anti")
triplets_not_mismatched.show(10,50)
"""
+------------------+----------------------------------------+-----+
|           song_id|                                 user_id|plays|
+------------------+----------------------------------------+-----+
|SOAAADE12A6D4F80CC|1c21e65e0e67ecc08f2da4856decda7e6cf6de2e|    1|
|SOAAADE12A6D4F80CC|21cd3a6311fc6820e086b864139e631ec602f69e|    1|
|SOAAADE12A6D4F80CC|edfc4dea143a03f061d7f834775b2d8119094cae|    1|
|SOAAADE12A6D4F80CC|a9777925ab522473c1bc01c1f0b9d64f24fc69dc|    1|
|SOAAADE12A6D4F80CC|ae6b5e9dbfdd799f23fb1a5887de42d02338e2e4|    1|
|SOAAADE12A6D4F80CC|f4058a3849ba9ef1a23884a5274ec92cb9d52649|    3|
|SOAAADE12A6D4F80CC|335253b4d6f0baedd75bcffc834d2862aec29289|    1|
|SOAAADE12A6D4F80CC|992fa95473cc053aa8cd689bcadec43d17cdc760|    1|
|SOAAADE12A6D4F80CC|736f40dcc9e45e2f03ffc0edb90ebb1954d8f9d6|    2|
|SOAAADF12A8C13DF62|b754639360a07a22a7155fc2c78e53ae16bf6e8f|    1|
+------------------+----------------------------------------+-----+
"""
triplets_not_mismatched = triplets_not_mismatched.repartition(partitions).cache()

#Q1
## (a)
def get_user_counts(triplets):
  return (
    triplets
    .groupBy("user_id")
    .agg(
      F.count(F.col("song_id")).alias("song_count"),
      F.sum(F.col("plays")).alias("play_count"),
    )
    .orderBy(F.col("play_count").desc())
  )

def get_song_counts(triplets):
  return (
    triplets
    .groupBy("song_id")
    .agg(
      F.count(F.col("user_id")).alias("user_count"),
      F.sum(F.col("plays")).alias("play_count"),
    )
    .orderBy(F.col("play_count").desc())
  )
  
# User statistics

user_counts = (
  triplets_not_mismatched
  .groupBy("user_id")
  .agg(
    F.count(F.col("song_id")).alias("song_count"),
    F.sum(F.col("plays")).alias("play_count"),
  )
  .orderBy(F.col("play_count").desc())
)
# user_counts.cache()
user_counts.count() #1019318
user_counts.show(10, False)
"""
+----------------------------------------+----------+----------+
|user_id                                 |song_count|play_count|
+----------------------------------------+----------+----------+
|093cb74eb3c517c5179ae24caf0ebec51b24d2a2|195       |13074     |
|119b7c88d58d0c6eb051365c103da5caf817bea6|1362      |9104      |
|3fa44653315697f42410a30cb766a4eb102080bb|146       |8025      |
|a2679496cd0af9779a92a13ff7c6af5c81ea8c7b|518       |6506      |
|d7d2d888ae04d16e994d6964214a1de81392ee04|1257      |6190      |
|4ae01afa8f2430ea0704d502bc7b57fb52164882|453       |6153      |
|b7c24f770be6b802805ac0e2106624a517643c17|1364      |5827      |
|113255a012b2affeab62607563d03fbdf31b08e7|1096      |5471      |
|99ac3d883681e21ea68071019dba828ce76fe94d|939       |5385      |
|6d625c6557df84b60d90426c0116138b617b9449|1307      |5362      |
+----------------------------------------+----------+----------+

"""
statistics = (
  user_counts
  .select("song_count", "play_count")
  .describe()
  .toPandas()
  .set_index("summary")
  .rename_axis(None)
  .T
)
print(statistics)
"""
             count                mean              stddev min    max
song_count  1019318   44.92720721109605   54.91113199747355   3   4316
play_count  1019318  128.82423149596102  175.43956510304744   3  13074

"""

user_counts.approxQuantile("song_count", [0.0, 0.25, 0.5, 0.75, 1.0], 0.05) #[3.0, 14.0, 26.0, 50.0, 4316.0]

user_counts.approxQuantile("play_count", [0.0, 0.25, 0.5, 0.75, 1.0], 0.05) # [3.0, 27.0, 63.0, 137.0, 13074.0]

# Song statistics

song_counts = (
  triplets_not_mismatched
  .groupBy("song_id")
  .agg(
    F.count(F.col("user_id")).alias("user_count"),
    F.sum(F.col("plays")).alias("play_count"),
  )
  .orderBy(F.col("play_count").desc())
)
# song_counts.cache()
song_counts.count() #378310

song_counts.show(10, False)
"""
+------------------+----------+----------+
|song_id           |user_count|play_count|
+------------------+----------+----------+
|SOBONKR12A58A7A7E0|84000     |726885    |
|SOSXLTC12AF72A7F54|80656     |527893    |
|SOEGIYH12A6D4FC0E3|69487     |389880    |
|SOAXGDH12A8C13F8A1|90444     |356533    |
|SONYKOW12AB01849C9|78353     |292642    |
|SOPUCYA12A8C13A694|46078     |274627    |
|SOUFTBI12AB0183F65|37642     |268353    |
|SOVDSJC12A58A7A271|36976     |244730    |
|SOOFYTN12A6D4F9B35|40403     |241669    |
|SOHTKMO12AB01843B0|46077     |236494    |
+------------------+----------+----------+

"""

statistics = (
  song_counts
  .select("user_count", "play_count")
  .describe()
  .toPandas()
  .set_index("summary")
  .rename_axis(None)
  .T
)
print(statistics)
"""
 count                mean              stddev min     max
user_count  378310  121.05181200602681    748.648978373691   1   90444
play_count  378310   347.1038513388491  2978.6053488382263   1  726885

"""
song_counts.approxQuantile("user_count", [0.0, 0.25, 0.5, 0.75, 1.0], 0.05) # [1.0, 4.0, 12.0, 44.0, 90444.0]

song_counts.approxQuantile("play_count", [0.0, 0.25, 0.5, 0.75, 1.0], 0.05) #[1.0, 7.0, 28.0, 124.0, 726885.0]

song_unique = triplets_not_mismatched.select(F.col('song_id')).distinct()
song_counts = song_unique.count()# 378310
user_unique = triplets_not_mismatched.select(F.col('user_id')).distinct()
user_counts = user_unique.count() #  1019318

 ##(b)
 # Filter the most active user
song_list_most_active = (
	triplets_not_mismatched
	.where(
		triplets_not_mismatched.user_id == "093cb74eb3c517c5179ae24caf0ebec51b24d2a2"
	)
)

song_list_most_active.count() # 195
user_active = (triplets_not_mismatched
                .groupBy('user_id')
                .agg(F.count('song_id').alias('song_count'),
                     F.sum('plays').alias('play_count'))
                
                .withColumn('percent', F.col('song_count')/ song_counts)
                .orderBy('song_count',ascending = True)
                )

user_active.show(5,False)             
"""
+----------------------------------------+----------+----------+---------------------+
|user_id                                 |song_count|play_count|percent              |
+----------------------------------------+----------+----------+---------------------+
|e3ae5528edb8f274737cb55b856b46a069d1cdf7|3         |7         |7.930004493669212E-6 |
|5bdc88d63c45222aa81e92707c44a4429e10853a|3         |3         |7.930004493669212E-6 |
|0a7929546d6232621a1c93e1a83969c5157e7e40|3         |3         |7.930004493669212E-6 |
|61943ef983a641107c746cc8178bcbda197f1dcb|3         |3         |7.930004493669212E-6 |
|b4cf50e9e2105445ffcc42c8f821e24ee27fc344|4         |5         |1.0573339324892285E-5|
+----------------------------------------+----------+----------+---------------------+

""" 


song_popularity = (triplets_not_mismatched
         .groupBy('song_id')  
         .agg(F.count('user_id').alias('user_count'), 
              F.sum('plays').alias('play_count'))
          .withColumn('percent_1', F.col('user_count')/user_counts )
          .orderBy('user_count', ascending = True) 
          )
song_popularity.show(5,False)
"""
+------------------+----------+----------+--------------------+
|song_id           |user_count|play_count|percent_1           |
+------------------+----------+----------+--------------------+
|SOILWJR12A6D4FAACB|1         |1         |9.810481125615363E-7|
|SOGZOEV12AC3DF6B5A|1         |1         |9.810481125615363E-7|
|SODBWXQ12A58A7A16E|1         |2         |9.810481125615363E-7|
|SOBLTJM12AB0186799|1         |1         |9.810481125615363E-7|
|SOSWBJQ12AB017FB95|1         |1         |9.810481125615363E-7|
+------------------+----------+----------+--------------------+
"""
## (c) Visualize the distribution 1. song popularity; 2. User activity 
#plot
user_activity_pandas = user_active.toPandas()
song_popularity_pandas = song_popularity.toPandas()

import numpy as np
from collections import OrderedDict
import pandas as pd
import matplotlib; matplotlib.use('agg')
import matplotlib.pyplot as plt
import os
import io
"""
def show_axes_legend(a, loc="upper right"):

  handles, labels = a.get_legend_handles_labels()
  by_label = OrderedDict(zip(labels, handles))
  legend = a.legend(
    by_label.values(), by_label.keys(),
    borderpad=0.5,
    borderaxespad=0,
    fancybox=False,
    edgecolor="black",
    framealpha=1,
    loc=loc,
    fontsize="x-small",
    ncol=1
  )
  frame = legend.get_frame().set_linewidth(0.75)
"""
plt.subplots(dpi=300, figsize=(10, 5))



#The distribution of user activity
x = user_activity_pandas["play_count"]

plt.figure(figsize=(16,9))
plt.hist(x,bins=300,color="#87CEFA")
plt.xlabel("count play")
plt.ylabel('# users ')
plt.title("Distribution of user activity")
plt.tight_layout()
plt.savefig(os.path.join(os.path.expanduser("~/user_activity_dtb.png")), bbox_inches="tight") # save as png and view in windows
plt.close()

#The distribution of song popularity
x1 = song_popularity_pandas["play_count"]

plt.figure(figsize=(16,9))
plt.hist(x1,bins=300,color="#87CEFA")
plt.xlabel("count play")
plt.ylabel('# songs ')
plt.title("Distribution of song popularity")
plt.tight_layout()
plt.savefig(os.path.join(os.path.expanduser("~/song_popularity_dtb.png")), bbox_inches="tight") # save as png and view in windows
plt.close()



# (d) Collaborative filtering determines similar users and songs based on their combined play history.
import numpy as np
N = user_active.toPandas().quantile(0.25)
print(N)
"""
song_count    15.00000
play_count    32.00000
percent        0.00004
Name: 0.25, dtype: float64
"""
M = song_popularity.toPandas().quantile(0.25)
print(M)
"""
user_count    4.000000
play_count    8.000000
percent_1     0.000004
Name: 0.25, dtype: float64
"""
# Songs having been played only a few times & users who have only listened to a few songs will not contribute much information to the overall dataset and be unlikely to be recommended.
user_less = user_active.where(user_active.play_count < 32).orderBy('play_count', ascending = False)
user_less.count() #254739
user_less.show(5,False)
"""
+----------------------------------------+----------+----------+---------------------+
|user_id                                 |song_count|play_count|percent              |
+----------------------------------------+----------+----------+---------------------+
|5bdc88d63c45222aa81e92707c44a4429e10853a|3         |3         |7.930004493669212E-6 |
|0a7929546d6232621a1c93e1a83969c5157e7e40|3         |3         |7.930004493669212E-6 |
|61943ef983a641107c746cc8178bcbda197f1dcb|3         |3         |7.930004493669212E-6 |
|61fb0cd2b37f13b0b1f3526a3089119af39acaee|4         |4         |1.0573339324892285E-5|
|8937e63ef2ed18b12e3873a16ec5ff73c6ee43eb|4         |4         |1.0573339324892285E-5|
+----------------------------------------+----------+----------+---------------------+
"""
song_less = song_popularity.where(song_popularity.play_count < 8).orderBy('play_count', ascending = True)
song_less.count() #90750
song_less.show(5,False)
"""
+------------------+----------+----------+--------------------+
|song_id           |user_count|play_count|percent_1           |
+------------------+----------+----------+--------------------+
|SOZSJMF12A6D4F750A|1         |1         |9.810481125615363E-7|
|SOFBIYX12AB0180186|1         |1         |9.810481125615363E-7|
|SOGQSAB12AC960D96B|1         |1         |9.810481125615363E-7|
|SOOWSCO12A6D4F9A39|1         |1         |9.810481125615363E-7|
|SOUYAMX12AB0181300|1         |1         |9.810481125615363E-7|
+------------------+----------+----------+--------------------+
"""
user_song = triplets_not_mismatched.join(user_less, on = 'user_id', how = 'left_anti').join(song_less,on = 'song_id', how = 'left_anti')
user_song.count() # 41978747 
user_song.show(5,False)
"""
+------------------+----------------------------------------+-----+
|song_id           |user_id                                 |plays|
+------------------+----------------------------------------+-----+
|SOAAADE12A6D4F80CC|ae6b5e9dbfdd799f23fb1a5887de42d02338e2e4|1    |
|SOAAADE12A6D4F80CC|a9777925ab522473c1bc01c1f0b9d64f24fc69dc|1    |
|SOAAADE12A6D4F80CC|f4058a3849ba9ef1a23884a5274ec92cb9d52649|3    |
|SOAAADE12A6D4F80CC|21cd3a6311fc6820e086b864139e631ec602f69e|1    |
|SOAAADE12A6D4F80CC|736f40dcc9e45e2f03ffc0edb90ebb1954d8f9d6|2    |
+------------------+----------------------------------------+-----+

"""


# Limiting 
user_song_count_threshold = 34
song_user_count_threshold = 5
triplets_limited = triplets_not_mismatched

triplets_limited = (
  triplets_limited
  .join(
    triplets_limited.groupBy("user_id").count().where(col("count") > user_song_count_threshold).select("user_id"),
    on="user_id",
    how="inner"
  )
)

triplets_limited = (
  triplets_limited
  .join(
    triplets_limited.groupBy("song_id").count().where(col("count") > user_song_count_threshold).select("song_id"),
    on="song_id",
    how="inner"
  )
)

triplets_limited.cache()
triplets_limited.count() # 31933097


(
  triplets_limited
  .agg(
    countDistinct(col("user_id")).alias('user_count'),
    countDistinct(col("song_id")).alias('song_count')
  )
  .toPandas()
  .T
  .rename(columns={0: "value"})
)
"""
             value
user_count  393885
song_count  101341

"""


print(get_user_counts(triplets_limited).approxQuantile("song_count", [0.0, 0.25, 0.5, 0.75, 1.0], 0.05)) # [1.0, 42.0, 59.0, 90.0, 2875.0]
print(get_song_counts(triplets_limited).approxQuantile("user_count", [0.0, 0.25, 0.5, 0.75, 1.0], 0.05)) # [35.0, 53.0, 103.0, 266.0, 53629.0]

# (e) Split the user-song plays into training and test sets.
# -----------------------------------------------------------------------------

# Encoding
user_id_indexer = StringIndexer(inputCol="user_id", outputCol="user_id_encoded")
song_id_indexer = StringIndexer(inputCol="song_id", outputCol="song_id_encoded")

user_id_indexer_model = user_id_indexer.fit(triplets_limited)
song_id_indexer_model = song_id_indexer.fit(triplets_limited)

triplets_limited = user_id_indexer_model.transform(triplets_limited)
triplets_limited = song_id_indexer_model.transform(triplets_limited)
triplets_limited.show(5,False)


# -----------------------------------------------------------------------------

# Splitting
train_us, test_us = triplets_limited.randomSplit([0.7,0.3])

test_not_training = test_us.join(train_us, on = 'user_id', how = 'left_anti' )

print('train set count: ', train_us.count()) # 29384472
print('test set count: ', test_us.count()) # 12594275
print('test_not_training', test_not_training) 

counts = test_not_training.groupBy("user_id").count().toPandas().set_index("user_id")["count"].to_dict()
temp = (
  test_not_training
  .withColumn("id", monotonically_increasing_id())
  .withColumn("random", rand())
  .withColumn(
    "row",
    row_number()
    .over(
      Window
      .partitionBy("user_id")
      .orderBy("random")
    )
  )
)

for k, v in counts.items():
  temp = temp.where((F.col("user_id") != k) | (F.col("row") < v * 0.7))

temp = temp.drop("id", "random", "row")
temp.cache()
temp.show(50, False)
"""
+-------+-------+-----+---------------+---------------+
|user_id|song_id|plays|user_id_encoded|song_id_encoded|
+-------+-------+-----+---------------+---------------+
+-------+-------+-----+---------------+---------------+
"""
train_us = train_us.union(temp.select(train_us.columns))
test_us = test_us.join(temp, on=["user_id", "song_id"], how="left_anti")
test_not_training = test_us.join(train_us, on="user_id", how="left_anti")
print(f"training_us:      {train_us.count()}") # 29385118
print(f"test_us:        {test_us.count()}") #12593629
print(f"test_not_training: {test_not_training.count()}") # 0
# Ensuring every user in the test set has some user-song plays in the training set too. So when test not training =0 will ensure it.


# Q2.(a) Use the spark.ml library to train an implicit matrix factorization model using Alternating Least Squares (ALS).
from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.mllib.evaluation import RankingMetrics
# withColumn---plays as rating
train_us_r = train_us.withColumn('rating',F.col('plays'))
test_us_r = test_us.withColumn('rating', F.col('plays'))

# train an implicit matrics factorization model
als = ALS(maxIter = 5, regParam = 0.01, userCol = 'user_id_encoded', itemCol = 'song_id_encoded', ratingCol = 'rating', seed=1000,implicitPrefs = True)
ALS_Model = als.fit(train_us_r)

# Prediction --- test set
ALS_predictions = ALS_Model.transform(test_us_r)
ALS_predictions.show()
"""
+------------------+--------------------+-----+---------------+---------------+-----------+-------------------+---+------+-----------+
|           song_id|             user_id|plays|user_id_encoded|song_id_encoded|         id|             random|row|rating| prediction|
+------------------+--------------------+-----+---------------+---------------+-----------+-------------------+---+------+-----------+
|SOHTKMO12AB01843B0|da7f88c65fb88a828...|    1|           27.0|           12.0|68720305732| 0.7944699605271831|884|     1|  0.0720131|
|SOHTKMO12AB01843B0|e8f6a8d06b0096737...|    1|          137.0|           12.0|68720305887| 0.5702513172440506|469|     1|0.016403496|
|SOHTKMO12AB01843B0|fa5d9eddc010bc3fc...|    8|          145.0|           12.0|68720313567|0.49577806479725617|403|     8|  0.9327209|
|SOHTKMO12AB01843B0|3d257235eadd853a6...|   13|          448.0|           12.0|68720329448|  0.776072746006292|479|    13| 0.07680438|
|SOHTKMO12AB01843B0|4ae408d1be742c745...|    1|          680.0|           12.0|68720306862| 0.1644212128630177| 96|     1|0.025611242|
|SOHTKMO12AB01843B0|53bf5530eda7bfadb...|    1|          800.0|           12.0|68720299399| 0.6416842556900657|348|     1| 0.17292246|
|SOHTKMO12AB01843B0|af727e1aca1a585f1...|    6|         1180.0|           12.0|68720322953|0.08248799132919249| 36|     6| 0.42460066|
|SOHTKMO12AB01843B0|7eb254c0080756060...|    1|         1777.0|           12.0|68720325065| 0.4014564742378688|184|     1| 0.42969692|
|SOHTKMO12AB01843B0|07b8418f4425c12d8...|   11|         1857.0|           12.0|68720316134|  0.530598101652246|227|    11|  0.3115755|
|SOHTKMO12AB01843B0|fa7d48e7e13a79c8e...|    5|         2272.0|           12.0|68720336454| 0.9843885952504322|417|     5| 0.27200133|
|SOHTKMO12AB01843B0|ce73467b148a4dad6...|   14|         2405.0|           12.0|68720300605|0.14313609152092577| 71|    14| 0.30437008|
|SOHTKMO12AB01843B0|70d1b8f61b9b79c4f...|   13|         2439.0|           12.0|68720329969| 0.3980929606851257|172|    13|  0.7310012|
|SOHTKMO12AB01843B0|103ee21d642c7dbcd...|    7|         2573.0|           12.0|68720298770|  0.216449454598725| 98|     7|  0.6937078|
|SOHTKMO12AB01843B0|362c24cc35bd19451...|    1|         2817.0|           12.0|68720321753| 0.6183092797586665|231|     1| 0.25669345|
|SOHTKMO12AB01843B0|96d9f022ede95299f...|    2|         4571.0|           12.0|68720332897|0.30085773604904364|112|     2|  0.6848034|
|SOHTKMO12AB01843B0|d41d1df73de6a21a5...|   13|         4708.0|           12.0|68720315626| 0.9881489682917871|344|    13|   0.839746|
|SOHTKMO12AB01843B0|7e0f32e28e266a1f7...|    1|         4735.0|           12.0|68720302333| 0.4990138810884782|168|     1| 0.45961463|
|SOHTKMO12AB01843B0|a3ec2640952f8c3ef...|    2|         4791.0|           12.0|68720300169|0.09337363128724252| 36|     2| 0.14224938|
|SOHTKMO12AB01843B0|56fb68a9e631a12c7...|    1|         4912.0|           12.0|68720311912|0.14309234439077811| 61|     1| 0.46742523|
|SOHTKMO12AB01843B0|4d9fd735d6e9c44e5...|   10|         6577.0|           12.0|68720327107|0.24185556348410953| 91|    10| 0.99873847|
+------------------+--------------------+-----+---------------+---------------+-----------+-------------------+---+------+-----------+

"""
# RMSE evaluates  feedback
evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating", predictionCol="prediction")
evaluator.evaluate(ALS_predictions.filter(F.col('prediction') != np.NaN))  #7.610432634025173


# Implicit feedback needs to be evaulated using ranking metrics
def extract_songs_top_k(x, k):
  x = sorted(x, key=lambda x: -x[1])
  return [x[0] for x in x][0:k]

extract_songs_top_k_udf = F.udf(lambda x: extract_songs_top_k(x, k), ArrayType(IntegerType()))

def extract_songs(x):
  x = sorted(x, key=lambda x: -x[1])
  return [x[0] for x in x]

extract_songs_udf = F.udf(lambda x: extract_songs(x), ArrayType(IntegerType()))


#(b) Generate recommendations for selected users from ALS
k = 10 
users = test_us.select(["user_id_encoded"]).distinct().limit(5)
users.cache()
users.show(5,False)
"""
+---------------+
|user_id_encoded|
+---------------+
|59604.0        |
|456306.0       |
|11633.0        |
|111928.0       |
|3268.0         |
+---------------+

"""
recommendations = ALS_Model.recommendForUserSubset(users, k)
recommendations.cache()
recommendations.show(5,False)
"""
+---------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
|user_id_encoded|recommendations                                                                                                                                                                             |
+---------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
|11241          |[{16, 0.0777009}, {193, 0.07621274}, {75, 0.0732392}, {302, 0.06870155}, {265, 0.060362175}, {29, 0.059879217}, {169, 0.05951733}, {256, 0.05895741}, {249, 0.05881747}, {536, 0.058548953}]|
|2127           |[{0, 0.2832091}, {2, 0.27200344}, {185, 0.2625257}, {6, 0.25039193}, {52, 0.23787771}, {10, 0.227658}, {24, 0.22088781}, {42, 0.21435928}, {56, 0.20693623}, {92, 0.19833961}]              |
|290106         |[{20, 0.16079102}, {24, 0.15618537}, {14, 0.1482407}, {11, 0.14802147}, {47, 0.123092614}, {35, 0.12271632}, {42, 0.1154379}, {37, 0.11149337}, {201, 0.10613282}, {123, 0.10098977}]       |
|2953           |[{4, 0.3939914}, {17, 0.35243672}, {2, 0.34275442}, {9, 0.26155466}, {57, 0.25033796}, {1, 0.24540512}, {15, 0.24044004}, {6, 0.23184204}, {10, 0.2309246}, {97, 0.22819963}]               |
|180658         |[{46, 0.1281973}, {40, 0.11234736}, {32, 0.1089267}, {30, 0.10664217}, {78, 0.10263657}, {5, 0.097742796}, {60, 0.0969932}, {65, 0.09356195}, {103, 0.09274814}, {150, 0.08886694}]         |
+---------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+

"""
recommended_songs = (recommendations
                    .withColumn('recommended_song', extract_songs_top_k_udf(F.col('recommendations')))
                    .select('user_id_encoded','recommended_song')
                    )
recommended_songs.count()
recommended_songs.cache()
recommended_songs.show(5,False)
"""
+---------------+-----------------------------------------------+
|user_id_encoded|recommended_song                               |
+---------------+-----------------------------------------------+
|11241          |[16, 193, 75, 302, 265, 29, 169, 256, 249, 536]|
|2127           |[0, 2, 185, 6, 52, 10, 24, 42, 56, 92]         |
|290106         |[20, 24, 14, 11, 47, 35, 42, 37, 201, 123]     |
|2953           |[4, 17, 2, 9, 57, 1, 15, 6, 10, 97]            |
|180658         |[46, 40, 32, 30, 78, 5, 60, 65, 103, 150]      |
+---------------+-----------------------------------------------+


"""
# actual played
actual_played = (
                 triplets_limited
                 .where(triplets_limited.user_id_encoded
                 .isin([row.user_id_encoded for row in users.select(["user_id_encoded"])
                 .collect()]))
                 )
actual_played.show(5,False)
"""
+------------------+----------------------------------------+-----+---------------+---------------+
|song_id           |user_id                                 |plays|user_id_encoded|song_id_encoded|
+------------------+----------------------------------------+-----+---------------+---------------+
|SOADVUP12AB0185246|ada90427553803d898ede90db391b4a78f055003|2    |2127.0         |1664.0         |
|SOAHJUT12AF729CAE1|7792fc3c8d194410be37b77ed5285406469dfc65|1    |11241.0        |93557.0        |
|SOAPKPS12A8AE476C2|5b5428dd2f3d816b2af667f169b1f813c371e6d5|1    |2953.0         |14827.0        |
|SOBJKHY12A67020041|0112f93589e6eb10aaab4c62679245fc637fb2b1|2    |290106.0       |13666.0        |
|SOBLDDB12AB0183223|ada90427553803d898ede90db391b4a78f055003|1    |2127.0         |15107.0        |
+------------------+----------------------------------------+-----+---------------+---------------+

"""

# Relevant songs 
relevant_songs = (
  test_us_r
  .select(
    F.col("user_id_encoded").cast(IntegerType()),
    F.col("song_id_encoded").cast(IntegerType()),
    F.col("rating").cast(IntegerType())
  )
  .groupBy('user_id_encoded')
  .agg(
    F.collect_list(  # [(user,preference),...] <-- from test not the model
      F.array(
        F.col("song_id_encoded"),
        F.col("rating")
      )
    ).alias('relevance')
  )
  .withColumn("relevant_songs", extract_songs_udf(F.col("relevance"))) #[songs, songs,....]
  .select("user_id_encoded", "relevant_songs")
)
relevant_songs.cache()
relevant_songs.count()
print(pretty(relevant_songs.head().asDict()))
"""
{
  'relevant_songs': [
    46349,
    23759,
    82439,
    3207,
    21109,
    ...
  ],
  'user_id_encoded': 12
}


"""
# Filter the relevant_songs for selected users
users_list = (
              users
              .select('user_id_encoded')
              .rdd
              .flatMap(lambda x:x)
              .collect()
              )

relevant_songs_f = relevant_songs.filter(F.col('user_id_encoded').isin(users_list))
relevant_songs_f.show(5,False)
"""
+---------------+----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
|user_id_encoded|relevant_songs                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      |
+---------------+----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
|11241          |[248, 437, 317, 130488, 6937, 62708, 123718, 261669, 97319, 133764, 247953, 193420, 132137, 18426, 26697, 16254, 133060, 103359, 4770, 20620, 140422, 83408, 128294, 2630, 99692, 125235, 27390, 85339, 3186, 122849, 39397, 42942, 106682, 148411, 38650, 30119, 279328, 166726, 128240, 45874, 96024, 80075, 83597, 93277, 95029, 161124, 12699, 43863, 113291, 59354, 39092, 68306, 118963, 1844, 61535, 93557, 81795, 2236, 104450, 14585, 9384, 22925, 133439, 185433, 14984, 48016, 41966, 112704, 26299, 4431, 91274]                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        |
|2127           |[7158, 1234, 9157, 3679, 42807, 5738, 21843, 14099, 30139, 3798, 18916, 9118, 8503, 14379, 7217, 14051, 37835, 17519, 21188, 10617, 22305, 29170, 15806, 16056, 20058, 21485, 60243, 11678, 14115, 16452, 13117, 10357, 52866, 14278, 1664, 34590, 8226, 22356, 32746, 39269, 1151, 3088, 11569, 3425, 9663, 27465, 26559, 3263, 8246, 313, 2341, 5243, 12467, 11293, 1588, 3154, 39231, 5566, 13091, 812, 106358, 38425, 54415, 1341, 31456, 7933, 30385, 1679, 39335, 7435, 65336, 13683, 24971, 3197, 10197, 105799, 17432, 26845, 46698, 14324, 75315, 21401, 2565, 1983, 124728, 63220, 84227, 37372, 16833, 3381, 616, 8152, 4722, 9653, 2770, 91110, 25250, 10277, 39040, 53116, 66318, 35464, 16607, 52136, 1383, 22238, 59971, 39384, 34352, 3776, 2932, 38301, 89210, 50303, 11385, 6670, 15909, 28699, 1928, 45062, 4815, 2808, 75, 1593, 26883, 3005, 1034, 10035, 834, 15858, 4160, 23012, 12006, 7851, 43916, 31856, 54741, 17488, 3724, 28372, 159, 31296, 10877, 17452, 2433, 37565, 38054, 17711, 2261, 72607, 93854, 10278, 45316]|
|290106         |[19666, 939, 4229, 558, 2247, 43804, 45, 56, 25015, 71, 19064, 40936, 160, 3679, 4860, 2506, 631, 24602, 4208]                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      |
|2953           |[3666, 16244, 15417, 2742, 16389, 11637, 2061, 8857, 15971, 2821, 19415, 3433, 21234, 18622, 19183, 10943, 16617, 14659, 18037, 23138, 10671, 5641, 37235, 8302, 10497, 128609, 16083, 25485, 17577, 10712, 22238, 14784, 21884, 10632, 122149, 1994, 28100, 11946, 1123, 31717, 19791, 22257, 15668, 13269, 109967, 31544, 14672, 28771, 4094, 1501, 17146, 27484, 1983, 13530, 44798, 4633, 20347, 37882, 52745, 19173, 4268, 23098, 26568, 2299, 4819, 25760, 3858, 17103, 20344, 5809, 15686, 21087, 16196, 25475, 2669, 1647, 2349, 7403, 95323, 17991, 2877, 19711, 34890, 49300, 3980, 2364, 26290, 65829, 4816, 729, 52136, 1356, 1497, 11835, 23784, 11029, 15638, 713, 22447, 3128, 8256, 21803, 974, 19094, 3956, 7749, 3253, 113348, 113539, 26731, 6029, 18106, 15525, 44095, 14421, 5052, 6157, 22927, 24272]                                                                                                                                                                                                                         |
|180658         |[19314, 6923, 51454, 54684, 1243, 37565, 6581, 14516, 711, 4186, 52136, 48145, 4803, 3087, 43381, 5255, 214112, 69481, 11365, 92059, 2188, 23664, 36614, 35007, 2820, 341, 998, 4813, 37431, 1930, 84440]                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           |
+---------------+----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+


"""
# pick up user_is_encoded is 290106, responding to recommended_songs and relevant_songs list
rec_list = [19666, 939, 4229, 558, 2247, 43804, 45, 56, 25015, 71, 19064, 40936, 160, 3679, 4860, 2506, 631, 24602, 4208] 
rev_list = [20, 24, 14, 11, 47, 35, 42, 37, 201, 123]   


# Load Metadata
metadata = (
    spark.read
    .format("com.databricks.spark.csv")
    .option("header", "true")
    .option("inferSchema", "true")
    .load("hdfs:///data/msd/main/summary/metadata.csv.gz")
)
print(pretty(metadata.head().asDict()))
"""
{
  'analyzer_version': None,
  'artist_7digitalid': 4069,
  'artist_familiarity': 0.6498221002008776,
  'artist_hotttnesss': 0.3940318927141434,
  'artist_id': 'ARYZTJS1187B98C555',
  'artist_latitude': None,
  'artist_location': None,
  'artist_longitude': None,
  'artist_mbid': '357ff05d-848a-44cf-b608-cb34b5701ae5',
  'artist_name': 'Faster Pussy cat',
  'artist_playmeid': '44895',
  'genre': None,
  'idx_artist_terms': 0,
  'idx_similar_artists': 0,
  'release': 'Monster Ballads X-Mas',
  'release_7digitalid': 633681,
  'song_hotttnesss': '0.5428987432910862',
  'song_id': 'SOQMMHC12AB0180CB8',
  'title': 'Silent Night',
  'track_7digitalid': '7032331'
}
"""

temp_md = (
    metadata
    .join(triplets_limited, on='song_id', how='left')
    .filter(F.col('song_id_encoded').isin(rec_list))   
    .sort(F.col('song_id_encoded')))

temp_md.select("song_id_encoded", "artist_name", "artist_location","genre", "release","song_hotttnesss","title").distinct().show(10,False)
"""
+---------------+----------------------------+--------------------+-----+-----------------------------+------------------+----------------------------------------+
|song_id_encoded|artist_name                 |artist_location     |genre|release                      |song_hotttnesss   |title                                   |
+---------------+----------------------------+--------------------+-----+-----------------------------+------------------+----------------------------------------+
|19666.0        |Beats Antique               |San Francisco CA    |null |Tribal Derivations           |0.6342037231574836|Derivation                              |
|558.0          |Coldplay                    |null                |null |In My Place                  |0.8102636131560579|One I Love                              |
|939.0          |Creedence Clearwater Revival|El Cerrito, CA      |null |Chronicle: 20 Greatest Hits  |0.8241522295913459|Who'll Stop The Rain                    |
|56.0           |Kid Cudi Vs Crookers        |Cleveland, Ohio     |null |R&B Yearbook                 |0.9571827746417184|Day 'N' Nite                            |
|25015.0        |Matt Costa                  |Huntington Beach, CA|null |Songs We Sing                |0.8832035878217067|Wash Away                               |
|71.0           |California Swag District    |null                |null |Teach Me How To Dougie       |0.590674195137202 |Teach Me How To Dougie                  |
|2247.0         |Regina Spektor              |Moscow, Russia      |null |Far                          |0.7331593637491108|The Sword & the Pen (Non-Album Track)   |
|43804.0        |Midlake                     |Denton, TX          |null |The Trials Of Van Occupanther|0.6874323111498083|It Covers The Hillsides                 |
|631.0          |4 Non Blondes               |null                |null |Total 90s                    |null              |What's Up?                              |
|4229.0         |Kenny Wayne Shepherd Band   |Shreveport, LA      |null |Ledbetter Heights            |0.2664833527594104|Born With A Broken Heart (Album Version)|
+---------------+----------------------------+--------------------+-----+-----------------------------+------------------+----------------------------------------+

"""



# (c) Use the test set of user-song plays and recommendations from the collaborative filtering model to compute the following metric
combined_rr = (
  recommended_songs.join(relevant_songs, on='user_id_encoded', how='inner')
  .rdd
  .map(lambda row: (row[1], row[2]))
)
combined_rr.cache()
combined_rr.count()
print(combined_rr.take(1))
"""
[([11, 182, 90, 48, 7, 277, 128, 342, 224, 203], 
[2086, 721, 62276, 1752, 16948, 7188, 704, 1874, 4236, 506, 1263, 4024, 15241, 2038, 4018, 1177, 786, 423, 6260, 298, 422, 43804, 4863, 11095, 1035, 651, 95242, 12293, 433, 2767, 46667, 284, 6767, 3463, 7728, 759, 47541, 53391, 5337, 25518, 3597, 15177, 1401, 2387, 3, 24845, 5440, 61908, 8181, 2794, 54290, 86459, 1956, 10387, 16274, 201, 2035, 49361, 1268, 11179, 84408, 4592, 683, 1559, 32698, 4616])]

"""
# NDCG @ k
ranking_Metrics = RankingMetrics(combined)
NDCG_At_k = ranking_Metrics.ndcgAt(k)
print('NDCG @ K: ', NDCG_At_k) # 0.04600561484724517
# Precision @ k
Precision_At_k = ranking_Metrics.precisionAt(k)
print('Precision @ K: ', Precision_At_k) #0.03732354853595094
# MAP
MAP_At_k = ranking_Metrics.meanAveragePrecision
print('Mean Average Precision (MAP): ', MAP_At_k) #0.01368342057479749




