
"""
Assignment 2 Audio Similarity
@author: Pengcheng Wu
Reference : some codes changed from the Credit Fraud, MovieLens Logistic Regression, SparkLMlib and Assignment Examples
"""
------------------------------------------------------------------------------------------------------------------------------------
####################################################### Audio Similarity #######################################################
------------------------------------------------------------------------------------------------------------------------------------

# start_pyspark_shell -e 4 -c 2 -w 4 -m 4
# Python and pyspark modules required
import sys

from pyspark import SparkContext
from pyspark.sql import SparkSession, Row, DataFrame, Window, functions as F
from pyspark.sql.types import *

from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS
from pyspark.mllib.evaluation import RankingMetrics

import numpy as np

# Required to allow the file to be submitted and run using spark-submit instead
# of using pyspark interactively

spark = SparkSession.builder.getOrCreate()
sc = SparkContext.getOrCreate()

# Compute suitable number of partitions

conf = sc.getConf()

N = int(conf.get("spark.executor.instances"))
M = int(conf.get("spark.executor.cores"))
partitions = 4 * N * M

# Q1-(a)
# Load
audio_dataset_name = "msd-jmir-methods-of-moments-all-v1.0"
schema = audio_dataset_schemas[audio_dataset_name]
data = spark.read.format("com.databricks.spark.csv") \
  .option("header", "true") \
  .option("inferSchema", "false") \
  .schema(schema) \
  .load(f"hdfs:///data/msd/audio/features/{audio_dataset_name}.csv") \
  .repartition(partitions)

print(pretty(data.head().asDict()))
"""
{
  'MSD_TRACKID': "'TRWXMNF12903CBFB52'",
  'Method_of_Moments_Overall_Average_1': 0.2772,
  'Method_of_Moments_Overall_Average_2': 41.87,
  'Method_of_Moments_Overall_Average_3': 2218.0,
  'Method_of_Moments_Overall_Average_4': 201000.0,
  'Method_of_Moments_Overall_Average_5': 34000000.0,
  'Method_of_Moments_Overall_Standard_Deviation_1': 0.1542,
  'Method_of_Moments_Overall_Standard_Deviation_2': 10.47,
  'Method_of_Moments_Overall_Standard_Deviation_3': 424.6,
  'Method_of_Moments_Overall_Standard_Deviation_4': 28750.0,
  'Method_of_Moments_Overall_Standard_Deviation_5': 3991000.0
}

"""
data.columns
"""
['Method_of_Moments_Overall_Standard_Deviation_1',
 'Method_of_Moments_Overall_Standard_Deviation_2',
 'Method_of_Moments_Overall_Standard_Deviation_3',
 'Method_of_Moments_Overall_Standard_Deviation_4',
 'Method_of_Moments_Overall_Standard_Deviation_5',
 'Method_of_Moments_Overall_Average_1',
 'Method_of_Moments_Overall_Average_2',
 'Method_of_Moments_Overall_Average_3',
 'Method_of_Moments_Overall_Average_4',
 'Method_of_Moments_Overall_Average_5',
 'MSD_TRACKID']
 """
data.show(5,False)
data.count()#994615
# -----------------------------------------------------------------------------
# Data analysis
# -----------------------------------------------------------------------------

# Imports

from pyspark.ml.stat import Correlation
from pyspark.ml.feature import VectorAssembler
"""
(
  data
  .groupBy('MSD_TRACKID')
  .count()
  .show(10)
)"""
"""
+--------------------+-----+
|         MSD_TRACKID|count|
+--------------------+-----+
|'TRCBZDM128F9313E49'|    1|
|'TRHDWIM128F421F8CC'|    1|
|'TRROUEF12903CC1CA7'|    1|
|'TRRZNVS12903CE85FD'|    1|
|'TRBNEEK12903CA2B0F'|    1|
|'TRCZZZE128F931A592'|    1|
|'TRRITVV128F9367027'|    1|
|'TRBNRRM12903CAAF0A'|    1|
|'TRBCWWE128F92FFA68'|    1|
|'TRHEFXV128F92C7BA4'|    1|
+--------------------+-----+
"""

# Numeric feature distributions
numdata = data.drop('MSD_TRACKID')
statistics = (
    numdata
    .select([col for col in numdata.columns if col.startswith("Method")])
    .describe()
    .toPandas()
    .set_index("summary")
    .rename_axis(None)
    .T
)
print(statistics)

"""
                                                 count                 mean               stddev        min       max
Method_of_Moments_Overall_Standard_Deviation_1  994615  0.15498169673074455  0.06646229428074175        0.0     0.959
Method_of_Moments_Overall_Standard_Deviation_2  994615   10.384537562272874    3.868009513405999        0.0     55.42
Method_of_Moments_Overall_Standard_Deviation_3  994615    526.8130097675993   180.43762555687198        0.0    2919.0
Method_of_Moments_Overall_Standard_Deviation_4  994615   35071.932226037214   12806.816343438646        0.0  407100.0
Method_of_Moments_Overall_Standard_Deviation_5  994615    5297864.862886645   2089358.5763428248        0.0   4.657E7
Method_of_Moments_Overall_Average_1             994615   0.3508444143530508   0.1855800798239869        0.0     2.647
Method_of_Moments_Overall_Average_2             994615    27.46386395707896    8.352670232009004        0.0     117.0
Method_of_Moments_Overall_Average_3             994615   1495.8090007090195    505.8953013461445        0.0    5834.0
Method_of_Moments_Overall_Average_4             994615    143165.5027677845    50494.29496131541  -146300.0  452500.0
Method_of_Moments_Overall_Average_5             994615  2.396784270518643E7    9307336.208176259        0.0   9.477E7
"""

print(statistics[["mean","stddev"]])

"""
                                                               mean               stddev
Method_of_Moments_Overall_Standard_Deviation_1  0.15498169673074455  0.06646229428074175
Method_of_Moments_Overall_Standard_Deviation_2   10.384537562272874    3.868009513405999
Method_of_Moments_Overall_Standard_Deviation_3    526.8130097675993   180.43762555687198
Method_of_Moments_Overall_Standard_Deviation_4   35071.932226037214   12806.816343438646
Method_of_Moments_Overall_Standard_Deviation_5    5297864.862886645   2089358.5763428248
Method_of_Moments_Overall_Average_1              0.3508444143530508   0.1855800798239869
Method_of_Moments_Overall_Average_2               27.46386395707896    8.352670232009004
Method_of_Moments_Overall_Average_3              1495.8090007090195    505.8953013461445
Method_of_Moments_Overall_Average_4               143165.5027677845    50494.29496131541
Method_of_Moments_Overall_Average_5             2.396784270518643E7    9307336.208176259
"""

# Correlations codes changed from CreditCardFraud and MovieLensLR
# VectorAssembler: A feature transformer that merges multiple columns into a vector column
from pyspark.ml.stat import Correlation
from pyspark.ml.feature import VectorAssembler

assembler = VectorAssembler(
  inputCols=numdata.columns,
  outputCol="Features"
).setHandleInvalid("skip")

features = assembler.transform(numdata).select(['Features'])

features.cache()
features.count()
features.show(10, 80)
"""
+---------------------------------------------------------------------+
|                                                             Features|
+---------------------------------------------------------------------+
|  [0.2152,8.958,481.6,25550.0,3528000.0,0.6299,32.74,1861.0,191800.0]|
|   [0.2165,8.551,333.2,15780.0,2041000.0,0.5444,28.95,1036.0,54250.0]|
|[0.08436,7.041,426.6,36910.0,6533000.0,0.07953,28.01,1457.0,181100.0]|
|  [0.2907,10.34,482.2,27630.0,3562000.0,0.5711,32.24,1826.0,187800.0]|
|  [0.1845,13.27,567.7,38640.0,3827000.0,0.4049,31.83,1977.0,177300.0]|
| [0.08394,10.58,718.4,55560.0,9617000.0,0.09377,19.13,1021.0,83370.0]|
| [0.09515,9.775,614.3,37680.0,5260000.0,0.2478,27.29,1695.0,173800.0]|
|    [0.1026,6.756,443.5,32320.0,4401000.0,0.2156,22.41,974.1,71000.0]|
|  [0.1695,7.833,410.4,26800.0,3948000.0,0.5304,28.63,1635.0,186600.0]|
|    [0.1889,9.78,377.7,25340.0,3127000.0,0.485,37.13,2157.0,199800.0]|
+---------------------------------------------------------------------+

"""



# Calculating correlations and determining what column could be strong
import pandas as pd
from pyspark.mllib.stat import Statistics
threshold = 0.8


col_names = numdata.columns
correlations = Correlation.corr(features,'Features', 'pearson').collect()[0][0].toArray() # Pearson correlation
"""
       0         1         2         3         4         5         6         7         8         9
0  1.000000  0.426283  0.296308  0.061038 -0.055337  0.754208  0.497931  0.447566  0.167465  0.100406
1  0.426283  1.000000  0.857549  0.609521  0.433796  0.025230  0.406925  0.396356  0.015610 -0.040898
2  0.296308  0.857549  1.000000  0.803009  0.682908 -0.082413  0.125913  0.184966 -0.088168 -0.135050
3  0.061038  0.609521  0.803009  1.000000  0.942244 -0.327691 -0.223218 -0.158228 -0.245031 -0.220869
4 -0.055337  0.433796  0.682908  0.942244  1.000000 -0.392551 -0.355018 -0.285964 -0.260195 -0.211809
5  0.754208  0.025230 -0.082413 -0.327691 -0.392551  1.000000  0.549016  0.518502  0.347110  0.278511
6  0.497931  0.406925  0.125913 -0.223218 -0.355018  0.549016  1.000000  0.903367  0.516501  0.422551
7  0.447566  0.396356  0.184966 -0.158228 -0.285964  0.518502  0.903367  1.000000  0.772808  0.685646
8  0.167465  0.015610 -0.088168 -0.245031 -0.260195  0.347110  0.516501  0.772808  1.000000  0.984866
9  0.100406 -0.040898 -0.135050 -0.220869 -0.211809  0.278511  0.422551  0.685646  0.984866  1.000000

"""
corr_df = pd.DataFrame(correlations)
corr_df.index, corr_df.columns = col_names, col_names
print(corr_df.to_string())
"""
                                                                 Method_of_Moments_Overall_Standard_Deviation_1  Method_of_Moments_Overall_Standard_Deviation_2  Method_of_Moments_Overall_Standard_Deviation_3  Method_of_Moments_Overall_Standard_Deviation_4  Method_of_Moments_Overall_Standard_Deviation_5  Method_of_Moments_Overall_Average_1  Method_of_Moments_Overall_Average_2  Method_of_Moments_Overall_Average_3  Method_of_Moments_Overall_Average_4  Method_of_Moments_Overall_Average_5
Method_of_Moments_Overall_Standard_Deviation_1                                        1.000000                                        0.426283                                        0.296308                                        0.061038                                       -0.055337                             0.754208                             0.497931                             0.447566                             0.167465                             0.100406
Method_of_Moments_Overall_Standard_Deviation_2                                        0.426283                                        1.000000                                        0.857549                                        0.609521                                        0.433796                             0.025230                             0.406925                             0.396356                             0.015610                            -0.040898
Method_of_Moments_Overall_Standard_Deviation_3                                        0.296308                                        0.857549                                        1.000000                                        0.803009                                        0.682908                            -0.082413                             0.125913                             0.184966                            -0.088168                            -0.135050
Method_of_Moments_Overall_Standard_Deviation_4                                        0.061038                                        0.609521                                        0.803009                                        1.000000                                        0.942244                            -0.327691                            -0.223218                            -0.158228                            -0.245031                            -0.220869
Method_of_Moments_Overall_Standard_Deviation_5                                       -0.055337                                        0.433796                                        0.682908                                        0.942244                                        1.000000                            -0.392551                            -0.355018                            -0.285964                            -0.260195                            -0.211809
Method_of_Moments_Overall_Average_1                                                   0.754208                                        0.025230                                       -0.082413                                       -0.327691                                       -0.392551                             1.000000                             0.549016                             0.518502                             0.347110                             0.278511
Method_of_Moments_Overall_Average_2                                                   0.497931                                        0.406925                                        0.125913                                       -0.223218                                       -0.355018                             0.549016                             1.000000                             0.903367                             0.516501                             0.422551
Method_of_Moments_Overall_Average_3                                                   0.447566                                        0.396356                                        0.184966                                       -0.158228                                       -0.285964                             0.518502                             0.903367                             1.000000                             0.772808                             0.685646
Method_of_Moments_Overall_Average_4                                                   0.167465                                        0.015610                                       -0.088168                                       -0.245031                                       -0.260195                             0.347110                             0.516501                             0.772808                             1.000000                             0.984866
Method_of_Moments_Overall_Average_5                                                   0.100406                                       -0.040898                                       -0.135050                                       -0.220869                                       -0.211809                             0.278511                             0.422551                             0.685646                             0.984866                             1.000000

"""
num_correlated_col_nsame = (correlations > threshold).sum() - correlations.shape[0]
corr_num = (correlations > threshold).astype(int)
corr_num_df = pd.DataFrame(corr_num)
corr_num_df.index, corr_num_df.columns = col_names, col_names
print(corr_num_df.to_string())

"""
                                                                        Method_of_Moments_Overall_Standard_Deviation_1  Method_of_Moments_Overall_Standard_Deviation_2  Method_of_Moments_Overall_Standard_Deviation_3  Method_of_Moments_Overall_Standard_Deviation_4  Method_of_Moments_Overall_Standard_Deviation_5  Method_of_Moments_Overall_Average_1  Method_of_Moments_Overall_Average_2  Method_of_Moments_Overall_Average_3  Method_of_Moments_Overall_Average_4  Method_of_Moments_Overall_Average_5
Method_of_Moments_Overall_Standard_Deviation_1                                               1                                               0                                               0                                               0                                               0                                    0                                    0                                    0                                    0                                    0
Method_of_Moments_Overall_Standard_Deviation_2                                               0                                               1                                               1                                               0                                               0                                    0                                    0                                    0                                    0                                    0
Method_of_Moments_Overall_Standard_Deviation_3                                               0                                               1                                               1                                               1                                               0                                    0                                    0                                    0                                    0                                    0
Method_of_Moments_Overall_Standard_Deviation_4                                               0                                               0                                               1                                               1                                               1                                    0                                    0                                    0                                    0                                    0
Method_of_Moments_Overall_Standard_Deviation_5                                               0                                               0                                               0                                               1                                               1                                    0                                    0                                    0                                    0                                    0
Method_of_Moments_Overall_Average_1                                                          0                                               0                                               0                                               0                                               0                                    1                                    0                                    0                                    0                                    0
Method_of_Moments_Overall_Average_2                                                          0                                               0                                               0                                               0                                               0                                    0                                    1                                    1                                    0                                    0
Method_of_Moments_Overall_Average_3                                                          0                                               0                                               0                                               0                                               0                                    0                                    1                                    1                                    0                                    0
Method_of_Moments_Overall_Average_4                                                          0                                               0                                               0                                               0                                               0                                    0                                    0                                    0                                    1                                    1
Method_of_Moments_Overall_Average_5                                                          0                                               0                                               0                                               0                                               0                                    0                                    0                                    0                                    1                                    1

"""
print(correlations.shape) # (10,10)
print(num_correlated_col_nsame) # 10

for i in range(0, correlations.shape[0]):
    for j in range(i + 1, correlations.shape[1]):
        if correlations[i, j] > threshold:
            print((numdata.columns[i], numdata.columns[j]), correlations[i,j])
"""
('Method_of_Moments_Overall_Standard_Deviation_2', 'Method_of_Moments_Overall_Standard_Deviation_3') 0.8575494246545305
('Method_of_Moments_Overall_Standard_Deviation_3', 'Method_of_Moments_Overall_Standard_Deviation_4') 0.8030091088989109
('Method_of_Moments_Overall_Standard_Deviation_4', 'Method_of_Moments_Overall_Standard_Deviation_5') 0.942244348087381
('Method_of_Moments_Overall_Average_2', 'Method_of_Moments_Overall_Average_3')                       0.903367167926527
('Method_of_Moments_Overall_Average_4', 'Method_of_Moments_Overall_Average_5')                       0.9848664625619973
"""
# Remove one of each pair of correlated variables iteratively 
n = correlations.shape[0]
counter = 0
indexes = np.array(list(range(0, n)))
matrix = correlations

for j in range(0, n):
  mask = matrix > threshold
  sums = mask.sum(axis=0)
  index = np.argmax(sums)
  value = sums[index]
  check = value > 1
  if check:
    k = matrix.shape[0]
    keep = [i for i in range(0, k) if i != index]
    matrix = matrix[keep, :][:, keep]
    indexes = indexes[keep]
  else:
    break
  counter += 1
  print(counter)
  
# check correlations against threshold
correlations_new = correlations[indexes, :][:, indexes]
num_correlated_columns_not_the_same = (correlations_new > threshold).sum() - correlations_new.shape[0]
print((correlations_new > threshold).astype(int))
print(correlations_new.shape)
print(num_correlated_columns_not_the_same)




#Q1-(b)
#Load MSD All Music Genre Dataset (MAGD).
schema_MAGD = StructType([
   StructField('TRACK_ID',StringType()),
   StructField('Genre_Name',StringType())
   ])
   
MAGD = (
    spark.read.format("com.databricks.spark.csv")
    .option("header", "false")
    .option("inferSchema", "false")
    .option("delimiter", "\t")
    .schema(schema_MAGD)
    .load("/data/msd/genre/msd-MAGD-genreAssignment.tsv")
    .repartition(partitions)
)
MAGD.show(10,80)
"""
+------------------+----------+
|          TRACK_ID|Genre_Name|
+------------------+----------+
|TRXOZJE128F4278282|  Pop_Rock|
|TRYEEUA128E078BCE9|  Pop_Rock|
|TRZAUBX128F92ED4D5|  Pop_Rock|
|TRXHIEF128F14654A0|Electronic|
|TRYIKQZ128F42B7B01|       Rap|
|TRYSBYB128F4277BB2|  Pop_Rock|
|TRTMKVP128F9302EFE|  Pop_Rock|
|TRUVQAH128F425B4A5|    Reggae|
|TRWKYJA128F149CBC4|  Pop_Rock|
|TRYPMLG128F427F301|  Pop_Rock|
+------------------+----------+
"""

MAGD.count() #422714

# Visualize the distribution of genres for the songs that were matched.
MAGD = MAGD.join(mismatches_not_accepted, on = 'TRACK_ID', how = 'left_anti')
MAGD.count() # 415350
## distribution of genres
genres_dtb = (
    MAGD
    .groupBy('Genre_Name')
    .agg({'Genre_Name':'count'})
    .select(F.col('Genre_Name'),F.col('count(Genre_Name)').alias('COUNT'))
    .orderBy('COUNT',ascending = False)
    )
genres_dtb.count() #21
genres_dtb.show(10,80)
"""
+-------------+------+
|   Genre_Name| COUNT|
+-------------+------+
|     Pop_Rock|234107|
|   Electronic| 40430|
|          Rap| 20606|
|         Jazz| 17673|
|        Latin| 17475|
|International| 14094|
|          RnB| 13874|
|      Country| 11492|
|    Religious|  8754|
|       Reggae|  6885|
+-------------+------+
"""

## plotting the distribution of genres
#export QT_QPA_PLATFORM='offscreen'

import numpy as np
from collections import OrderedDict
import pandas as pd
import matplotlib; matplotlib.use('agg')
import matplotlib.pyplot as plt
import os
import io

genres = genres_dtb.toPandas()

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

f, a = plt.subplots(dpi=300, figsize=(10, 5))

N = len(genres)

index = np.arange(N)
width = 0.5
name_list = genres.Genre_Name.tolist()
value_list = genres.COUNT.tolist()

a.bar(index, value_list,width,label="genre count", color="#87CEFA")

plt.xlabel('Genre')
plt.ylabel('Genre (count)')
plt.title('Distribution of Genre')
plt.xticks(index, name_list,rotation=45)
plt.legend(loc="upper right")

plt.tight_layout() #reduce whitespace
f.savefig(os.path.join(os.path.expanduser("~/genres_dtb.png")), bbox_inches="tight") # save as png and view in windows
plt.close()




# Q1-(c) Merge the genres dataset and the audio features dataset so that every song has a label.
data_new = data.withColumn("MSD_TRACKID",F.regexp_replace("MSD_TRACKID","'","")).withColumnRenamed("MSD_TRACKID","TRACK_ID")
au_ge_data = data_new.join(MAGD,on = "TRACK_ID",how="left").where(F.col("Genre_Name").isNotNull()).dropDuplicates()

au_ge_data.show(5,False)
au_ge_data.count()#413289
"""
+------------------+----------------------------------------------+----------------------------------------------+----------------------------------------------+----------------------------------------------+----------------------------------------------+-----------------------------------+-----------------------------------+-----------------------------------+-----------------------------------+-----------------------------------+----------+
|TRACK_ID          |Method_of_Moments_Overall_Standard_Deviation_1|Method_of_Moments_Overall_Standard_Deviation_2|Method_of_Moments_Overall_Standard_Deviation_3|Method_of_Moments_Overall_Standard_Deviation_4|Method_of_Moments_Overall_Standard_Deviation_5|Method_of_Moments_Overall_Average_1|Method_of_Moments_Overall_Average_2|Method_of_Moments_Overall_Average_3|Method_of_Moments_Overall_Average_4|Method_of_Moments_Overall_Average_5|Genre_Name|
+------------------+----------------------------------------------+----------------------------------------------+----------------------------------------------+----------------------------------------------+----------------------------------------------+-----------------------------------+-----------------------------------+-----------------------------------+-----------------------------------+-----------------------------------+----------+
|TRAAABD128F429CF47|0.1308                                        |9.587                                         |459.9                                         |27280.0                                       |4303000.0                                     |0.2474                             |26.02                              |1067.0                             |67790.0                            |8281000.0                          |Pop_Rock  |
|TRAABPK128F424CFDB|0.1208                                        |6.738                                         |215.1                                         |11890.0                                       |2278000.0                                     |0.4882                             |41.76                              |2164.0                             |220400.0                           |3.79E7                             |Pop_Rock  |
|TRAACER128F4290F96|0.2838                                        |8.995                                         |429.5                                         |31990.0                                       |5272000.0                                     |0.5388                             |28.29                              |1656.0                             |185100.0                           |3.164E7                            |Pop_Rock  |
|TRAADYB128F92D7E73|0.1346                                        |7.321                                         |499.6                                         |38460.0                                       |5877000.0                                     |0.2839                             |15.75                              |929.6                              |116500.0                           |2.058E7                            |Jazz      |
|TRAAGHM128EF35CF8E|0.1563                                        |9.959                                         |502.8                                         |26190.0                                       |3660000.0                                     |0.3835                             |28.24                              |1864.0                             |180800.0                           |2.892E7                            |Electronic|
+------------------+----------------------------------------------+----------------------------------------------+----------------------------------------------+----------------------------------------------+----------------------------------------------+-----------------------------------+-----------------------------------+-----------------------------------+-----------------------------------+-----------------------------------+----------+
"""
#Q2-(b) Convert the genre column into a binary column that represents if the song is ”Electronic” or some other genre.
data_binary = au_ge_data.withColumn('label', F.when(au_ge_data.Genre_Name == 'Electronic',1).otherwise(0))
print(pretty(data_binary.head().asDict()))
"""
{
  'Genre_Name': 'Pop_Rock',
  'label': 0,
  'Method_of_Moments_Overall_Average_1': 0.2474,
  'Method_of_Moments_Overall_Average_2': 26.02,
  'Method_of_Moments_Overall_Average_3': 1067.0,
  'Method_of_Moments_Overall_Average_4': 67790.0,
  'Method_of_Moments_Overall_Average_5': 8281000.0,
  'Method_of_Moments_Overall_Standard_Deviation_1': 0.1308,
  'Method_of_Moments_Overall_Standard_Deviation_2': 9.587,
  'Method_of_Moments_Overall_Standard_Deviation_3': 459.9,
  'Method_of_Moments_Overall_Standard_Deviation_4': 27280.0,
  'Method_of_Moments_Overall_Standard_Deviation_5': 4303000.0,
  'TRACK_ID': 'TRAAABD128F429CF47'
}
"""
# Class balance of the binary label
data_binary.groupBy("label").count().show(2)
"""
+-----+------+
|label| count|
+-----+------+
|    0|373263|
|    1| 40026|
+-----+------+
"""
#Assemble Features
# As the above removing each of high correlation, we get the indexes. Assemble vector only from the remaining columns
inputCols = np.array(data_binary.columns[:-1])[indexes]  # assemble only the feature columns that aren't highly correlated
inputDrops = data_binary.select([col for col in data_binary.columns if col in inputCols])

# new_data_binary = data_binary.drop('Method_of_Moments_Overall_Standard_Deviation_4', 'Method_of_Moments_Overall_Average_4')
assembler_1 = VectorAssembler(
  
  inputCols = [col for col in inputDrops.columns if col.startswith("Method")],
  outputCol= "features" 
).setHandleInvalid("skip")

features_1 = assembler_1.transform(data_binary).select(["TRACK_ID","Genre_Name","label","features"])
features_1.count() #413289
#features_1.cache()
features_1.show()
"""
+------------------+-------------+-----+--------------------+
|          TRACK_ID|   Genre_Name|label|          Features_1|
+------------------+-------------+-----+--------------------+
|TRAAABD128F429CF47|     Pop_Rock|    0|[0.1308,27280.0,4...|
|TRAABPK128F424CFDB|     Pop_Rock|    0|[0.1208,11890.0,2...|
|TRAACER128F4290F96|     Pop_Rock|    0|[0.2838,31990.0,5...|
|TRAADYB128F92D7E73|         Jazz|    0|[0.1346,38460.0,5...|
|TRAAGHM128EF35CF8E|   Electronic|    1|[0.1563,26190.0,3...|
|TRAAGRV128F93526C0|     Pop_Rock|    0|[0.1076,19350.0,2...|
|TRAAGTO128F1497E3C|     Pop_Rock|    0|[0.1069,43100.0,7...|
|TRAAHAU128F9313A3D|     Pop_Rock|    0|[0.08485,23750.0,...|
|TRAAHEG128E07861C3|          Rap|    0|[0.1699,52440.0,8...|
|TRAAHZP12903CA25F4|          Rap|    0|[0.1654,33100.0,5...|
|TRAAICW128F1496C68|International|    0|[0.1104,19540.0,3...|
|TRAAJJW12903CBDDCB|International|    0|[0.2267,37980.0,4...|
|TRAAKLX128F934CEE4|   Electronic|    1|[0.1647,64420.0,1...|
|TRAAKWR128F931B29F|     Pop_Rock|    0|[0.04881,34410.0,...|
|TRAALQN128E07931A4|   Electronic|    1|[0.1989,30690.0,4...|
|TRAAMFF12903CE8107|     Pop_Rock|    0|[0.1385,31590.0,4...|
|TRAAMHG128F92ED7B2|International|    0|[0.1799,29170.0,4...|
|TRAAROH128F42604B0|   Electronic|    1|[0.1192,41670.0,6...|
|TRAARQN128E07894DF|     Pop_Rock|    0|[0.2559,61750.0,1...|
|TRAASBB128F92E5354|     Pop_Rock|    0|[0.1959,50310.0,7...|
+------------------+-------------+-----+--------------------+

"""
# Q2-(c) Split the dataset into training and test sets (Assignment Examples)
from pyspark.sql.window import *
from pyspark.ml.feature import VectorAssembler

# Helpers
def print_class_balance(datas, name):
  N = datas.count()
  counts = datas.groupBy("label").count().toPandas()
  counts["ratio"] = counts["count"] / N
  print(name)
  print(N)
  print(counts)
  print("")



# 1. randomSplit (No Stratified)
training, test = features_1.randomSplit([0.8, 0.2])
training.cache()
test.cache()

print_class_balance(features_1, "features")
print_class_balance(training, "training")
print_class_balance(test, "test")
"""
features
413289
   label   count     ratio
     1   40026  0.096847
     0  373263  0.903153

training
330680
   label   count     ratio
     1   32200  0.097375
     0  298480  0.902625

test
82609
   label  count     ratio
      1   7826  0.094735
      0  74783  0.905265

"""

# 2-1. Stratified Sample
temp = features_1.withColumn("TRACK_ID", F.monotonically_increasing_id())
training_stratified = temp.sampleBy("label", fractions={0: 0.8, 1: 0.8})
training_stratified.cache()
training_stratified.show()
"""
+--------+-------------+-----+--------------------+
|TRACK_ID|   Genre_Name|label|          Features_1|
+--------+-------------+-----+--------------------+
|       0|     Pop_Rock|    0|[0.1308,27280.0,4...|
|       1|     Pop_Rock|    0|[0.1208,11890.0,2...|
|       2|     Pop_Rock|    0|[0.2838,31990.0,5...|
|       4|   Electronic|    1|[0.1563,26190.0,3...|
|       6|     Pop_Rock|    0|[0.1069,43100.0,7...|
|       7|     Pop_Rock|    0|[0.08485,23750.0,...|
|       8|          Rap|    0|[0.1699,52440.0,8...|
|       9|          Rap|    0|[0.1654,33100.0,5...|
|      10|International|    0|[0.1104,19540.0,3...|
|      11|International|    0|[0.2267,37980.0,4...|
|      13|     Pop_Rock|    0|[0.04881,34410.0,...|
|      15|     Pop_Rock|    0|[0.1385,31590.0,4...|
|      16|International|    0|[0.1799,29170.0,4...|
|      17|   Electronic|    1|[0.1192,41670.0,6...|
|      18|     Pop_Rock|    0|[0.2559,61750.0,1...|
|      19|     Pop_Rock|    0|[0.1959,50310.0,7...|
|      20|       Reggae|    0|[0.2276,44820.0,5...|
|      21|     Pop_Rock|    0|[0.09249,50830.0,...|
|      24|     Pop_Rock|    0|[0.2812,27860.0,4...|
|      25|     Pop_Rock|    0|[0.118,19700.0,30...|
+--------+-------------+-----+--------------------+

"""

test_stratified = temp.join(training_stratified, on="TRACK_ID", how="left_anti")
#test_stratified.cache()

training_stratified = training_stratified.drop("TRACK_ID")
test_stratified = test_stratified.drop("TRACK_ID")

print_class_balance(features_1, "features_1")
print_class_balance(training_stratified, "training_stratified")
print_class_balance(test_stratified, "test_stratified")
"""
features_1
413289
   label   count     ratio
0      1   40026  0.096847
1      0  373263  0.903153

training
330652
   label   count     ratio
0      1   31895  0.096461
1      0  298757  0.903539

test
82637
   label  count     ratio
0      1   8131  0.098394
1      0  74506  0.901606

"""
# 2-2 Exact stratification using Window (multi-class variant in comments) 
temp = (
  features_1
  .withColumn("TRACK_ID", F.monotonically_increasing_id())
  .withColumn("Random", F.rand()) #random number between 0 and 1
  .withColumn(
    "Row",
    F.row_number() # row number in each calss partition (start at 0,1,...for positive and separately for negative
    .over(Window.partitionBy("label").orderBy("Random"))
  )
)
training_estratify = temp.where(
  ((F.col("label") == 0) & (F.col("Row") < 373263 * 0.8)) | 
  ((F.col("label") == 1) & (F.col("Row") < 40026 * 0.8))
  
)

test_estratify = temp.join(training_estratify, on="TRACK_ID", how="left_anti")
test_estratify.cache()

training_estratify = training_estratify.drop("TRACK_ID", "Random", "Row")
test_estratify = test_estratify.drop("TRACK_ID", "Random", "Row")

print_class_balance(features_1, "features")
print_class_balance(training_estratify, "training_estratify")
print_class_balance(test_estratify, "test_estratify")
"""
features
413289
   label   count     ratio
0      1   40026  0.096847
1      0  373263  0.903153

training
330630
   label   count     ratio
0      1   32020  0.096845
1      0  298610  0.903155

test
82659
   label  count     ratio
0      1   8006  0.096856
1      0  74653  0.903144

"""

# 3.Downsampling
training_downsampled = (
    training
    .withColumn("Random", rand())
    .where((col("label") != 0) | ((col("label") == 0) & (col("Random") < 2 * (40026 / 373263))))
)
training_downsampled.cache()
test_downsampled = test
print_class_balance(training_downsampled, "training_downsampled")
print_class_balance(test_downsampled, "test_downsampled")
"""
training_downsampled
96356
   label  count     ratio
0      1  32040  0.332517
1      0  64316  0.667483

test_downsampled
82679
   label  count    ratio
0      1   7986  0.09659
1      0  74693  0.90341

"""
#4. observation weight
training_weighted = (
    training
    .withColumn(
        "Weight",F.when(F.col("label") == 0, 1)
          .otherwise(0)   
    )
)
test_weighted=test
print_class_balance(training_weighted, "training_weighted")
print_class_balance(test_weighted, "test_weighted")
"""
training_weighted
330645
   label   count     ratio
0      1   31941  0.096602
1      0  298704  0.903398

test_weighted
82644
   label  count     ratio
0      1   8085  0.097829
1      0  74559  0.902171

"""
# Imports
import numpy as np

from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator

# Helpers

def with_custom_prediction(predictions, threshold, probabilityCol="probability", customPredictionCol="customPrediction"):

  def apply_custom_threshold(probability, threshold):
    return int(probability[1] > threshold)

  apply_custom_threshold_udf = udf(lambda x: apply_custom_threshold(x, threshold), IntegerType())

  return predictions.withColumn(customPredictionCol, apply_custom_threshold_udf(F.col(probabilityCol)))

def print_class_balance(data, name):
  N = data.count()
  counts = data.groupBy("label").count().toPandas()
  counts["ratio"] = counts["count"] / N
  print(name)
  print(N)
  print(counts)
  print("")

def print_binary_metrics(predictions, threshold=0.5, labelCol="label", predictionCol="prediction", rawPredictionCol="rawPrediction", probabilityCol="probability"):

  if threshold != 0.5:

    predictions = with_custom_prediction(predictions, threshold)
    predictionCol = "customPrediction"

  total = predictions.count()
  positive = predictions.filter((F.col(labelCol) == 1)).count()
  negative = predictions.filter((F.col(labelCol) == 0)).count()
  nP = predictions.filter((F.col(predictionCol) == 1)).count()
  nN = predictions.filter((F.col(predictionCol) == 0)).count()
  TP = predictions.filter((F.col(predictionCol) == 1) & (F.col(labelCol) == 1)).count()
  FP = predictions.filter((F.col(predictionCol) == 1) & (F.col(labelCol) == 0)).count()
  FN = predictions.filter((F.col(predictionCol) == 0) & (F.col(labelCol) == 1)).count()
  TN = predictions.filter((F.col(predictionCol) == 0) & (F.col(labelCol) == 0)).count()

  binary_evaluator = BinaryClassificationEvaluator(rawPredictionCol=rawPredictionCol, labelCol=labelCol, metricName="areaUnderROC")
  auroc = binary_evaluator.evaluate(predictions)

  print('actual total:    {}'.format(total))
  print('actual positive: {}'.format(positive))
  print('actual negative: {}'.format(negative))
  print('threshold:       {}'.format(threshold))
  print('nP:              {}'.format(nP))
  print('nN:              {}'.format(nN))
  print('TP:              {}'.format(TP))
  print('FP:              {}'.format(FP))
  print('FN:              {}'.format(FN))
  print('TN:              {}'.format(TN))
  print('precision:       {}'.format(TP / (TP + FP)))
  print('recall:          {}'.format(TP / (TP + FN)))
  print('accuracy:        {}'.format((TP + TN) / total))
  print('auroc:           {}'.format(auroc))

# Check stuff is cached
# features_1.cache()
# training.cache()
# test.cache()






# Q2-(d) & (e) train model and test performance in three classification algorithms & (f)   

# Logistic Regression
lr_t = LogisticRegression(featuresCol='features', labelCol='label')
lrt_model = lr_t.fit(training)
predictions_lrt = lrt_model.transform(test)
#predictions_lrt.cache()
print_binary_metrics(predictions_lrt)
print_binary_metrics(predictions_lrt, threshold=0.5)
"""
actual total:    82562
actual positive: 8013
actual negative: 74549
threshold:       0.5
nP:              150
nN:              82412
TP:              27
FP:              123
FN:              7986
TN:              74426
precision:       0.18
recall:          0.0033695245226506927
accuracy:        0.9017829025459655
auroc:           0.6905673477047771
actual total:    82562
actual positive: 8013
actual negative: 74549
threshold:       0.2
nP:              4783
nN:              77779
TP:              1657
FP:              3126
FN:              6356
TN:              71423
precision:       0.3464352916579553
recall:          0.20678896792711843
accuracy:        0.8851529759453501
auroc:           0.6905596379631909

"""

# Logistic Regression Downsample train
from datetime import datetime
ts = datetime.now()
lr = LogisticRegression(featuresCol='features', labelCol='label')
lr_model = lr.fit(training_downsampled)
predictions_lr = lr_model.transform(test)
#predictions_lr.cache()
print_binary_metrics(predictions_lr)
print_binary_metrics(predictions_lr, threshold=0.5)
te = datetime.now()
print(" This running time is: ", format(te-ts))
"""
actual total:    82950
actual positive: 8104
actual negative: 74846
threshold:       0.5
nP:              6950
nN:              76000
TP:              2094
FP:              4856
FN:              6010
TN:              69990
precision:       0.301294964028777
recall:          0.258390918065153
accuracy:        0.869005424954792
auroc:           0.6884335110838579
actual total:    82950
actual positive: 8104
actual negative: 74846
threshold:       0.5
nP:              6950
nN:              76000
TP:              2094
FP:              4856
FN:              6010
TN:              69990
precision:       0.301294964028777
recall:          0.258390918065153
accuracy:        0.869005424954792
auroc:           0.68843137359188

 This running time is:  0:01:00.897088


"""
# Random Forest original training
from pyspark.ml.classification import RandomForestClassifier
rft = RandomForestClassifier (featuresCol='features', labelCol='label',numTrees = 20, maxDepth=5, maxBins=32)
rft_model = rft.fit(training)
predictions_rft = rft_model.transform(test)
predictions_rft.cache()
print_binary_metrics(predictions_rft)
print_binary_metrics(predictions_rft, threshold=0.5)
"""
actual total:    82950
actual positive: 8104
actual negative: 74846
threshold:       0.5
nP:              0
nN:              82950
TP:              0
FP:              0
FN:              8104
TN:              74846

Remaining results are N/A because division by zero

"""

#Random Forest Downsample training
from pyspark.ml.classification import RandomForestClassifier
from datetime import datetime
ts = datetime.now()
rf = RandomForestClassifier (featuresCol='features', labelCol='label',numTrees = 20, maxDepth=5, maxBins=32)
rf_model = rf.fit(training_downsampled)
predictions_rf = rf_model.transform(test)
# predictions_rf.cache()
print_binary_metrics(predictions_rf)
print_binary_metrics(predictions_rf, threshold=0.5)
te = datetime.now()
print(" This running time is: ", format(te-ts))
"""
actual total:    82950
actual positive: 8104
actual negative: 74846
threshold:       0.5
nP:              9233
nN:              73717
TP:              2898
FP:              6335
FN:              5206
TN:              68511
precision:       0.31387414708112205
recall:          0.3576011846001974
accuracy:        0.860867992766727
auroc:           0.7477327013408962
actual total:    82950
actual positive: 8104
actual negative: 74846
threshold:       0.5
nP:              9233
nN:              73717
TP:              2898
FP:              6335
FN:              5206
TN:              68511
precision:       0.31387414708112205
recall:          0.3576011846001974
accuracy:        0.860867992766727
auroc:           0.7477385268926926
 This running time is:  0:00:38.904187


"""
"""
Maybe data balance one of factor for Random Forest
"""
# GBT original trianing
from pyspark.ml.classification import GBTClassifier
gbtt = GBTClassifier(featuresCol='features', labelCol='label')
gbtt_model = gbtt.fit(training)
predictions_gbtt = gbtt_model.transform(test)
predictions_gbtt.cache()
print_binary_metrics(predictions_gbtt)
print_binary_metrics(predictions_gbtt, threshold=0.5)
"""
actual total:    82950
actual positive: 8104
actual negative: 74846
threshold:       0.5
nP:              834
nN:              82116
TP:              481
FP:              353
FN:              7623
TN:              74493
precision:       0.5767386091127098
recall:          0.05935340572556762
accuracy:        0.9038456901748041
auroc:           0.7735420473045556
actual total:    82950
actual positive: 8104
actual negative: 74846
threshold:       0.5
nP:              834
nN:              82116
TP:              481
FP:              353
FN:              7623
TN:              74493
precision:       0.5767386091127098
recall:          0.05935340572556762
accuracy:        0.9038456901748041
auroc:           0.7735384062316413

"""

# GBT Downsampling training
from pyspark.ml.classification import GBTClassifier
from datetime import datetime 
ts = datetime.now()
gbt = GBTClassifier(featuresCol='features', labelCol='label')
gbt_model = gbt.fit(training_downsampled)
predictions_gbt = gbt_model.transform(test)
#predictions_gbt.cache()
print_binary_metrics(predictions_gbt)
print_binary_metrics(predictions_gbt, threshold=0.5)
te = datetime.now()
print(" This running time is: ", format(te-ts))
"""
actual total:    82950
actual positive: 8104
actual negative: 74846
threshold:       0.5
nP:              11739
nN:              71211
TP:              3555
FP:              8184
FN:              4549
TN:              66662
precision:       0.3028366981855354
recall:          0.43867226061204345
accuracy:        0.8464978902953586
auroc:           0.7719346533371491
actual total:    82950
actual positive: 8104
actual negative: 74846
threshold:       0.5
nP:              11739
nN:              71211
TP:              3555
FP:              8184
FN:              4549
TN:              66662
precision:       0.3028366981855354
recall:          0.43867226061204345
accuracy:        0.8464978902953586
auroc:           0.771934087021303
This running time is:  0:02:08.904187
"""

## Q3-(b)Use cross-validation to tune some of the hyperparameters of the best performing binary classification model.
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.ml.evaluation import BinaryClassificationEvaluator,MulticlassClassificationEvaluator
from pyspark.mllib.evaluation import BinaryClassificationMetrics
# Create ParamGrid for Cross Validation
lr_Grid = ParamGridBuilder().addGrid(lr.regParam, [0, 0.01,0.1]).addGrid(lr.elasticNetParam, [0, 1]).build()
rf_Grid = ParamGridBuilder().addGrid(rf.numTrees, [10,20,30]).addGrid(rf.maxDepth, [3,5,8]).addGrid(rf.maxBins,[28,32,40]).build()
gbt_Grid = ParamGridBuilder().addGrid(gbt.maxDepth, [2, 5, 8]).addGrid(gbt.maxBins, [20, 40]).addGrid(gbt.stepSize, [0.01, 0.1]).build()
grid_list = [lr_Grid, rf_Grid, gbt_Grid]
binary_evaluator = BinaryClassificationEvaluator(metricName="areaUnderROC") #default areaUnderROC
muti_evaluator = MulticlassClassificationEvaluator()
model_list = [lr,rf,gbt]
# Evaluate Model
# Create 5-fold CrossValidation
finalModel_list_b = []    
for i in range(len(model_list)):
    cv = CrossValidator(
                    estimator=model_list[i],
                    estimatorParamMaps=grid_list[i],
                    evaluator=binary_evaluator,  #areaUnderROC 
                    numFolds=10,seed=1000, parallelism=8)
    cv_model = cv.fit(training_downsampled)   #train 
    cv_prediction = cv_model.transform(test) #test
 
    finalModel_list_b.append(cv_model.bestModel) #save the best model
    
    print("------Cross-Validation performance of {}:--------".format(model_list[i]))
    print_binary_metrics(cv_prediction,labelCol="label")   ##'Field "Class" does not exist.\n
"""
------Cross-Validation performance of LogisticRegression_2607a04481cc:--------
actual total:    82950
actual positive: 8104
actual negative: 74846
threshold:       0.5
nP:              5174
nN:              77776
TP:              1655
FP:              3519
FN:              6449
TN:              71327
precision:       0.3198685736374179
recall:          0.20422013820335636
accuracy:        0.879831223628692
auroc:           0.6871691289365233

------Cross-Validation performance of RandomForestClassifier_e142e74bc088:--------
actual total:    82950
actual positive: 8104
actual negative: 74846
threshold:       0.5
nP:              11044
nN:              71906
TP:              3478
FP:              7566
FN:              4626
TN:              67280
precision:       0.3149221296631655
recall:          0.42917077986179664
accuracy:        0.8530198915009042
auroc:           0.7720277442864649

------Cross-Validation performance of GBTClassifier_291f8175b0e3:--------
actual total:    82950
actual positive: 8104
actual negative: 74846
threshold:       0.5
nP:              12225
nN:              70725
TP:              3827
FP:              8398
FN:              4277
TN:              66448
precision:       0.31304703476482615
recall:          0.4722359328726555
accuracy:        0.8471971066907775
auroc:           0.7844548514410598


"""


# Q4-(b)
# Covert the genre column into an integer index
from pyspark.ml.feature import VectorAssembler,StringIndexer, StandardScaler

label_strIdx = StringIndexer(inputCol = 'Genre_Name', outputCol = 'label')
multi_data = label_strIdx.fit(au_ge_data).transform(au_ge_data).drop('TRACK_ID')

assembler_multi = VectorAssembler(
    inputCols = [col for col in multi_data.columns if col.startswith('Method')],
    outputCol =  'new_features'
)
multi_data_features = assembler_multi.transform(multi_data)
multi_data_features.cache()
print(pretty(multi_data_features.head().asDict()))
"""
{
  'Genre_Name': 'Pop_Rock',
  'Method_of_Moments_Overall_Average_1': 0.2474,
  'Method_of_Moments_Overall_Average_2': 26.02,
  'Method_of_Moments_Overall_Average_3': 1067.0,
  'Method_of_Moments_Overall_Average_4': 67790.0,
  'Method_of_Moments_Overall_Average_5': 8281000.0,
  'Method_of_Moments_Overall_Standard_Deviation_1': 0.1308,
  'Method_of_Moments_Overall_Standard_Deviation_2': 9.587,
  'Method_of_Moments_Overall_Standard_Deviation_3': 459.9,
  'Method_of_Moments_Overall_Standard_Deviation_4': 27280.0,
  'Method_of_Moments_Overall_Standard_Deviation_5': 4303000.0,
  'label': 0.0,
  'new_features': DenseVector([0.1308, 9.587, 459.9, 27280.0, 4303000.0, 0.2474, 26.02, 1067.0, 67790.0, 8281000.0])
}

"""
def print_class_balance(data, name):
    N = data.count()
    counts = data.groupBy("label").count().orderBy('count').toPandas()
    counts["ratio"] = counts["count"] / N
    print(name)
    print(N)
    print(counts)
    print("")
    
# Random split 
    
train_random, test_random = multi_data_features.randomSplit([0.8,0.2],seed = 1000)
print_class_balance(train_random, 'train_random')
"""
train_random
330544
    label   count     ratio
0    20.0     150  0.000454
1    19.0     376  0.001138
2    18.0     448  0.001355
3    17.0     805  0.002435
4    16.0    1221  0.003694
5    15.0    1315  0.003978
6    14.0    1613  0.004880
7    13.0    3178  0.009614
8    12.0    4520  0.013674
9    11.0    4818  0.014576
10   10.0    5428  0.016421
11    9.0    5525  0.016715
12    8.0    6946  0.021014
13    7.0    9146  0.027670
14    6.0   11064  0.033472
15    5.0   11258  0.034059
16    4.0   13884  0.042003
17    3.0   14112  0.042693
18    2.0   16505  0.049933
19    1.0   31977  0.096741
20    0.0  186255  0.563480


"""
# train lr
lr = LogisticRegression(featuresCol='new_features', labelCol = 'label')
lr_model = lr.fit(train_random)   #train
predictions = lr_model.transform(test_random) #test
print(pretty(predictions.head().asDict()))
"""
{
  'Genre_Name': 'Jazz',
  'Method_of_Moments_Overall_Average_1': 0.0,
  'Method_of_Moments_Overall_Average_2': 0.0,
  'Method_of_Moments_Overall_Average_3': 0.0,
  'Method_of_Moments_Overall_Average_4': 0.0,
  'Method_of_Moments_Overall_Average_5': 0.0,
  'Method_of_Moments_Overall_Standard_Deviation_1': 0.0,
  'Method_of_Moments_Overall_Standard_Deviation_2': 0.0,
  'Method_of_Moments_Overall_Standard_Deviation_3': 0.0,
  'Method_of_Moments_Overall_Standard_Deviation_4': 0.0,
  'Method_of_Moments_Overall_Standard_Deviation_5': 0.0,
  'label': 3.0,
  'new_features': SparseVector(10, {}),
  'prediction': 3.0,
  'probability': DenseVector([0.1193, 0.0309, 0.0072, 0.3134, 0.0389, 0.0586, 0.0353, 0.0994, 0.0275, 0.0102, 0.0354, 0.0727, 0.0463, 0.0673, 0.0083, 0.0101, 0.0087, 0.0052, 0.0028, 0.0018, 0.0007]),
  'rawPrediction': DenseVector([1.7876, 0.4369, -1.0167, 2.7535, 0.6679, 1.0772, 0.5694, 1.6053, 0.3186, -0.6753, 0.5741, 1.2923, 0.8422, 1.2147, -0.8741, -0.6798, -0.8337, -1.3447, -1.9755, -2.4319, -3.308])
}


"""
trainingSummary = lr_model.summary
accuracy = trainingSummary.accuracy
falsePositiveRate = trainingSummary.weightedFalsePositiveRate
truePositiveRate = trainingSummary.weightedTruePositiveRate
fMeasure = trainingSummary.weightedFMeasure()
precision = trainingSummary.weightedPrecision
recall = trainingSummary.weightedRecall
print("Accuracy: %s\nFPR: %s\nTPR: %s\nF-measure: %s\nPrecision: %s\nRecall: %s"
       % (accuracy, falsePositiveRate, truePositiveRate, fMeasure, precision, recall))

print("F1: ", muti_evaluator.evaluate(predictions, {muti_evaluator.metricName: "f1"}))
print('Accuracy:', muti_evaluator.evaluate(predictions, {muti_evaluator.metricName: "accuracy"}))
print("WeightedPrecision: ", muti_evaluator.evaluate(predictions, {muti_evaluator.metricName: "weightedPrecision"}))
print('WeightedRecall:', muti_evaluator.evaluate(predictions, {muti_evaluator.metricName: "weightedRecall"}))
"""
Accuracy: 0.5722142891717895
FPR: 0.49458331350539386
TPR: 0.5722142891717896
F-measure: 0.45178187589615254
Precision: 0.4099984292307204
Recall: 0.5722142891717896
F1:  0.4535277999948288
Accuracy: 0.5737023385098797
WeightedPrecision:  0.4114306863886344
WeightedRecall: 0.5737023385098798

"""

# Downsampling

train_downsampled =(
      train_random.withColumn("Random", F.rand(seed=1000))
     .where((F.col("label") != 0) | ((F.col("label") == 0) & (F.col("Random")<0.2)))
     )
     
print_class_balance(train_downsampled,"train_downsampled")
test_downsampled=test_random
"""
train_downsampled
181204
    label  count     ratio
0    20.0    150  0.000828
1    19.0    376  0.002075
2    18.0    448  0.002472
3    17.0    805  0.004443
4    16.0   1221  0.006738
5    15.0   1315  0.007257
6    14.0   1613  0.008902
7    13.0   3178  0.017538
8    12.0   4520  0.024944
9    11.0   4818  0.026589
10   10.0   5428  0.029955
11    9.0   5525  0.030490
12    8.0   6946  0.038332
13    7.0   9146  0.050473
14    6.0  11064  0.061058
15    5.0  11258  0.062129
16    4.0  13884  0.076621
17    3.0  14112  0.077879
18    2.0  16505  0.091085
19    1.0  31977  0.176470
20    0.0  36915  0.203721



"""
cv_2 = CrossValidator(
                    estimator=lr,
                    estimatorParamMaps=lr_Grid,
                    evaluator=muti_evaluator,  
                    numFolds=10,seed=1000, parallelism=8 )
cv2_model = cv_2.fit(train_downsampled)   #train 
cv2_prediction = cv2_model.transform(test_downsampled) #test no change
        
print("------Cross-Validation performance of Logistic Regression-------")
print("F1: ", muti_evaluator.evaluate(cv2_prediction, {muti_evaluator.metricName: "f1"}))
print('Accuracy:', muti_evaluator.evaluate(cv2_prediction, {muti_evaluator.metricName: "accuracy"}))
print("WeightedPrecision: ", muti_evaluator.evaluate(cv2_prediction, {muti_evaluator.metricName: "weightedPrecision"}))
print('WeightedRecall:', muti_evaluator.evaluate(cv2_prediction, {muti_evaluator.metricName: "weightedRecall"}))
print("--------------------------------------------------------------")
"""
------Cross-Validation performance of Logistic Regression-------
F1:  0.45458205296526427
Accuracy: 0.46744818418031303
WeightedPrecision:  0.47608964371820306
WeightedRecall: 0.4674481841803129
--------------------------------------------------------------


"""
multi_data_features.select("Genre_Name").distinct().count()#21
multi_data_features.select("label").distinct().count()

label_stringIdx = StringIndexer(inputCol = "Genre_Name", outputCol = "label")
multi_data_features = label_stringIdx.fit(au_ge_data).transform(au_ge_data)


# Multiclass Classification metrics
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib; matplotlib.use('agg')
import matplotlib.pyplot as plt

lr_model = lr.fit(train_downsampled) 
lr_pred = lr_pred = lr_model.transform(test_downsampled)   # RDD
preds = np.array(lr_pred.select('prediction').collect())
actuals=np.array(test_downsampled.select('label').collect())


f = plt.figure(figsize=(15,12), dpi=80)
ax=plt.subplot()
cm = confusion_matrix(actuals, preds)
print(pd.DataFrame(cm))
"""
     0     1     2     3    4   5    6    7   8   9   10   11  12  13  14  15  16  17  18  19  20
0   30932  8791  1219  3926  136  14  877  543   0  14  47  160   0  16  26  37   0   2   0   0   0
1    1929  4262   800   744   17   0  167   54   0  14   8   21   0   7  21   5   0   0   0   0   0
2     571  1945  1306    94   11   0  101    9   0   7   3    5   0   0   7   2   0   0   0   0   0
3     725   769    45  1674    8   1  143   53   0   3  17   40   0   5   3  13   0   0   0   0   0
4    1618  1177   136   325   12   1  166   42   0   0  11   13   0   0   3   1   0   0   0   0   0
5    1167   880   129   460    5   3   71   27   0   1  19   21   0   1   0   5   0   0   0   0   0
6     788  1034   225   380   12   2  283   36   0   2   6   17   0   0   3   2   0   0   0   0   0
7    1297   372    25   419   14   1   48   54   0   0  18   12   0   0   2   1   0   0   0   0   0
8     961   425    40   232    4   2   89    8   0   1   3    5   0   0   4   0   0   0   0   0   0
9     208   763   192    51    9   0  102   13   0   3   1    0   1   0   2   0   0   0   0   0   0
10    607   293    17   340    2   0    8   15   0   0  21    5   0   0   2   3   0   0   0   0   0
11    358   296    26   410    2   1   37   27   0   0  19   68   0   0   1   1   0   0   0   0   0
12    389   344    37   335    4   1   16   30   0   0   6   16   0   0   3   0   0   0   0   0   0
13    180   104     3   409    1   0   10   15   0   0   3   12   0   9   1   0   0   0   0   0   0
14    123    63    81    47    3   0   43   10   0   1   1   20   0   0  46   0   0   0   0   0   0
15     55    47     3   163    0   0    0    4   0   1   3    5   0   1   0   6   0   0   0   0   0
16    120    38     4    98    0   0    4    9   0   0  11    7   0   2   0   9   0   0   0   0   0
17     64    38     8    67    0   1    3    2   0   0   0    4   0   2   1   3   0   0   0   0   0
18     27     7     1    48    0   1    1    2   0   0   1    2   0   0   0   3   0   0   0   0   0
19     28    25    11    11    0   0    1    1   0   0   0    3   0   0   1   0   0   0   0   0   0
20     16    19     2     8    0   0    1    2   0   0   0    0   0   0   0   0   0   0   0   0   0

"""

sns.heatmap(pd.DataFrame(cm), annot=True, fmt='d', cmap='YlGnBu', alpha=0.8, vmin=0)
ax.set_xlabel('Predicted')
ax.set_ylabel('Actual')
ax.set_title('Confusion Matrix of Multi-Class')
f.savefig(os.path.join(os.path.expanduser("~/MultiClass-ConfustionMetrics.png")), bbox_inches="tight")  # save as png
plt.close(f)

print(classification_report(actual, pred))
"""
              precision    recall  f1-score   support

        0.0       0.00      0.00      0.00     46740
        1.0       0.13      0.67      0.22      8049
        2.0       0.29      0.34      0.31      4061
        3.0       0.14      0.53      0.23      3499
        4.0       0.05      0.11      0.07      3505
        5.0       0.05      0.06      0.05      2789
        6.0       0.12      0.13      0.13      2790
        7.0       0.05      0.11      0.07      2263
        8.0       0.02      0.02      0.02      1774
        9.0       0.06      0.00      0.00      1345
       10.0       0.09      0.05      0.06      1313
       11.0       0.14      0.06      0.09      1246
       12.0       0.00      0.00      0.00      1181
       13.0       0.16      0.04      0.06       747
       14.0       0.37      0.11      0.17       438
       15.0       0.01      0.00      0.01       288
       16.0       0.00      0.00      0.00       302
       17.0       0.10      0.01      0.01       193
       18.0       0.00      0.00      0.00        93
       19.0       0.00      0.00      0.00        81
       20.0       0.00      0.00      0.00        48

avg / total       0.05      0.12      0.06     82745

"""
