"""
Assignment 2 Data Processing
@author: Pengcheng Wu
Reference : some codes changed from the Credit Fraud, MovieLens Logistic Regression, Spark and Assignment Examples
"""

------------------------------------------------------------------------------------------------------------------------------------
####################################################### Data Processing #######################################################
------------------------------------------------------------------------------------------------------------------------------------

#Q1
#(a) Overview the structure of dataset
# Show the data sets details
hdfs dfs -ls /data/msd/
"""
drwxr-xr-x   - jsw93 supergroup          0 2021-09-29 10:35 /data/msd/audio
drwxr-xr-x   - jsw93 supergroup          0 2021-09-29 10:35 /data/msd/genre
drwxr-xr-x   - jsw93 supergroup          0 2021-09-29 10:28 /data/msd/main
drwxr-xr-x   - jsw93 supergroup          0 2021-09-29 10:35 /data/msd/tasteprofile
"""

hdfs dfs -ls -h /data/msd/audio/
hdfs dfs -ls -h /data/msd/audio/attributes
hdfs dfs -ls -h /data/msd/audio/features
hdfs dfs -ls -h /data/msd/audio/statistics

hdfs dfs -ls -h /data/msd/genre

hdfs dfs -ls -h /data/msd/main
hdfs dfs -ls -h /data/msd/main/summary

hdfs dfs -ls -h /data/msd/tasteprofile
hdfs dfs -ls -h /data/msd/tasteprofile/mismatches
hdfs dfs -ls -h /data/msd/tasteprofile/triplets.tsv

# tree directory 
tree /scratch-network/courses/2022/DATA420-22S1/data/msd/

#Show the data size
hdfs dfs -du -s -h -v /data/msd/
"""
SIZE    DISK_SPACE_CONSUMED_WITH_ALL_REPLICAS  FULL_PATH_NAME
12.9 G  103.5 G                                /data/msd
"""
hdfs dfs -du -h -v /data/msd/
"""
SIZE     DISK_SPACE_CONSUMED_WITH_ALL_REPLICAS  FULL_PATH_NAME
12.3 G   98.1 G                                 /data/msd/audio
30.1 M   241.0 M                                /data/msd/genre
174.4 M  1.4 G                                  /data/msd/main
490.4 M  3.8 G                                  /data/msd/tasteprofile
"""
hdfs dfs -du -h -v /data/msd/audio
hdfs dfs -du -h -v /data/msd/audio/attributes
hdfs dfs -du -h -v /data/msd/audio/features
hdfs dfs -du -h -v /data/msd/audio/statistics

hdfs dfs -du -h -v /data/msd/genre

hdfs dfs -du -h -v /data/msd/main
hdfs dfs -du -h -v /data/msd/main/summary

hdfs dfs -du -h -v /data/msd/tasteprofile
hdfs dfs -du -h -v /data/msd/tasteprofile/mismatches
hdfs dfs -du -h -v /data/msd/tasteprofile/triplets.tsv

#check files
#audio/attributes
for filename in `hdfs dfs -ls /data/msd/audio/attributes | awk '{print $NF}' `; do echo $filename; hdfs dfs -cat $filename |head -n5; done
hdfs dfs -cat "/data/msd/audio/attributes/*" | awk -F',' '{print $2}' | sort | uniq
#audio/features
hdfs dfs -cat "/data/msd/audio/features/msd-jmir-area-of-moments-all-v1.0.csv*" | awk -F',' '{print $2}' | sort | uniq
for filename in `hdfs dfs -ls /data/msd/audio/features/msd-jmir-area-of-moments-all-v1.0.csv | awk -F " " '{print $NF}'`; do echo $filename; hdfs dfs -cat $filename |gunzip |head -n5 |sort| uniq; done #Numeric
for filename in `hdfs dfs -ls /data/msd/audio/features/msd-jmir-lpc-all-v1.0.csv | awk '{print $NF}' `; do echo $filename; hdfs dfs -cat $filename |gunzip |head -n5; done
for filename in `hdfs dfs -ls /data/msd/audio/features/msd-jmir-methods-of-moments-all-v1.0.csv | awk '{print $NF}' `; do echo $filename; hdfs dfs -cat $filename |gunzip |head -n5; done
for filename in `hdfs dfs -ls /data/msd/audio/features/msd-jmir-mfcc-all-v1.0.csv | awk '{print $NF}' `; do echo $filename; hdfs dfs -cat $filename |gunzip |head -n5; done
for filename in `hdfs dfs -ls /data/msd/audio/features/msd-jmir-spectral-all-all-v1.0.csv | awk '{print $NF}' `; do echo $filename; hdfs dfs -cat $filename |gunzip |head -n5; done
for filename in `hdfs dfs -ls /data/msd/audio/features/msd-jmir-spectral-derivatives-all-all-v1.0.csv | awk '{print $NF}' `; do echo $filename; hdfs dfs -cat $filename |gunzip |head -n5; done
for filename in `hdfs dfs -ls /data/msd/audio/features/msd-marsyas-timbral-v1.0.csv | awk '{print $NF}' `; do echo $filename; hdfs dfs -cat $filename |gunzip |head -n5; done
for filename in `hdfs dfs -ls /data/msd/audio/features/msd-mvd-v1.0.csv | awk '{print $NF}' `; do echo $filename; hdfs dfs -cat $filename |gunzip |head -n5; done
for filename in `hdfs dfs -ls /data/msd/audio/features/msd-rh-v1.0.csv | awk '{print $NF}' `; do echo $filename; hdfs dfs -cat $filename |gunzip |head -n5; done
for filename in `hdfs dfs -ls /data/msd/audio/features/msd-rp-v1.0.csv | awk '{print $NF}' `; do echo $filename; hdfs dfs -cat $filename |gunzip |head -n5; done
for filename in `hdfs dfs -ls /data/msd/audio/features/msd-ssd-v1.0.csv | awk '{print $NF}' `; do echo $filename; hdfs dfs -cat $filename |gunzip |head -n5; done
for filename in `hdfs dfs -ls /data/msd/audio/features/msd-trh-v1.0.csv | awk '{print $NF}' `; do echo $filename; hdfs dfs -cat $filename |gunzip |head -n5; done
for filename in `hdfs dfs -ls /data/msd/audio/features/msd-tssd-v1.0.csv | awk '{print $NF}' `; do echo $filename; hdfs dfs -cat $filename |gunzip |head -n5; done

#audio/statistics
hdfs dfs -cat  /data/msd/audio/statistics/sample_properties.csv.gz|gunzip |head -n5 # string, character , numeric
#genre
for filename in `hdfs dfs -ls /data/msd/genre | awk '{print $NF}' `; do echo $filename; hdfs dfs -cat $filename |head -n5; done

#main
for filename in `hdfs dfs -ls /data/msd/main/summary | awk '{print $NF}' `; do echo $filename; hdfs dfs -cat $filename |gunzip |head -n5; done
#tasteprofile
for filename in `hdfs dfs -ls /data/msd/tasteprofile/mismatches | awk '{print $NF}' `; do echo $filename; hdfs dfs -cat $filename |head -n5; done
for filename in `hdfs dfs -ls /data/msd/tasteprofile/triplets.tsv | awk '{print $NF}' `; do echo $filename; hdfs dfs -cat $filename |gunzip |head -n5; done

#Q3 Counting the number of rows in each of the datasets.
# audio/attributes
for filename in `hdfs dfs -ls /data/msd/audio/attributes | awk '{print $NF}' `; do echo $filename; hdfs dfs -cat $filename |gunzip |wc -l; done
"""
/data/msd/audio/attributes/msd-jmir-area-of-moments-all-v1.0.attributes.csv
21
/data/msd/audio/attributes/msd-jmir-lpc-all-v1.0.attributes.csv
21
/data/msd/audio/attributes/msd-jmir-methods-of-moments-all-v1.0.attributes.csv
11
/data/msd/audio/attributes/msd-jmir-mfcc-all-v1.0.attributes.csv
27
/data/msd/audio/attributes/msd-jmir-spectral-all-all-v1.0.attributes.csv
17
/data/msd/audio/attributes/msd-jmir-spectral-derivatives-all-all-v1.0.attributes.csv
17
/data/msd/audio/attributes/msd-marsyas-timbral-v1.0.attributes.csv
125
/data/msd/audio/attributes/msd-mvd-v1.0.attributes.csv
421
/data/msd/audio/attributes/msd-rh-v1.0.attributes.csv
61
/data/msd/audio/attributes/msd-rp-v1.0.attributes.csv
1441
/data/msd/audio/attributes/msd-ssd-v1.0.attributes.csv
169
/data/msd/audio/attributes/msd-trh-v1.0.attributes.csv
421
/data/msd/audio/attributes/msd-tssd-v1.0.attributes.csv
1177
"""
#audio/features
for filename in `hdfs dfs -ls /data/msd/audio/features/msd-jmir-area-of-moments-all-v1.0.csv | awk -F " " '{print $NF}'`; do echo $filename; hdfs dfs -cat $filename |gunzip |wc -l; done
hdfs dfs -cat /data/msd/audio/features/msd-jmir-area-of-moments-all-v1.0.csv/part-* |gunzip|wc -l #994623
"""
/data/msd/audio/features/msd-jmir-area-of-moments-all-v1.0.csv/part-00000.csv.gz
33417
/data/msd/audio/features/msd-jmir-area-of-moments-all-v1.0.csv/part-00001.csv.gz
33066
/data/msd/audio/features/msd-jmir-area-of-moments-all-v1.0.csv/part-00002.csv.gz
33453
/data/msd/audio/features/msd-jmir-area-of-moments-all-v1.0.csv/part-00003.csv.gz
33952
/data/msd/audio/features/msd-jmir-area-of-moments-all-v1.0.csv/part-00004.csv.gz
33486
/data/msd/audio/features/msd-jmir-area-of-moments-all-v1.0.csv/part-00005.csv.gz
33569
/data/msd/audio/features/msd-jmir-area-of-moments-all-v1.0.csv/part-00006.csv.gz
33561
/data/msd/audio/features/msd-jmir-area-of-moments-all-v1.0.csv/part-00007.csv.gz
31719
"""
for filename in `hdfs dfs -ls /data/msd/audio/features/msd-jmir-lpc-all-v1.0.csv | awk '{print $NF}' `; do echo $filename; hdfs dfs -cat $filename |wc -l; done
hdfs dfs -cat /data/msd/audio/features/msd-jmir-lpc-all-v1.0.csv/part-* |gunzip|wc -l #994623
"""
/data/msd/audio/features/msd-jmir-lpc-all-v1.0.csv/part-00000.csv.gz
25932
/data/msd/audio/features/msd-jmir-lpc-all-v1.0.csv/part-00001.csv.gz
25687
/data/msd/audio/features/msd-jmir-lpc-all-v1.0.csv/part-00002.csv.gz
25395
/data/msd/audio/features/msd-jmir-lpc-all-v1.0.csv/part-00003.csv.gz
25783
/data/msd/audio/features/msd-jmir-lpc-all-v1.0.csv/part-00004.csv.gz
25564
/data/msd/audio/features/msd-jmir-lpc-all-v1.0.csv/part-00005.csv.gz
25546
/data/msd/audio/features/msd-jmir-lpc-all-v1.0.csv/part-00006.csv.gz
25240
/data/msd/audio/features/msd-jmir-lpc-all-v1.0.csv/part-00007.csv.gz
24387
"""
for filename in `hdfs dfs -ls /data/msd/audio/features/msd-jmir-methods-of-moments-all-v1.0.csv | awk '{print $NF}' `; do echo $filename; hdfs dfs -cat $filename |wc -l; done
hdfs dfs -cat /data/msd/audio/features/msd-jmir-methods-of-moments-all-v1.0.csv/part-* |gunzip|wc -l #994623
"""
/data/msd/audio/features/msd-jmir-methods-of-moments-all-v1.0.csv/part-00000.csv.gz
19691
/data/msd/audio/features/msd-jmir-methods-of-moments-all-v1.0.csv/part-00001.csv.gz
19811
/data/msd/audio/features/msd-jmir-methods-of-moments-all-v1.0.csv/part-00002.csv.gz
19886
/data/msd/audio/features/msd-jmir-methods-of-moments-all-v1.0.csv/part-00003.csv.gz
19314
/data/msd/audio/features/msd-jmir-methods-of-moments-all-v1.0.csv/part-00004.csv.gz
19535
/data/msd/audio/features/msd-jmir-methods-of-moments-all-v1.0.csv/part-00005.csv.gz
19859
/data/msd/audio/features/msd-jmir-methods-of-moments-all-v1.0.csv/part-00006.csv.gz
19527
/data/msd/audio/features/msd-jmir-methods-of-moments-all-v1.0.csv/part-00007.csv.gz
19099
"""
for filename in `hdfs dfs -ls /data/msd/audio/features/msd-jmir-mfcc-all-v1.0.csv | awk '{print $NF}' `; do echo $filename; hdfs dfs -cat $filename |wc -l; done
hdfs dfs -cat /data/msd/audio/features/msd-jmir-mfcc-all-v1.0.csv/part-* |gunzip|wc -l #994623
"""
/data/msd/audio/features/msd-jmir-mfcc-all-v1.0.csv/part-00000.csv.gz
32472
/data/msd/audio/features/msd-jmir-mfcc-all-v1.0.csv/part-00001.csv.gz
32614
/data/msd/audio/features/msd-jmir-mfcc-all-v1.0.csv/part-00002.csv.gz
32735
/data/msd/audio/features/msd-jmir-mfcc-all-v1.0.csv/part-00003.csv.gz
32464
/data/msd/audio/features/msd-jmir-mfcc-all-v1.0.csv/part-00004.csv.gz
32675
/data/msd/audio/features/msd-jmir-mfcc-all-v1.0.csv/part-00005.csv.gz
32563
/data/msd/audio/features/msd-jmir-mfcc-all-v1.0.csv/part-00006.csv.gz
32333
/data/msd/audio/features/msd-jmir-mfcc-all-v1.0.csv/part-00007.csv.gz
30973
"""
for filename in `hdfs dfs -ls /data/msd/audio/features/msd-jmir-spectral-all-all-v1.0.csv | awk '{print $NF}' `; do echo $filename; hdfs dfs -cat $filename |wc -l; done
hdfs dfs -cat /data/msd/audio/features/msd-jmir-spectral-all-all-v1.0.csv/part-* |gunzip|wc -l #994623
"""
/data/msd/audio/features/msd-jmir-spectral-all-all-v1.0.csv/part-00000.csv.gz
24959
/data/msd/audio/features/msd-jmir-spectral-all-all-v1.0.csv/part-00001.csv.gz
25191
/data/msd/audio/features/msd-jmir-spectral-all-all-v1.0.csv/part-00002.csv.gz
25480
/data/msd/audio/features/msd-jmir-spectral-all-all-v1.0.csv/part-00003.csv.gz
24958
/data/msd/audio/features/msd-jmir-spectral-all-all-v1.0.csv/part-00004.csv.gz
25380
/data/msd/audio/features/msd-jmir-spectral-all-all-v1.0.csv/part-00005.csv.gz
25086
/data/msd/audio/features/msd-jmir-spectral-all-all-v1.0.csv/part-00006.csv.gz
25176
/data/msd/audio/features/msd-jmir-spectral-all-all-v1.0.csv/part-00007.csv.gz
24206
"""
for filename in `hdfs dfs -ls /data/msd/audio/features/msd-jmir-spectral-derivatives-all-all-v1.0.csv | awk '{print $NF}' `; do echo $filename; hdfs dfs -cat $filename |wc -l; done
hdfs dfs -cat /data/msd/audio/features/msd-jmir-spectral-derivatives-all-all-v1.0.csv/part-* |gunzip|wc -l #994623
"""
/data/msd/audio/features/msd-jmir-spectral-derivatives-all-all-v1.0.csv/part-00000.csv.gz
24959
/data/msd/audio/features/msd-jmir-spectral-derivatives-all-all-v1.0.csv/part-00001.csv.gz
25191
/data/msd/audio/features/msd-jmir-spectral-derivatives-all-all-v1.0.csv/part-00002.csv.gz
25480
/data/msd/audio/features/msd-jmir-spectral-derivatives-all-all-v1.0.csv/part-00003.csv.gz
24958
/data/msd/audio/features/msd-jmir-spectral-derivatives-all-all-v1.0.csv/part-00004.csv.gz
25380
/data/msd/audio/features/msd-jmir-spectral-derivatives-all-all-v1.0.csv/part-00005.csv.gz
25086
/data/msd/audio/features/msd-jmir-spectral-derivatives-all-all-v1.0.csv/part-00006.csv.gz
25176
/data/msd/audio/features/msd-jmir-spectral-derivatives-all-all-v1.0.csv/part-00007.csv.gz
24206
"""
for filename in `hdfs dfs -ls /data/msd/audio/features/msd-marsyas-timbral-v1.0.csv | awk '{print $NF}' `; do echo $filename; hdfs dfs -cat $filename |wc -l; done
hdfs dfs -cat /data/msd/audio/features/msd-marsyas-timbral-v1.0.csv/part-* |gunzip|wc -l #995001
"""
/data/msd/audio/features/msd-marsyas-timbral-v1.0.csv/part-00000.csv.gz
211585
/data/msd/audio/features/msd-marsyas-timbral-v1.0.csv/part-00001.csv.gz
211830
/data/msd/audio/features/msd-marsyas-timbral-v1.0.csv/part-00002.csv.gz
212396
/data/msd/audio/features/msd-marsyas-timbral-v1.0.csv/part-00003.csv.gz
210986
/data/msd/audio/features/msd-marsyas-timbral-v1.0.csv/part-00004.csv.gz
211635
/data/msd/audio/features/msd-marsyas-timbral-v1.0.csv/part-00005.csv.gz
211093
/data/msd/audio/features/msd-marsyas-timbral-v1.0.csv/part-00006.csv.gz
211955
/data/msd/audio/features/msd-marsyas-timbral-v1.0.csv/part-00007.csv.gz
203682
"""
for filename in `hdfs dfs -ls /data/msd/audio/features/msd-mvd-v1.0.csv | awk '{print $NF}' `; do echo $filename; hdfs dfs -cat $filename |wc -l; done
hdfs dfs -cat /data/msd/audio/features/msd-mvd-v1.0.csv/part-* |gunzip|wc -l #994188
"""
/data/msd/audio/features/msd-mvd-v1.0.csv/part-00000.csv.gz
663899
/data/msd/audio/features/msd-mvd-v1.0.csv/part-00001.csv.gz
665675
/data/msd/audio/features/msd-mvd-v1.0.csv/part-00002.csv.gz
663516
/data/msd/audio/features/msd-mvd-v1.0.csv/part-00003.csv.gz
663391
/data/msd/audio/features/msd-mvd-v1.0.csv/part-00004.csv.gz
662462
/data/msd/audio/features/msd-mvd-v1.0.csv/part-00005.csv.gz
663993
/data/msd/audio/features/msd-mvd-v1.0.csv/part-00006.csv.gz
663818
/data/msd/audio/features/msd-mvd-v1.0.csv/part-00007.csv.gz
633908
"""
for filename in `hdfs dfs -ls /data/msd/audio/features/msd-rh-v1.0.csv | awk '{print $NF}' `; do echo $filename; hdfs dfs -cat $filename |wc -l; done
hdfs dfs -cat /data/msd/audio/features/msd-rh-v1.0.csv/part-* |gunzip|wc -l #994188
"""
/data/msd/audio/features/msd-rh-v1.0.csv/part-00000.csv.gz
113134
/data/msd/audio/features/msd-rh-v1.0.csv/part-00001.csv.gz
112560
/data/msd/audio/features/msd-rh-v1.0.csv/part-00002.csv.gz
112892
/data/msd/audio/features/msd-rh-v1.0.csv/part-00003.csv.gz
113354
/data/msd/audio/features/msd-rh-v1.0.csv/part-00004.csv.gz
113032
/data/msd/audio/features/msd-rh-v1.0.csv/part-00005.csv.gz
112661
/data/msd/audio/features/msd-rh-v1.0.csv/part-00006.csv.gz
113230
/data/msd/audio/features/msd-rh-v1.0.csv/part-00007.csv.gz
108668
"""
for filename in `hdfs dfs -ls /data/msd/audio/features/msd-rp-v1.0.csv | awk '{print $NF}' `; do echo $filename; hdfs dfs -cat $filename |wc -l; done
hdfs dfs -cat /data/msd/audio/features/msd-rp-v1.0.csv/part-* |gunzip|wc -l #994188
"""
/data/msd/audio/features/msd-rp-v1.0.csv/part-00000.csv.gz
2179745
/data/msd/audio/features/msd-rp-v1.0.csv/part-00001.csv.gz
2181398
/data/msd/audio/features/msd-rp-v1.0.csv/part-00002.csv.gz
2181997
/data/msd/audio/features/msd-rp-v1.0.csv/part-00003.csv.gz
2185657
/data/msd/audio/features/msd-rp-v1.0.csv/part-00004.csv.gz
2184739
/data/msd/audio/features/msd-rp-v1.0.csv/part-00005.csv.gz
2183239
/data/msd/audio/features/msd-rp-v1.0.csv/part-00006.csv.gz
2181977
/data/msd/audio/features/msd-rp-v1.0.csv/part-00007.csv.gz
2081171
"""
for filename in `hdfs dfs -ls /data/msd/audio/features/msd-ssd-v1.0.csv | awk '{print $NF}' `; do echo $filename; hdfs dfs -cat $filename |wc -l; done
hdfs dfs -cat /data/msd/audio/features/msd-ssd-v1.0.csv/part-* |gunzip|wc -l #994188
"""
/data/msd/audio/features/msd-ssd-v1.0.csv/part-00000.csv.gz
286349
/data/msd/audio/features/msd-ssd-v1.0.csv/part-00001.csv.gz
285764
/data/msd/audio/features/msd-ssd-v1.0.csv/part-00002.csv.gz
284379
/data/msd/audio/features/msd-ssd-v1.0.csv/part-00003.csv.gz
284436
/data/msd/audio/features/msd-ssd-v1.0.csv/part-00004.csv.gz
285610
/data/msd/audio/features/msd-ssd-v1.0.csv/part-00005.csv.gz
286054
/data/msd/audio/features/msd-ssd-v1.0.csv/part-00006.csv.gz
285284
/data/msd/audio/features/msd-ssd-v1.0.csv/part-00007.csv.gz
272151
"""
for filename in `hdfs dfs -ls /data/msd/audio/features/msd-trh-v1.0.csv | awk '{print $NF}' `; do echo $filename; hdfs dfs -cat $filename |wc -l; done
hdfs dfs -cat /data/msd/audio/features/msd-trh-v1.0.csv/part-* |gunzip|wc -l #994188
"""
/data/msd/audio/features/msd-trh-v1.0.csv/part-00000.csv.gz
675893
/data/msd/audio/features/msd-trh-v1.0.csv/part-00001.csv.gz
676897
/data/msd/audio/features/msd-trh-v1.0.csv/part-00002.csv.gz
675325
/data/msd/audio/features/msd-trh-v1.0.csv/part-00003.csv.gz
675019
/data/msd/audio/features/msd-trh-v1.0.csv/part-00004.csv.gz
675434
/data/msd/audio/features/msd-trh-v1.0.csv/part-00005.csv.gz
676747
/data/msd/audio/features/msd-trh-v1.0.csv/part-00006.csv.gz
675558
/data/msd/audio/features/msd-trh-v1.0.csv/part-00007.csv.gz
643112
"""
for filename in `hdfs dfs -ls /data/msd/audio/features/msd-tssd-v1.0.csv | awk '{print $NF}' `; do echo $filename; hdfs dfs -cat $filename |wc -l; done
hdfs dfs -cat /data/msd/audio/features/msd-tssd-v1.0.csv/part-* |gunzip|wc -l #994623
"""
/data/msd/audio/features/msd-tssd-v1.0.csv/part-00000.csv.gz
1860302
/data/msd/audio/features/msd-tssd-v1.0.csv/part-00001.csv.gz
1862402
/data/msd/audio/features/msd-tssd-v1.0.csv/part-00002.csv.gz
1862556
/data/msd/audio/features/msd-tssd-v1.0.csv/part-00003.csv.gz
1860987
/data/msd/audio/features/msd-tssd-v1.0.csv/part-00004.csv.gz
1863542
/data/msd/audio/features/msd-tssd-v1.0.csv/part-00005.csv.gz
1862813
/data/msd/audio/features/msd-tssd-v1.0.csv/part-00006.csv.gz
1861818
/data/msd/audio/features/msd-tssd-v1.0.csv/part-00007.csv.gz
1776530
"""
#audio/statistics
for filename in `hdfs dfs -ls /data/msd/audio/statistics | awk '{print $NF}' `; do echo $filename; hdfs dfs -cat $filename |wc -l; done #183262
#genre
hdfs dfs -cat /data/msd/genre/msd-MAGD-genreAssignment.tsv|wc -l #422714
hdfs dfs -cat /data/msd/genre/msd-MASD-styleAssignment.tsv|wc -l #273936
hdfs dfs -cat /data/msd/genre/msd-topMAGD-genreAssignment.tsv|wc -l #406427
#main
hdfs dfs -cat /data/msd/main/summary/analysis.csv.gz |wc -l #239762
#tasteprofile
for filename in `hdfs dfs -ls /data/msd/tasteprofile/mismatches | awk '{print $NF}' `; do echo $filename; hdfs dfs -cat $filename |wc -l; done
"""
/data/msd/tasteprofile/mismatches/sid_matches_manually_accepted.txt
938
/data/msd/tasteprofile/mismatches/sid_mismatches.txt
19094
"""
for filename in `hdfs dfs -ls /data/msd/tasteprofile/triplets.tsv | awk '{print $NF}' `; do echo $filename; hdfs dfs -cat $filename |wc -l; done
"""
/data/msd/tasteprofile/triplets.tsv/part-00000.tsv.gz
210041
/data/msd/tasteprofile/triplets.tsv/part-00001.tsv.gz
209199
/data/msd/tasteprofile/triplets.tsv/part-00002.tsv.gz
208896
/data/msd/tasteprofile/triplets.tsv/part-00003.tsv.gz
209050
/data/msd/tasteprofile/triplets.tsv/part-00004.tsv.gz
209315
/data/msd/tasteprofile/triplets.tsv/part-00005.tsv.gz
210353
/data/msd/tasteprofile/triplets.tsv/part-00006.tsv.gz
209326
/data/msd/tasteprofile/triplets.tsv/part-00007.tsv.gz
208540
"""



#Q2-(a) Reference from presetting codes

# Python and pyspark modules required
import sys
from pyspark import SparkContext
from pyspark.sql import SparkSession
from pyspark.sql.types import *
from pyspark.sql.functions import *
from pretty import SparkPretty  # download pretty.py from LEARN
pretty = SparkPretty(limit=5)
# Required to allow the file to be submitted and run using spark-submit instead
# of using pyspark interactively
spark = SparkSession.builder.getOrCreate()
sc = SparkContext.getOrCreate()


#define mismat
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
#matches_manually_accepted.cache()
matches_manually_accepted.show(10, 40)
"""
+------------------+-----------------+----------------------------------------+------------------+----------------------------------------+----------------------------------------+
|           song_id|      song_artist|                              song_title|          track_id|                            track_artist|                             track_title|
+------------------+-----------------+----------------------------------------+------------------+----------------------------------------+----------------------------------------+
|SOFQHZM12A8C142342|     Josipa Lisac|                                 razloga|TRMWMFG128F92FFEF2|                            Lisac Josipa|                            1000 razloga|
|SODXUTF12AB018A3DA|       Lutan Fyah|     Nuh Matter the Crisis Feat. Midnite|TRMWPCD12903CCE5ED|                                 Midnite|                   Nah Matter the Crisis|
|SOASCRF12A8C1372E6|Gaetano Donizetti|L'Elisir d'Amore: Act Two: Come sen v...|TRMHIPJ128F426A2E2|Gianandrea Gavazzeni_ Orchestra E Cor...|L'Elisir D'Amore_ Act 2: Come Sen Va ...|
|SOITDUN12A58A7AACA|     C.J. Chenier|                               Ay, Ai Ai|TRMHXGK128F42446AB|                         Clifton Chenier|                               Ay_ Ai Ai|
|SOLZXUM12AB018BE39|           許志安|                                男人最痛|TRMRSOF12903CCF516|                                Andy Hui|                        Nan Ren Zui Tong|
|SOTJTDT12A8C13A8A6|                S|                                       h|TRMNKQE128F427C4D8|                             Sammy Hagar|                 20th Century Man (Live)|
|SOGCVWB12AB0184CE2|                H|                                       Y|TRMUNCZ128F932A95D|                                Hawkwind|                25 Years (Alternate Mix)|
|SOKDKGD12AB0185E9C|     影山ヒロノブ|Cha-La Head-Cha-La (2005 ver./DRAGON ...|TRMOOAH12903CB4B29|                        Takahashi Hiroki|Maka fushigi adventure! (2005 Version...|
|SOPPBXP12A8C141194|    Αντώνης Ρέμος|                        O Trellos - Live|TRMXJDS128F42AE7CF|                           Antonis Remos|                               O Trellos|
|SODQSLR12A8C133A01|    John Williams|Concerto No. 1 for Guitar and String ...|TRWHMXN128F426E03C|               English Chamber Orchestra|II. Andantino siciliano from Concerto...|
+------------------+-----------------+----------------------------------------+------------------+----------------------------------------+----------------------------------------+
"""
print(matches_manually_accepted.count())  # 488

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
mismatches.cache()
mismatches.show(10, 40)
"""
+------------------+-------------------+----------------------------------------+------------------+--------------+----------------------------------------+
|           song_id|        song_artist|                              song_title|          track_id|  track_artist|                             track_title|
+------------------+-------------------+----------------------------------------+------------------+--------------+----------------------------------------+
|SOUMNSI12AB0182807|Digital Underground|                        The Way We Swing|TRMMGKQ128F9325E10|      Linkwood|           Whats up with the Underground|
|SOCMRBE12AB018C546|         Jimmy Reed|The Sun Is Shining (Digitally Remaste...|TRMMREB12903CEB1B1|    Slim Harpo|               I Got Love If You Want It|
|SOLPHZY12AC468ABA8|      Africa HiTech|                                Footstep|TRMMBOC12903CEB46E|Marcus Worgull|                 Drumstern (BONUS TRACK)|
|SONGHTM12A8C1374EF|     Death in Vegas|                            Anita Berber|TRMMITP128F425D8D0|     Valen Hsu|                                  Shi Yi|
|SONGXCA12A8C13E82E| Grupo Exterminador|                           El Triunfador|TRMMAYZ128F429ECE6|     I Ribelli|                               Lei M'Ama|
|SOMBCRC12A67ADA435|      Fading Friend|                             Get us out!|TRMMNVU128EF343EED|     Masterboy|                      Feel The Heat 2000|
|SOTDWDK12A8C13617B|       Daevid Allen|                              Past Lives|TRMMNCZ128F426FF0E| Bhimsen Joshi|            Raga - Shuddha Sarang_ Aalap|
|SOEBURP12AB018C2FB|  Cristian Paduraru|                              Born Again|TRMMPBS12903CE90E1|     Yespiring|                          Journey Stages|
|SOSRJHS12A6D4FDAA3|         Jeff Mills|                      Basic Human Design|TRMWMEL128F421DA68|           M&T|                           Drumsettester|
|SOIYAAQ12A6D4F954A|           Excepter|                                      OG|TRMWHRI128F147EA8E|    The Fevers|Não Tenho Nada (Natchs Scheint Die So...|
+------------------+-------------------+----------------------------------------+------------------+--------------+----------------------------------------+
"""
print(mismatches.count())  # 19094

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
triplets.cache()
triplets.show(10, 50)
"""
+----------------------------------------+------------------+-----+
|                                 user_id|           song_id|plays|
+----------------------------------------+------------------+-----+
|f1bfc2a4597a3642f232e7a4e5d5ab2a99cf80e5|SOQEFDN12AB017C52B|    1|
|f1bfc2a4597a3642f232e7a4e5d5ab2a99cf80e5|SOQOIUJ12A6701DAA7|    2|
|f1bfc2a4597a3642f232e7a4e5d5ab2a99cf80e5|SOQOKKD12A6701F92E|    4|
|f1bfc2a4597a3642f232e7a4e5d5ab2a99cf80e5|SOSDVHO12AB01882C7|    1|
|f1bfc2a4597a3642f232e7a4e5d5ab2a99cf80e5|SOSKICX12A6701F932|    1|
|f1bfc2a4597a3642f232e7a4e5d5ab2a99cf80e5|SOSNUPV12A8C13939B|    1|
|f1bfc2a4597a3642f232e7a4e5d5ab2a99cf80e5|SOSVMII12A6701F92D|    1|
|f1bfc2a4597a3642f232e7a4e5d5ab2a99cf80e5|SOTUNHI12B0B80AFE2|    1|
|f1bfc2a4597a3642f232e7a4e5d5ab2a99cf80e5|SOTXLTZ12AB017C535|    1|
|f1bfc2a4597a3642f232e7a4e5d5ab2a99cf80e5|SOTZDDX12A6701F935|    1|
+----------------------------------------+------------------+-----+
"""
# filter the triplets
mismatches_not_accepted = mismatches.join(matches_manually_accepted, on="song_id", how="left_anti")
triplets_not_mismatched = triplets.join(mismatches_not_accepted, on="song_id", how="left_anti")
triplets_not_mismatched.show(10,50)
"""
+------------------+----------------------------------------+-----+
|           song_id|                                 user_id|plays|
+------------------+----------------------------------------+-----+
|SOQEFDN12AB017C52B|f1bfc2a4597a3642f232e7a4e5d5ab2a99cf80e5|    1|
|SOQOIUJ12A6701DAA7|f1bfc2a4597a3642f232e7a4e5d5ab2a99cf80e5|    2|
|SOQOKKD12A6701F92E|f1bfc2a4597a3642f232e7a4e5d5ab2a99cf80e5|    4|
|SOSDVHO12AB01882C7|f1bfc2a4597a3642f232e7a4e5d5ab2a99cf80e5|    1|
|SOSKICX12A6701F932|f1bfc2a4597a3642f232e7a4e5d5ab2a99cf80e5|    1|
|SOSNUPV12A8C13939B|f1bfc2a4597a3642f232e7a4e5d5ab2a99cf80e5|    1|
|SOSVMII12A6701F92D|f1bfc2a4597a3642f232e7a4e5d5ab2a99cf80e5|    1|
|SOTUNHI12B0B80AFE2|f1bfc2a4597a3642f232e7a4e5d5ab2a99cf80e5|    1|
|SOTXLTZ12AB017C535|f1bfc2a4597a3642f232e7a4e5d5ab2a99cf80e5|    1|
|SOTZDDX12A6701F935|f1bfc2a4597a3642f232e7a4e5d5ab2a99cf80e5|    1|
+------------------+----------------------------------------+-----+
"""
print(triplets.count())         # 48373586
print(triplets_not_mismatched.count())  # 45795111


#Q2-(b)
# check all the unique datatype of attribute:
## hdfs dfs -cat "/data/msd/audio/attributes/*" | awk -F',' '{print $2}' | sort | uniq
"""
NUMERIC
real
real
string
string
STRING
"""

# Note that the attribute files and feature datasets share the same prefix and that the attribute types are named consistently.

audio_attribute_type_mapping = {
  "NUMERIC": DoubleType(),
  "real": DoubleType(),
  "string": StringType(),
  "STRING": StringType()
}

audio_dataset_names = [
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

audio_dataset_schemas = {}
for audio_dataset_name in audio_dataset_names:
  print(audio_dataset_name)

  audio_dataset_path = f"/scratch-network/courses/2022/DATA420-22S1/data/msd/audio/attributes/{audio_dataset_name}.attributes.csv"
  with open(audio_dataset_path, "r") as f:
    rows = [line.strip().split(",") for line in f.readlines()]
 
  """
  feature_fullname = [Statistical_Spectrum_Descriptors, Rhythm_Histograms, Temporal_Statistical_Spectrum_Descriptors, Temporal_Rhythm_Histograms, 
                      Modulation_Frequency_Variance, 
                      MARSYAS_Timbral_features,
                      Spectral_Centroid, Spectral_Rolloff_Point, Spectral_Flux, Compactness, 
                      Spectral_Variability, Root_Mean_Square, Zero_Crossings, Fraction_of_Low_Energy_Windows, 
                      Low-level_features_derivatives, Method_of_Moments, Area_of_Moments, Linear_Predictive_Coding, MFCC_features]
  
  """
  """
  rows[-1][0] = "track_id"
  for i, row in enumerate(rows[0:-1]):
    row[0] = f"feature_{i:04d}"
  """
  audio_dataset_schemas[audio_dataset_name] = StructType([
  StructField(row[0], audio_attribute_type_mapping[row[1]], True) for row in rows
  ])
  
  print(audio_dataset_schemas[audio_dataset_name])

