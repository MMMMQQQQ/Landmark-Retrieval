# Configuration Constants

TRAIN_IMAGE_DIR = 'train'	# directory for training images
TRAIN_FET_DIR = 'train_fet'	# directory for training image features  stored as numpy arrays
RESET = 100					# How many images to extract features from at a time. Depends on GPU RAM
K = 10000					# Number of Cluster Centres.
d = 40						# Length of DELF local feature vector. DO NOT CHANGE
db = 64						# Number of bits for binary signatures

# Imports

from PIL import Image
import os
from delf import feature_io
import numpy as np
import pandas as pd
import multiprocessing as mp
from delfex import fetex
import sys
import findspark
from pyspark import SparkContext, SparkConf
from pyspark.sql import SparkSession
import time
from pyspark.ml.linalg import DenseVector
from pyspark import StorageLevel
from pyspark.ml.clustering import KMeans
from pyspark.ml.clustering import KMeansModel
from pyspark.sql import Row

# First I resize images to 250 X 250

l = os.listdir(TRAIN_IMAGE_DIR)
n = 0
for file in l:
    try:
        im = Image.open(TRAIN_IMAGE_DIR+'/'+file)
        im = im.resize((250, 250), Image.BICUBIC)
        im.save(TRAIN_IMAGE_DIR+'/'+file)
        del im
    except:
        os.remove(TRAIN_IMAGE_DIR+'/'+file)
    if n%1000 == 0:
        print(n/len(l)*100)
    n += 1
	
# Here I extract DELF features from the images

l = os.listdir(TRAIN_IMAGE_DIR)

for i in range(0, len(l),RESET):
    with open('list_images.txt', 'w') as f:
        [f.write(os.path.join(os.getcwd(), TRAIN_IMAGE_DIR, file)+'\n') for file in l[i:i+RESET]]
    p = mp.Process(target=fetex, args=(TRAIN_FET_DIR,))
    p.start()
    p.join()
    for file in l[i:i+RESET]:
        _, _, desc, _, _ = feature_io.ReadFromFile(TRAIN_FET_DIR+'/'+file.split('.')[0]+'.delf')
        np.save(TRAIN_FET_DIR+'/'+file.split('.')[0]+'.npy', desc)
		
# Next, loaded the features of all images into a spark rdd. This takes a long time

findspark.init()

conf = SparkConf().setAppName("App")
conf = (conf.setMaster('local[*]')
        .set('spark.executor.memory', '1G')
        .set('spark.driver.memory', '12G'))
sc = SparkContext(conf=conf)

spark = SparkSession \
    .builder \
    .appName("Python Spark SQL basic example") \
    .config("spark.some.config.option", "some-value") \
    .getOrCreate()

n = 0
k = 0
l = os.listdir(TRAIN_FET_DIR)
start = time.clock()
data = None
pan = None
for file in l:
    if file.endswith('.npy'):
        desc = np.load(TRAIN_FET_DIR+'/'+file)
        if pan is None:
            pan = pd.DataFrame({'index' : list(range(n,n+desc.shape[0])),
                                'file': [file.split('.')[0]]*desc.shape[0], 
                                'features' : desc.tolist()})
        else:
            pan = pd.concat([pan, pd.DataFrame({'index' : list(range(n,n+desc.shape[0])),
                                                'file': [file.split('.')[0]]*desc.shape[0],
                                                'features' : desc.tolist()})])
        if k%1000==0:
            print(k)
            if not os.path.isfile('train_fet.csv'):
                pan.to_csv('train_fet.csv', mode='w', index=False, header=True)
            else:
                pan.to_csv('train_fet.csv', mode='a', index=False, header=False)
            del pan
            pan = None
        k += 1
        n += desc.shape[0]

if k%1000!=0:
    if not os.path.isfile('train_fet.csv'):
        pan.to_csv('train_fet.csv', mode='w', index=False, header=True)
    else:
        pan.to_csv('train_fet.csv', mode='a', index=False, header=False)
    del pan
    pan = None
print(time.clock()-start)
			
# I converted csv to a parquet file to save space and time

def lis(x):
    return [float(i) for i in x[1:-1].split(',')]

spark.read.load("train_fet.csv", format="csv", inferSchema="true", header="true").rdd \
          .map(lambda x: (x[2], x[1], DenseVector(lis(x[0])))) \
          .toDF(["index", "file", "features"])
          .write.parquet("train_fet.parquet")

# Now I create the Bag of Visual Words representation using K-means

schema = spark.read.parquet("train_fet.parquet").persist(StorageLevel(True, True, False, False, 1))
start = time.clock()
kmeans = KMeans(k=K, initMode='random')
print(time.clock()-start)
start = time.clock()
model = kmeans.fit(schema)
print(time.clock()-start)
start = time.clock()
centers = model.clusterCenters()
print(time.clock()-start)
model.save('KmeansModel')

# Next I create the Hamming Embedding Matrix

G = np.random.randn(db, d)
P, _ = np.linalg.qr(G)
np.save('P.npy', P)

predictions = model.transform(schema)
df = predictions.rdd \
    .map(lambda x: (x[2], x[0], DenseVector(np.matmul(np.array(x[0]), P.T)), x[1], x[3])) \
    .toDF(["Index", "Features", "Projections", "File", "VisualWords"])

df = df.persist(StorageLevel(True, True, False, False, 1))
df.createOrReplaceTempView("train_delf")
tau = np.zeros((K, db))

for l in range(K):
    ss = spark.sql("SELECT Projections FROM train_delf WHERE visual_words = %d" % l) \
            .rdd.persist(StorageLevel(True, True, False, False, 1))
    for h in range(db):
        tau[l,h] = ss.map(lambda x: Row(float(x[0][h]))) \
                        .toDF(["h"]) \
                        .approxQuantile("h", [0.5], 0.25)[0]
    ss.unpersist()
    print(l)

np.save('tau.npy', tau)
	
# Then, I generate binary signatures for all features for use at test time

def binsig(z, c, tau):
    return DenseVector((z > tau[c,:]))
	
df = df.rdd.map(lambda x: (x[0], x[3], x[4], binsig(x[2], x[4], tau),)) \
    .toDF(["Index", "File", "VisualWords", "BinarySignature"])
	
df.write.parquet("index_fet.parquet")

# I also save the tables

df.createOrReplaceTempView("train_delf")

for i in range(K):
    spark.sql("SELECT File, BinarySignature FROM train_delf WHERE VisualWords = %d" % i) \
    .toPandas().to_pickle("bintable/train%d.pkl" % i)