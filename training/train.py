# First I resize images to 250 X 250

TRAIN_DIR = 'train'
TRAIN_FET = 'train_fet'
RESET = 100

from PIL import Image
import os

l = os.listdir(TRAIN_DIR)
n = 0
for file in l:
    try:
        im = Image.open(TRAIN_DIR+'/'+file)
        im = im.resize((250, 250), Image.BICUBIC)
        im.save(TRAIN_DIR+'/'+file)
        del im
    except:
        os.remove(TRAIN_DIR+'/'+file)
    if n%1000 == 0:
        print(n/len(l)*100)
    n += 1
	
# Here I extract DELF features from the images

from delf import feature_io
import os
import numpy as np
import pandas as pd
import multiprocessing as mp
from delfex import fetex

l = os.listdir(TRAIN_DIR)

for i in range(0, len(l),RESET):
    with open('list_images.txt', 'w') as f:
        [f.write(os.path.join(os.getcwd(), TRAIN_DIR, file)+'\n') for file in l[i:i+RESET]]
    p = mp.Process(target=fetex, args=(TRAIN_FET,))
    p.start()
    p.join()
    for file in l[i:i+RESET]:
        _, _, desc, _, _ = feature_io.ReadFromFile(TRAIN_FET+'/'+file.split('.')[0]+'.delf')
        np.save(TRAIN_FET+'/'+file.split('.')[0]+'.npy', desc)
		
# Next, loaded the features of all images into a spark rdd. This takes a long time

import os
import sys

import findspark
findspark.init()

from pyspark import SparkContext

sc = SparkContext(master="local[4]")

from pyspark.sql import SparkSession

spark = SparkSession \
    .builder \
    .appName("Python Spark SQL basic example") \
    .config("spark.some.config.option", "some-value") \
    .getOrCreate()
	
import os
import numpy as np
import pandas as pd

l = os.listdir(TRAIN_FET)
data = None
n = 0
for file in l:
    if file.endswith('.npy'):
        desc = np.load(TRAIN_FET+'/'+file)
        d = pd.DataFrame({'index' : list(range(n,n+desc.shape[0])), 
                          'file': [file.split('.')[0]]*desc.shape[0], 
                          'features' : desc.tolist()})
        n += desc.shape[0]
        if data is None:
            data = spark.createDataFrame(d).rdd
        else:
            p = spark.createDataFrame(d).rdd
            data = data.union(p)
			
# I wrote the rdd to a parquet file so that I don't lose it

from pyspark.ml.linalg import DenseVector

schema = data.map(lambda x: (DenseVector(x[0]),x[1],x[2],)).toDF(["features", "file", "index"])
schema.write.parquet("all_fet.parquet")

schema = spark.read.parquet("all_fet.parquet")

# Now I create the Bag of Visual Words representation using K-means

from pyspark.ml.clustering import KMeans
import time
K = 100
start = time.clock()
kmeans = KMeans(k=K)
print(time.clock()-start)
start = time.clock()
model = kmeans.fit(schema)
print(time.clock()-start)
start = time.clock()
centers = model.clusterCenters()
print(time.clock()-start)

# Next I create the Hamming Embedding Matrix

import numpy as np

d = 40
db = 64
G = np.random.randn(db, d)
P, _ = np.linalg.qr(G)
centers = np.array(centers)

from pyspark.ml.linalg import DenseVector
predictions = model.transform(schema)
df = predictions.rdd \
        .map(lambda x: (x[2], x[0], DenseVector(np.matmul(np.array(x[0]), P.T)), x[1], x[3])) \
        .toDF(["Index", "Features", "Projections", "File", "VisualWords"])
		
from pyspark import StorageLevel
from pyspark.sql import Row

df = df.persist(StorageLevel(True, True, False, False, 1))
df.createOrReplaceTempView("train_delf")
tau = np.zeros((K, db))

for l in range(K):
    ss = spark.sql("SELECT * FROM train_delf WHERE visual_words = %d" % l) \
            .rdd.persist(StorageLevel(True, True, False, False, 1))
    for h in range(db):
        tau[l,h] = ss.map(lambda x: Row(float(x[1][h]))) \
                        .toDF(["h"]) \
                        .approxQuantile("h", [0.5], 0.25)[0]
    ss.unpersist()
    print(l)
	
# Lastly, I generate binary signatures for all features for use at test time

from pyspark.ml.linalg import DenseVector
def binsig(z, c, tau):
    return DenseVector((z[0] > tau[c,:]))
	
df.rdd.map(lambda x: (x[0], x[1], x[2], x[3], x[4], binsig(x[1], x[4], tau),)) \
    .toDF(["Index", "Features", "Projections", "File", "VisualWords", "BinarySignature"]).show()
	
df.write.parquet("all_fet.parquet")