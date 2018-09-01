# First I resize images to 250 X 250

TEST_DIR = 'train'
TEST_FET = 'train_fet'
RESET = 100

from PIL import Image
import os

l = os.listdir(TEST_DIR)
n = 0
for file in l:
    try:
        im = Image.open(TEST_DIR+'/'+file)
        im = im.resize((250, 250), Image.BICUBIC)
        im.save(TEST_DIR+'/'+file)
        del im
    except:
        os.remove(TEST_DIR+'/'+file)
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

l = os.listdir(TEST_DIR)

for i in range(0, len(l),RESET):
    with open('list_images.txt', 'w') as f:
        [f.write(os.path.join(os.getcwd(), TEST_DIR, file)+'\n') for file in l[i:i+RESET]]
    p = mp.Process(target=fetex, args=(TEST_FET,))
    p.start()
    p.join()
    for file in l[i:i+RESET]:
        _, _, desc, _, _ = feature_io.ReadFromFile(TEST_FET+'/'+file.split('.')[0]+'.delf')
        np.save(TEST_FET+'/'+file.split('.')[0]+'.npy', desc)
		
# Next, loaded the features of all images into a spark rdd. This takes a long time

import os
import sys

import findspark
findspark.init()

from pyspark import SparkContext, SparkConf

conf = SparkConf().setAppName("App")
conf = (conf.setMaster('local[*]')
        .set('spark.executor.memory', '1G')
        .set('spark.driver.memory', '12G'))
sc = SparkContext(conf=conf)

from pyspark.sql import SparkSession

spark = SparkSession \
    .builder \
    .appName("Python Spark SQL basic example") \
    .config("spark.some.config.option", "some-value") \
    .getOrCreate()
	
import os
import numpy as np
import pandas as pd
import time
n = 0
k = 0
l = os.listdir(TEST_FET)
start = time.clock()
data = None
pan = None
for file in l:
    if file.endswith('.npy'):
        desc = np.load(TEST_FET+'/'+file)
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
            if not os.path.isfile('test_fet.csv'):
                pan.to_csv('test_fet.csv', mode='w', index=False, header=True)
            else:
                pan.to_csv('test_fet.csv', mode='a', index=False, header=False)
            del pan
            pan = None
        k += 1
        n += desc.shape[0]

if k%1000!=0:
    if not os.path.isfile('test_fet.csv'):
        pan.to_csv('test_fet.csv', mode='w', index=False, header=True)
    else:
        pan.to_csv('test_fet.csv', mode='a', index=False, header=False)
    del pan
    pan = None
print(time.clock()-start)
			
# I converted csv to a parquet file to save space and time

def lis(x):
    return [float(i) for i in x[1:-1].split(',')]

from pyspark.ml.linalg import DenseVector

spark.read.load("test_fet.csv", format="csv", inferSchema="true", header="true").rdd \
          .map(lambda x: (x[2], x[1], DenseVector(lis(x[0])))) \
          .toDF(["index", "file", "features"])
          .write.parquet("test_fet.parquet")

# Now I get the Bag of Visual Words representation using K-means model built on training data
from pyspark import StorageLevel
schema = spark.read.parquet("test_fet.parquet").persist(StorageLevel(True, True, False, False, 1))

import numpy as np
from pyspark.ml.clustering import KMeansModel

model = KMeansModel.load('KmeansModel')
P = np.load('P.npy')

from pyspark.ml.linalg import DenseVector
predictions = model.transform(schema)
df = predictions.rdd \
    .map(lambda x: (x[2], x[0], DenseVector(np.matmul(np.array(x[0]), P.T)), x[1], x[3])) \
    .toDF(["Index", "Features", "Projections", "File", "VisualWords"])
	
# Then, I generate binary signatures for all test images
tau = np.load('tau.npy')

from pyspark.ml.linalg import DenseVector
def binsig(z, c, tau):
    return DenseVector((z > tau[c,:]))
	
df = df.rdd.map(lambda x: (x[0], x[3], x[4], binsig(x[2], x[4], tau),)) \
    .toDF(["Index", "File", "VisualWords", "BinarySignature"])
	
df.write.parquet("query_fet.parquet")

# Next, get hamming distances for all test image features

import pandas as pd
ht = 24
sigma = db/4
def score(b, c):
    d = pd.read_pickle("bintable/train%d.pkl" % c)
    d['Score'] = d['BinarySignature'].apply(lambda x: np.sum(np.logical_xor(x, b)))
    d2 = d[d['Score'] < ht][['File', 'Score']]
    d2['Score'] = d2['Score'].apply(lambda x: np.exp(-x**2/sigma**2))
    return d2.groupby('File').sum()

# Reduce according to filename by adding valid hamming distances and sorting according to total score
	
def red(x, y):
    return pd.concat([x, y]).groupby('File').sum().sort_values(by=['Score'], ascending=False)

# Convert to submission file format
	
df.rdd.map(lambda x: (x[1], score(x[3], x[2]))) \
    .reduceByKey(lambda x,y: red(x, y)) \
    .map(lambda x: (x[0], ' '.join(list(x[1].index)[:100]))) \
    .toDF(['id', 'images']) \
    .toPandas().to_csv('submit2.csv', index=False)