# this tensorflow part is completed by ghe10, yingyiz2 and gjin7

from tensorflow.contrib.factorization.python.ops.gmm import GMM
from tensorflow.contrib.factorization.python.ops.kmeans import KMeansClustering as KMeans
from tensorflow.contrib.framework.python.framework import checkpoint_utils
import numpy as np
from PIL import Image
import time
# first, we read the data
im = Image.open("input.png")
pix = im.load()
width = im.size[0]
height = im.size[1]
# generate training set
X = []

for i in range(width):
    for j in range(height):
        r, g, b, a = pix[i, j]
        X.append(np.array([r, g, b]))

X = np.asarray(X, dtype=np.float32)

# set cluster number 
m = 10
# build GMM model
model = GMM(num_clusters=m, model_dir=None, random_seed=0,
            params='wmc', initial_clusters='random', covariance_type='full',
            batch_size=width * height, steps=10, continue_training=False, config=None, verbose=1)
# train GMM model with X
model.fit(x=X, y=None, monitors=None, logdir=None, steps=10)
# predict the cluster id of each pixel
result = model.predict(X, None)
result = np.asarray(result)
result = result[0]
# get ws, aves and vars of each cluster
mu = model.clusters()
mu = np.asarray(mu)
var = model.covariances()
var = np.asarray(var)
print("mu:")
print(mu.shape)
print(mu)
print("var:")
print(var.shape)
print(var)
W = checkpoint_utils.load_variable(model.model_dir, 'Variable')
print(W)
# build the result image
im = Image.new("RGB", (width, height))
count = 0
for i in range(width):
    for j in range(height):
        max_mu = mu[result[count]]
        im.putpixel((i, j), (int(max_mu[0]), int(max_mu[1]), int(max_mu[2])))
        count += 1
im.show()

im.save("flower_m=3.png")
