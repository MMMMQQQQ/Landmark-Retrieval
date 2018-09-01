# Landmark-Retrieval

The Landmark Retrieval problem is defined as - given a query image, retrieve all images in the dataset containing the same landmark as the query image. This project implements Bag of Visual Words with Hamming Embedding using DELF local image features. I use the Google Landmarks dataset to test performance. It contains 1,000,000 training images and 100,000 test images.

Local image features have advantages over global image features. In general, global image features are hindered by clutter, occlusion, and variations in viewpoint and illumination. Global descriptors also lack the ability to find patchlevel matches between images. Local image features solve these problems by extracting features from parts of the image. DELF features introduced in ["Large-Scale Image Retrieval with Attentive Deep Local Features", Hyeonwoo Noh, Andre Araujo, Jack Sim, Tobias Weyand, Bohyung Han, Proc. ICCV'17] use convnets to produce high quality local features for feature matching. DELF extracts ~100 features from each image for a dataset size of 100,000,000 features.

Feature matching is done by testing the match between each feature of the query image and each feature of every dataset image and finding the images that contain the best matches. This is a very computationally intensive task. So, I narrowed the search space by quantizing the features using K-means clustering with K=10,000. The query image features compared with the quantized vectors (visual words) to find the closest match and then they are compared to training set features that are represented by the same visual word. This is the Bag of Visual Words technique.

However, if K is too small the representation is too coarse and the clusters will be too large and contain features that do not represent the same object. If K is too large, then features that represent the same object will be represented by different words and it will be more computationally intensive to search. To mitigate this problem, I used Hamming Embedding [Jegou H., Douze M., Schmid C. (2008) Hamming Embedding and Weak Geometric Consistency for Large Scale Image Search. In: Forsyth D., Torr P., Zisserman A. (eds) Computer Vision – ECCV 2008. ECCV 2008. Lecture Notes in Computer Science, vol 5302. Springer, Berlin, Heidelberg]. By embedding local features belonging to the same visual word in the Hamming space, they are now represented by a vector of binary numbers. This simplifies distance calculation using the L1-norm, i.e., the hamming distance. Features that are within ht=24 hamming distance of the query features can be thought of as noisy versions of the query features and the files that contain them are returned. This has the advantage of reducing false matches even when K is too small.

This repository uses delf features from https://github.com/tensorflow/models/tree/master/research/delf. It also uses Apache Spark, Tensorlfow and Pandas. To generate a submission file for kaggle landmark retrieval challenge, run train.py on the training data (index data) and then run test.py on the test data (query data).
