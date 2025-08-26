# VIT for MNIST
This repository goal is to experiment with Visual Transformer.
It contains the following 
1. Simple CNN model for MNIST classification
2. Self attention: Given a sequence of MNIST images, predicting their sum (Image -> encoder -> attention -> regression prediction)
3. Cross attention: Given an anchor image and sequence of images, predicting if the anchor exists in the sequence  