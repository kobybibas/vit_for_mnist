# VIT for MNIST
This repository goal is to experiment with Visual Transformer.
It contains the following 
1. Simple CNN model for MNIST classification
2. Self attention: Given a sequence of MNIST images, predicting their sum (Image -> encoder -> attention -> regression prediction)
3. Cross attention: Given an anchor image and sequence of images, predicting if the anchor exists in the sequence  

# TODOs
Self attention:
- [x] Visualizations of the sequence, ground truth and prediction
- [ ] Add loss vs epoch figure
- [ ] Add accuracy metric (if round(output)=GT)
- [ ] Sequence of varying length 
- [ ] Compare to traditional model (CNN + fc)

Cross attention:
- [ ] Baseline
- [ ] Sequence of varying length 
- [ ] Compare to traditional model (CNN + fc)