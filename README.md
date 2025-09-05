# VIT for MNIST
This repository goal is to experiment with Visual Transformer.
It contains the following 
1. Simple CNN model for MNIST classification
2. Self attention: Given a sequence of MNIST images, predicting their sum (Image -> encoder -> attention -> regression prediction)
3. Cross attention: Given an anchor image and sequence of images, predicting if the anchor exists in the sequence  

# TODOs
Self attention:
- [x] Visualizations of the sequence, ground truth and prediction
- [x] Add loss vs epoch figure
- [x] Add accuracy metric (if round(output)=GT)
- [x] Sequence of varying length (1-10)
- [ ] Add output transformation for output to be between -1 and 1: (y_hat + 0.5) * y_max
- [ ] Compare to traditional model (CNN + fc)
- [ ] Histogram of error residual 

Cross attention:
- [ ] Baseline
- [ ] Sequence of varying length 
- [ ] Compare to traditional model (CNN + fc)


# Results

Self attention: 
 - Baseline: Validation [Loss Acc]=[1.40 0.53] Training [Loss Acc]=[0.48 0.57]
 - With output normalization scaling: 
 - CNN + FC:
