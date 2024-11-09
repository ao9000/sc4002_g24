# sc4002_g24


Download GloVe embeddings: https://nlp.stanford.edu/projects/glove/
<br>
Part 1: <br>
Run Part1.ipynb <br>
1) Size of Vocab: 16535
2) Size of OOV words: 604
3) Multi-pass strategy: Stemming, Synonym Substitution, Edit distance matching, Subword Decomposition, <UNK> Token

Part 2: <br>
Run: 
1) Part 2/simple_rnn (tensorflow).ipynb
2) Part 2/simple_rnn_(2a2b).ipynb
3) Part 2/simple_rnn_(2c).ipynb

RNN Model Training and Evaluation <br>
Accuracy: 75.8% <br>
Different Strategies and respective test accuracy to deriving the final sentence representation to perform sentiment classification:
1) Max Pooling: 74.86%
2) Mean Polling: 74.11%
3) Concantenation Pooling: 75.05%
4) Attention Layer: 74.95%


Part 3:

For part 3.1, 3.2 and 3.4, we implemented a workflow function to control the embedding methods, datasets and models used in each training along with a dictionary variable called params to pass the respective hyper parameters. To change the model used, can take a look into the code definition and choose the corresponding string as the input for model_type parameter. To use the OOV handling solution discussed in part 1, pass True to handle_oov parameter. Pass False to disable it. The output should indicate the average training loss and validation loss for each epoch, followed by an accuracy obtained from the test set.

1) Test Accuracy with updated word embeddings: 73% (part3.ipynb)
2) test Accuracy when dealing with OOV words: 73% (part3.ipynb)
3) Test Accuracy of 75.98% for biGRU, and 77.39% for biLSTM. (Part 3.3/Part 3.3 - augmented dataset.ipynb)
4) Test Accuracy of CNN of 72% (part3.ipynb)
5) Final Enhancement with data augmentation with fine tuning on pretrained model with distil_roberta, achieving an accuracy of 89% (dillroberta.ipynb)
