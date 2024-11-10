# Prerequisites
Download GloVe embeddings: https://nlp.stanford.edu/projects/glove/

Required NLTk packages:
```
import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
```

Part 1: Generate Embedding Matrix
Run `Part1.ipynb` in the following order to generate the embedding matrix and vocabulary mapping:

1. Load Rotten Tomatoes Dataset
Uses the `datasets` library to load movie reviews for sentiment analysis:
from datasets import load_dataset
dataset = load_dataset('rotten_tomatoes')

2. Build Vocabulary
Processes training data to create vocabulary through:
- Case folding (lowercase)
- NLTK tokenization
- Special character handling
- Hyphenated word splitting
Output: Size of Vocab: ```16535```

3. Load GloVe Embeddings
4. Create Embedding Matrix
Maps each vocabulary word to its GloVe embedding vector, identifying words missing from GloVe (OOV words).
Output: Size of OOV words: 604

Output of mapping:
```
{'determination': 0,
 'simple': 1,
 'taking': 2,...}
```
Output of embedding matrix:
```
array([[ 0.25093  ,  0.83451  ,  0.25677  , ...,  0.31425  , -0.24449  ,
        -0.0023992],
       [ 0.57959  ,  0.14576  ,  0.32607  , ...,  0.050995 , -0.24176  ,
        -0.1596   ],
       [-0.049447 ,  0.14972  , -0.2371   , ..., -0.11361  ,  0.048788 ,
        -0.19525  ],
       ...,]])
```

5. Handle OOV Words
Implements multiple strategies to handle Out-of-Vocabulary words:
   1. Stemming: Uses Lancaster Stemmer to match word variants
      1. Output: ```OOV word: abandone, substitute word: abandonar```
   1. WordNet Synonyms: Finds semantically similar words using WordNet based on POS from training set
   - Utilizes WordNet's synonym database
      1. Output: ```OOV word: juiceless, synonym: dry```
   1. Edit Distance: Catches misspellings using Levenshtein distance
      1. Output: ```OOV word: bizzarre, substitute word: bizarre, Distance: 1```
   1. Subword Embeddings: Splits and averages embeddings for compound words
      1. Output: ```OOV word: cipherlike, subwords: ('cipher', 'like')```
   1. Unknown Token : Assigns random embeddings to remaining OOV words
6. Save Embedding Matrix and Vocabulary Mapping

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

Run ```part3.ipynb``` for part 3.1, 3.2 and 3.4.
we implemented a workflow function to control the embedding methods, datasets and models used in each training along with a dictionary variable called params to pass the respective hyper parameters. To change the model used, can take a look into the code definition and choose the corresponding string as the input for model_type parameter. To use the OOV handling solution discussed in part 1, pass True to handle_oov parameter. Pass False to disable it. The output should indicate the average training loss and validation loss for each epoch, followed by an accuracy obtained from the test set.

Run Part 3.3/Part 3.3 - original dataset.ipynb to train / evaluate biGRU and biLSTM models on original dataset. Run Part 3.3/Part 3.3 - augmented dataset.ipynb to train / evaluate biGRU and biLSTM models on original dataset. Part 3.3 contains subdirectories with the presaved best checkpoints from training, which can be used for evaluation to obtain test accuracy without training the models again. Simply open the relevant notebook and click 'Run All' as the training code has been commented. Please uncomment the training code if you wish to train the models again.

1) Test Accuracy with updated word embeddings: 73% (part3.ipynb)
2) test Accuracy when dealing with OOV words: 73% (part3.ipynb)
3) Test Accuracy of 75.98% for biGRU, and 77.39% for biLSTM on original dataset. (Part 3.3/Part 3.3 - original dataset.ipynb)
4) Test Accuracy of 77.86% for biGRU, and 76.92% for biLSTM on augmented dataset. (Part 3.3/Part 3.3 - augmented dataset.ipynb)
5) Test Accuracy of CNN of 72% (part3.ipynb)
6) Final Enhancement with data augmentation with fine tuning on pretrained model with distil_roberta, achieving an accuracy of 89% (dillroberta.ipynb)

- Using backtranslation, augment the training dataset translating the training samples to an Engish-like language like French to avoid losing information and translating back into English.
- Using a word-level edit distance threshold of 5 to ensure that samples have substantial changes in the backtranslation process, and to avoid duplicating the training samples.
Sample output:
```
Original: the rock is destined to be the 21st century s new conan and that he s going to make a splash even greater than arnold schwarzenegger jean claud van damme or steven segal
Back Translated: the rock is destined to be the new conan of the 21st century and that he will cause an even more sensation than arnold schwarzenegger jean claude van damme or steven segal
```

- Combining the augmented dataset with the original dataset, we fine-tune the distil_roberta model to achieve an accuracy of 89% on the test set.
