# Adding Convolution Operations to an LSTM for WSD
Word Sense Disambiguation (WSD) is an important field of natural language processing. The goal of WSD is to determine
the sense of a target word in a sentence based on the context. This is required since some words have different meanings depending on
the context, for example the word 'bank'. Typically, supervised models are used for this task. Since WSD is in the natural language
processing domain network structures that can learn sereis information are used such as Recurrent Neural Network (RNN) and Long Short
Term Memory (LSTM). This project was modeled after an approach that utilizes a Bidirectional LSTM (BiLSTM) [1]. The code is a modified
version of the code from Jeff09 [2]. The code needed some improvement as it did not work 'out of the box' in addition to implementing
a testing structure and making the test set generation work, different model approaches were tested.

## Data Set overview
The datasets used are Senseval 2 and Senseval 3. These datasets consist of context texts that contain the target word. In addition each
target word as an associated sense annotation id which gives the intended meaning of the target word.

## Vectorization
When working with text as data, which is the case for WSD, then the text needs to be transferred into number form. 
This process is referred to as vectorization. There are vectorization approaches that produce sparse data vectors such as 
Term Frequency - Inverse Document Frequency (TF-IDF) where each word has a single value. The other approaches produce dense data such 
as Word2Vec or GloVe. These approaches are referred to as embedding, and they produce a vector per word instead of just a single value. 
Both GloVe and Word2Vec preserve the correlation between different words in the corpus, and they can be used to determine words based 
on other vectors. GloVe tries to be more transparent compared to Word2Vec and achieve higher accuracy by only relying on word 
occurrence statistics in the corpus. GloVe was used for this project.

### The different models
The basic model utilizes only a BiLSTM followed by 2 dense layers. The sencond dense layer is the output layer which outputs the embedded
vector for the sense annotation id.</br>
The first model that was implemened for this project used convolutional layers on the input data before feeding it to the BiLSTM.</br>
The second model implemented used convolution layers after the LSTM as well as on the input before the BiLSMT.

## Test Results
The results can be view in the report pdf.

## Trained Model
A trained model can be retrieved from https://drive.google.com/open?id=1LqHUuSIEMGjTPEF2ozGGIGXCb196pNii as well as other needed supplementary files such as the Senseval datasets and the GloVe pretrained embedder. 

## Authors
* **Bleau Moores**
* **Lisa Ewen**
* **Tim Heydrich**

See also the list of [contributors](https://github.com/Legerdemainist/NLPAssignment2/graphs/contributors) who participated in this project.


## References
[1] Mikael K ̊ageb ̈ack and Hans Salomonsson. “Word SenseDisambiguation  using  a  Bidirectional  LSTM”.  In:Pro-ceedings of the 5th Workshop
    on Cognitive Aspects ofthe Lexicon (CogALex - V). Osaka, Japan: The COLING2016 Organizing Committee, Dec. 2016, pp. 51–56.
    URL:https://www.aclweb.org/anthology/W16-5307</br>
[2]  Jeff09 (Kun Li).Word Sense Disambiguation using Bidi-rectional LSTM. 
      https://github.com/Jeff09/Word-Sense-Disambiguation-using-Bidirectional-LSTM. 2018

