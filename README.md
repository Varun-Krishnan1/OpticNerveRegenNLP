# OpticNerveRegenNLP
Using NLP methods to gain insights from optic nerve regeneration literature.

## Dataset 
Our corpus consisted of _ papers ranging from years _ - _ chosen by members of the optic nerve regeneration wet lab team. 

The research papers were in PDF format. We applied optical character recognition to extract the text. 

## Word Embeddings from Literature 
The first step was to generate high-quality word embeddings from the corpus. These word embeddings were later used for inputs to GraphSage (see below). Choices had to be made on how to pre-process the data prior to generating the word embeddings including using or not using stopwords, the use of stemming vs lemmatization, and the choice of the word embedding model (Gensim vs GloVe). Different combinations were compared in the file Determining Best Embedding Model.ipynb which **trained models on research papers from a single year** and evaluated their output. The word embeddings later used in the project were trained over the entire corpus. 

Empirically training on research papers from a given year and comparing the different models outputs of most_similar to various options of words we determined that lemmatization combined with the use of stopwords and the gensim word embedding model provided the best results.

An example of the comparison between models trained on the research papers from year 1907 and the 20 most similar words the model gave for the word 'eye':  
<img width="1345" alt="image" src="https://github.com/Varun-Krishnan1/OpticNerveRegenNLP/assets/19865419/516ca8db-389e-476b-aeb6-3402fed510b8">

### Stopwords 
1. We used the nltk package for a pre-defined list of stopwords and used the english, german, dutch, french, and spanish stopwords. 
2. We used additional stopwords found here: https://github.com/igorbrigadir/stopwords/blob/master/en/alir3z4.txt 
3. We added 4 stopwords specific to our research papers: {'pubmed', 'et', 'al', 'page'}
4. All words with length < 2 were filtered out 

### Stemming and Lemmatization 
It was determined that lemmatizating resulted in better embeddings than stemming. 
Stemming was tested using the PorterStemmer from nltk. 
Lemmatizing was performed using spacy. 

### Model Choice: Gensim vs GloVe 
The Gensim model uses contionuous bag of words to train word embeddings which is a neural-network approach. 
GloVe uses word co-occurences matrices rather than a neural network to train word embeddings 




 
