# OpticNerveRegenNLP
Using NLP methods to gain insights from optic nerve regeneration literature.

## Previous Work 
This work is an expansion of the lab's previous paper found here: _ 
Our previous paper used Dynamic Topic Modeling with Latent Dimension Analysis to cluster research papers from optic nerve regeneration literature. 

![LDAPaper](https://github.com/Varun-Krishnan1/OpticNerveRegenNLP/assets/19865419/9ff2cc79-3589-4246-978b-6a493be1f96c)

## Dataset 
Our corpus consisted of _ papers ranging from years _ - _ chosen by members of the optic nerve regeneration wet lab team. 

The research papers were in PDF format. We applied optical character recognition to extract the text. 

## Known Promoter and Inhibitors 
To generate supervised learning sentences as well as to test accuracy of the various methods below we needed to have a set of known molecules that promoted optic nerve regeneration as well as molecules that inhibited optic nerve regeneration. These we will refer throughout this as known promoters and known inhibitors. This was manually created by a domain expert in optic nerve regeneration and can be found at KnownPromotersInhibitors. The original list consisted of 23 promoters and 20 inhibitors but 5 promoters were added later on for better class balancing since inhibitors had a lot more sentences in the literature than promoters. 

## Novel Molecules 
One of the main goals of this work is to gain insight into new molecules that can be used for promoting optic nerve regeneration and inhibiting optic nerve regeneration. These will will refer throughout this as novel molecules or novel promoters or novel inhibitor. 

Before classifying molecules however we need a way to extract molecules from the literature. SciSpacy models for Named Entity Recognition (https://scispacy.apps.allenai.org/) was attempted to be used using all 4 named entity recognition models (craft_md, jnlpba_md, bc5cdr_md, bionlp13cg_md). The results can be found at __ but were determined to be too noisy and not enough molecules were able to be extracted. 

We came up with another method of extracting molecules from the literature using the scispacy abbreviation detector (https://github.com/allenai/scispacy). This was found to be a much more robust method generating much higher-quality candidate texts and also provided the full name for a molecule abbrevation. Having the full name and the abbrevation allowed us to use a domain expert to easily curate the generated list of text from the abbreviation detector to only include molecules. The fully curated list can be found at _ 

## Determining Model for Word Embeddings from Literature 
The first step was to generate high-quality word embeddings from the corpus. These word embeddings were later used for inputs to GraphSage (see below). Choices had to be made on how to pre-process the data prior to generating the word embeddings including using or not using stopwords, the use of stemming vs lemmatization, and the choice of the word embedding model (Gensim vs GloVe). Different combinations were compared in the file Determining Best Embedding Model.ipynb which **trained models on research papers from a single year** and evaluated their output. The word embeddings later used in the project were trained over the entire corpus. 

Empirically training on research papers from a given year and comparing the different models outputs of most_similar to various options of words we determined that lemmatization combined with *not* filtering out stopwords and the gensim word embedding model provided the best results.

An example of the comparison between models trained on the research papers from year 1907 and the 20 most similar words the model gave for the word 'eye':  
<img width="1345" alt="image" src="https://github.com/Varun-Krishnan1/OpticNerveRegenNLP/assets/19865419/516ca8db-389e-476b-aeb6-3402fed510b8">

### Stopwords 
NOTE: It was determined that not filtering out stopwords resulted in better embeddings. 
1. We used the nltk package for a pre-defined list of stopwords and used the english, german, dutch, french, and spanish stopwords. 
2. We used additional stopwords found here: https://github.com/igorbrigadir/stopwords/blob/master/en/alir3z4.txt 
3. We added 4 stopwords specific to our research papers: {'pubmed', 'et', 'al', 'page'}
4. All words with length < 2 were filtered out 

### Stemming and Lemmatization 
NOTE: It was determined that lemmatizating resulted in better embeddings than stemming. 
Stemming was tested using the PorterStemmer from nltk. 
Lemmatizing was performed using spacy. 

### Model Choice: Gensim vs GloVe 
NOTE: It was determined that the Gensim model resulted in better embeddings than GloVe. 
The Gensim model uses contionuous bag of words to train word embeddings which is a neural-network approach. 
GloVe uses word co-occurences matrices rather than a neural network to train word embeddings 

## Using Our Word Embedding Model 
For the word embedding models trained individually on each year we used the gensim word vector embedding model with lemmatization and *filtering* out stopwords 
For the word embedding model trained on all the research papers we used the gensim word vector embedding model with lemmatization and *not* filtering out stopwords 

### Single Year Analysis 
We tried to characterize trends across years in the litreature by training individual models for each year we had in our time span. For the single year analysis we used the gensim model with lemmatization but filtered out the stopwords. Full analysis can be found at SingleYearAnalysis.ipynb 

Examples of trends included how the similarity score between our known promoters and inhibitors and the words 'promote' and 'inhibit' changed across the different years. We also looked at how empirically chosen optic nerve regeneration words similarity scores between each others changed over years such as 'neuron' and 'lipid'. Examples shown below: 

Similarity between 'neuron synpase' and 'bipolar cells' across years: 
![image](https://github.com/Varun-Krishnan1/OpticNerveRegenNLP/assets/19865419/cf16c151-3ae4-48b2-a58d-5f91cf143ea8)

### Binned Year Analysis 
Our original paper found _ binned papers into time periods based on when they were written. We used our gensim model with lemmatization and *not* filtering out stopwords and trained individual models across each binned time period. We then tried to characterize trends across the years. Full analysis can be found at BinnedTimePeriodAnalysis.ipynb 

Similarity between 'neuron' and 'axon' across binned time periods: 
![image](https://github.com/Varun-Krishnan1/OpticNerveRegenNLP/assets/19865419/70007890-4f0b-495c-9289-cc129591238a)

### All Year Analysis 
We trained a gensim model with lemmatization and *not* filtering out stopwords across all research papers to get a better representation of words across all contexts. Full analysis can be found at GensimModelEntireCorpusAnalysis.ipynb

To determine how much the word embedding model captured we looked at similar words between our known promoters and inhibitors. Most similar words for the molecule 'socs3'. As you can see the words are meaningful meaning the model has a good vector representation of 'socs3': 
<img width="395" alt="image" src="https://github.com/Varun-Krishnan1/OpticNerveRegenNLP/assets/19865419/6a61f0b4-53b7-4fb9-b52c-a44dae4f1dad">

We also  plotted the word embeddings using PCA to determine if any meaningful clustering could be seen: 
![image](https://github.com/Varun-Krishnan1/OpticNerveRegenNLP/assets/19865419/45d412b3-2c15-41de-ac1c-aabdfd69db8a)

We compared this word embedding model trained on all research papers to the gensim model just trained on the latest years of research papers (i.e the last time period binning see above) to see if the noise from earlier papers were resulting in worse representations of words. To evaluate which model was better we used the similarity scores between each known promtoer and inhibitor to the words 'promote' and 'inhibit'. If the 'promote' similarity score was greater than 'inhibit' we classified that molecule as a promoter and vice versa. We then compared these to the true labels of the molecule. Using the gensim model trained over the entire dataset we got an accuracy of 62%. With the gensim model trained just over the most recent years we got anaccuracy of 65%. 

## Causal Verbs 


## GraphSAGE 


### Creating the Graph 

### Clustering using the GraphSAGE algorithm

## Classification 

### Extracting Novel Molecules

### Naive-Bayes and Logistic Regression 

### BERT 








 
