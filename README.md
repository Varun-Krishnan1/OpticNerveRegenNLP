# OpticNerveRegenNLP
Using NLP methods to gain insights from optic nerve regeneration literature.

NOTE: Some files are not uploaded due to file size limits of github. The files not included are: 
* Balanced_Supervised_2_classes_no_molecules_10_epochs.pth - BERT model trained on balanced dataset 
* Supervised_10_epochs.pth
* Supervised_3_classes_no_molecules_10_epochs.pth - BERT model trained for 3 classes on unbalanced dataset 
* Supervised_3_classes_random_neither_10_epochs.pth - BERT model trained for 3 classes on unbalanced dataset without removal of - molecules 
* FastText.wordvectors.vectors_ngrams.npy 
* glove.6b.50d.txt
* glove.6b.100d.txt
* glove.6b.200d.txt
* glove.6b.300d.txt


## Previous Work 
This work is an expansion of the lab's previous paper found here: _   
The lab's previous paper used Dynamic Topic Modeling with Latent Dimension Analysis to cluster research papers from optic nerve regeneration literature. 

![LDAPaper](https://github.com/Varun-Krishnan1/OpticNerveRegenNLP/assets/19865419/9ff2cc79-3589-4246-978b-6a493be1f96c)

## Dataset 
Our corpus consisted of _ papers ranging from years _ - _ chosen by members of the optic nerve regeneration wet lab team. 

The research papers were in PDF format. We applied optical character recognition (OCR) to extract the text. 

## Known Promoter and Inhibitors 
To generate supervised learning sentences as well as to test accuracy of the various methods below we needed to have a set of known molecules that promoted optic nerve regeneration as well as molecules that inhibited optic nerve regeneration. These we will refer throughout this as **known promoters** and **known inhibitors** or simply **known molecules**. This was manually created by a domain expert in optic nerve regeneration and can be found at KnownPromotersInhibitors. The original list consisted of 23 promoters and 20 inhibitors but 5 promoters were added later on for better class balancing since inhibitors had a lot more sentences in the literature than promoters. 

## Novel Molecules 
One of the main goals of this work is to gain insight into new molecules that can be used for promoting optic nerve regeneration and inhibiting optic nerve regeneration. These will will refer throughout this as **novel promoters** or **novel inhibitor** or simply **novel molecules**. 

Before classifying molecules however we need a way to extract molecules from the literature. SciSpacy models for Named Entity Recognition (https://scispacy.apps.allenai.org/) was attempted to be used and all 4 named entity recognition models (craft_md, jnlpba_md, bc5cdr_md, bionlp13cg_md) were tried. The results can be found at __ but were determined to be too noisy and not enough molecules were able to be extracted. 

We came up with another method of extracting molecules from the literature using the scispacy abbreviation detector (https://github.com/allenai/scispacy). This was found to be a much more robust method generating much higher-quality candidate texts and also provided the full name for a molecule abbrevation. Having the full name and the abbrevation allowed us to use a domain expert to easily curate the generated list of text from the abbreviation detector to only include molecules. The fully curated list can be found at _ and consists of 822 novel molecules. 

## Determining Model for Word Embeddings from Literature 
The first step was to generate high-quality word embeddings from the corpus. These word embeddings were later used for inputs to GraphSage. Choices had to be made on how to pre-process the data prior to generating the word embeddings including using or not using stopwords, the use of stemming vs lemmatization, and the choice of the word embedding model (Gensim vs GloVe). Different combinations were compared in the file Determining Best Embedding Model.ipynb which **trained models on research papers from a single year** and evaluated their output. The word embeddings later used in the project were trained over the entire corpus. 

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
![image](https://github.com/Varun-Krishnan1/OpticNerveRegenNLP/assets/19865419/6a61f0b4-53b7-4fb9-b52c-a44dae4f1dad)

We also  plotted the word embeddings using PCA to determine if any meaningful clustering could be seen:    
![image](https://github.com/Varun-Krishnan1/OpticNerveRegenNLP/assets/19865419/45d412b3-2c15-41de-ac1c-aabdfd69db8a)

We compared this word embedding model trained on all research papers to the gensim model just trained on the latest years of research papers (i.e the last time period binning see above) to see if the noise from earlier papers were resulting in worse representations of words. To evaluate which model was better we used the similarity scores between each known promtoer and inhibitor to the words 'promote' and 'inhibit'. If the 'promote' similarity score was greater than 'inhibit' we classified that molecule as a promoter and vice versa. We then compared these to the true labels of the molecule. Using the gensim model trained over the entire dataset we got an accuracy of 62%. With the gensim model trained just over the most recent years we got anaccuracy of 65%. 


## GraphSAGE 
One of the main goals of this project was learning the relationships between different molecules. To accomplish this we used the GraphSAGE algorithm as described here: https://cs.stanford.edu/people/jure/pubs/graphsage-nips17.pdf. This algorithm allows for inductive learning of vector representations of entities that are linked together in a graph. The alogirthm then allows for using clustering these learned representations using a 2D TSNE plot. Code for analysis is at "GraphSAGE on all molecules.ipynb" 

### Creating the Graph 
To construct the graph we first started with our known promoters and inhibitors. The values of the nodes of the graph were the previously learned Gensim Word Embeddings for the molecules. Links between molecules were if the molecule occurred in the same sentence as another. The relationships empirically seen on the graph were found to match known pathways of how the known promoters and inhibitors relate to each other validating this approach. 

Promoters are in green and inhibitors are in red. Relationships between molecules such as neigbors a molecule has matches known pathways for these promoters and inhibitors. 
![image](https://github.com/Varun-Krishnan1/OpticNerveRegenNLP/assets/19865419/12a6ff5a-6348-4be4-a44e-c074224d8423)

A similar method was performed to graph all novel molecules and the known promoters or inhibitors in one graph (too big to be shown). However, because Gensim word embeddings were not available for all molecules, due to some molecules being too sparse in the corpus for an embedding to be learned, a new embedding approach had to be used. 

### Byte-Pair Encoding 
Because Gensim word embeddings were not available for all molecules, due to some molecules being too sparse in the corpus for an embedding to be learned, not all novel molecules could be included in the graph for GraphSAGE. To overcome this, we used Byte Pair Encoding to generate embedding for *all* novel molecules. Code found at GensimModeloverEntireCorpusGeneration.ipynb  

### Clustering using the GraphSAGE algorithm
GraphSAGE clustering using just the known promoters/inhibitors graph did not have enough molecules to see any reasonable clustering. Hence, the algorithm was doing using the graph that consisted of all novel and known molecules. The GraphSAGE algorithm was performed to generate GraphSAGE-learned word embeddings and these word embeddings were then graphed using TSNE onto a 2D plot. Known promoters and inhibitors are labeled in the graph. 

![image](https://github.com/Varun-Krishnan1/OpticNerveRegenNLP/assets/19865419/721a1998-3ede-4395-82a7-ea2cd3ffcc65)

Known promoters and inhibitors were seen to cluster in known optic nerve regeneration pathways validating this approach. 

## Classification 
It was important to not only determine the relationships between novel molecules in our project but also to be able to classify these molecules as promoters, inhibitors, or neither. 

### Using Linguistic Causation 
We were interested in using the natural language subfield of linguistic causation in helping us determine classifying our novel molecules as promoters or inhibitors. Specifically we wanted to see if these molecules *caused* promotion or inhibition of optic nerve regeneration. 

Our first step was building a model that could successfully characterize sentences into causal sentences or not causal sentences. We used the SemEval 2010 Task 8 dataset which consists of sentences labeled as causal or not suggested by Yang et al (https://arxiv.org/abs/2101.06426). We used this dataset to train a decision tree model inspired from Girju et al (https://dl.acm.org/doi/10.3115/1119312.1119322) which consisted of transforming sentences into <noun phrase, verb, noun phrase 2> for training. However, we departured from the paper by turning our noun phrases into word embeddings rather than semantic features. We used the pre-trained 100-dimesion GloVe model (glove.6B.100d) for transforming the noun phrases into word embeddings. However, this resulted in some noun phrases not being found. This could later be improved by using byte pair encoding. Our decision tree resulted in a weighted avg F1 score of 0.92 on the SemEval 2010 Task 8 dataset. Full analysis can be found at Decision Tree on SemEval Dataset.ipynb 

Classification report of our decision tree model (using sklearn.tree) 
![image](https://github.com/Varun-Krishnan1/OpticNerveRegenNLP/assets/19865419/ad59040b-ef9c-4aff-938c-5b66337cc2fe) 

### Using Causal Verbs 
Asghar showed in his linguistic causation survey (https://arxiv.org/abs/1605.07895) the use of causal verbs in determining causal sentences. Causal verbs can be categorized in terms of frequency and ambiguity with high frequency, low ambiguity verbs being the easiest to differentiate causal sentences vs non-causal sentences by. We used the verbs described in the paper combined with our known promoters and inhibitors to extract sentences with a known molecule and one of the causal verbs. However, not enough percentage of sentences for the molecules had causaul verbs in them for this approach to be useful and hence, we did not further pursue it. 

Percentage of sentences which mentioned a known molecule that had a causal verb:   
![image](https://github.com/Varun-Krishnan1/OpticNerveRegenNLP/assets/19865419/cb244f83-616c-4d61-b1d0-a99df289e6ff)


### Creating a Supervised Learning Dataset 
To use machine learning models to predict whether the novel molecules were promoters or inhibitors we needed to have a training dataset. Our training set was constructed using the known promoters and inhibitors described above. The corpus was iterated through and each sentence that containing a *known* promoter or inhibitor was extracted. This allows for sentences that could be presumed to be about promotion (sentences with a promoter) or sentences that could be presumed to be about inhibition (sentence with an inhibitor) of optic nerve regeneration. 

To ensure that there was sentences couldn't be about promotion and inhibition meaning it contained both a promoter and an inhibitor about it we plotted the number of promoters and inhibitors in each sentence we extracted. 
![image](https://github.com/Varun-Krishnan1/OpticNerveRegenNLP/assets/19865419/9eac9a54-2c00-4a91-81a8-dfcae75c0591)

You can notice from the figure that the vast majority of sentences have only one promoter or one inhibitor. In fact this consists of 97% of the sentences in the corpus. The other 3% that contained both were discarded from the supervised learning dataset. 

The unbalanced version of the dataset consisted of 1279 promoter sentences and 2631 inhibitor sentences and 5000 *randomly* chosen neutral sentences.   
The balanced version of the dataset consisted of 2625 promoter sentences and 2603 inhibitor sentences. We did not include neutral sentences for simplicity. 

### Logistic Regression and Naive-Bayes 
**Logistic Regression** was performed by first lemmatizing the corpus and removing all stop words. Then a frequency table of the supervised learning dataset sentences was constructed and this frequency table was used to construct vector representation of each of our sentences. We plotted the vector representation of our sentences with green being promotion sentences and red being inhibition sentences. You can clearly see from the figure that they can be separated in the 2D vector space and hence, it is feasible a logistic regression model can learn to distinguish between the sentences. The figure and the results from the logistic regression model are below. Note the results shown are with the unbalanced dataset which explains why it does much better labeling sentences ans inhibition rather than promotion. 

![image](https://github.com/Varun-Krishnan1/OpticNerveRegenNLP/assets/19865419/ddee0e14-9b99-4709-9e07-d3adee975f1a)
![image](https://github.com/Varun-Krishnan1/OpticNerveRegenNLP/assets/19865419/3875b9b3-1fa2-400f-9739-93fb39b9cbd8)

**Naive-Bayes** was performed using smoothed ratio values from the frequency table to represent each sentence. The vector representations were also plotted which showed the sentences could be split. The results running Naive-Bayes with the balanced dataset are below. 
![image](https://github.com/Varun-Krishnan1/OpticNerveRegenNLP/assets/19865419/030690d6-b89b-4570-b7e2-7d9331947917)

**It was found from analyzing the frequency tables of both models that words we would expect to be inhibitory (eg, inhibition, deletion, myelin) were very likely to be seen in known inhibitory sentences whereas words that we would expect to be promotery (eg, promote, increase, lengthen) were not likely to be words occurring with high frequency in known promotion sentences. This is important as models need to distinguish inhibitory words vs promotion words but it doesn't look like sentences in the dataset do a good job of allowing for this. This may be why BERT (see below) was not able to differenitate promotion vs inhibition sentences either.** 

### BERT 
The state-of-the-art NLP model of BERT (Bidrectional Encoder Representations from Transformers) was used to better classify the novel molecules. We used the "roberta-base" model from the hugging face transformers library. 

To validate BERT's ability to differentiate causual sentences we used the SemEval 2010 Task 8 Dataset (described above) and achieved a .97 AUC on the dataset. 

We fine-tuned the roberta-base model on our supervised dataset (workflow described below). Note that the supervised molecules were removed from the sentence prior to training so that BERT could not over-train on the supervised molecule's names. We originally trained BERT on the unbalanced version of the dataset and validated it by its ability to classify promoters, inhibitors, and neither from a set of manually labeled molecules, different from the ones it were trained on, but it achieved an accuracy of only 34% (essentially random guessing). It's most common mistake was labelling a promoter as an inhibitor - most likely due to the unbalanced nature of the dataset. Code is at BERT Training for 3 Classes.ipynb and Evaluating BERT 3 Classes on Manually Labelled Data.ipynb. 

We tried again using the balanced version of the dataset and trained it to classify novel molecules as only 2 classes, promoter or inhibitor, yet when validated with the manually labelled data it achieved an accuracy of only 48% again essentially random. Empirically giving BERT sentences that should be easy to label as promotory or inhibitory, BERT did not show an ability to acurrately classify the sentences. Code is at BERT 2 Class Training on Balanced Dataset.ipynb. 

Bert Workflow:
1. Go through corpus and use known molecules to get sentences with a promoter and sentences with an inhibitor 
2. Presumably, these sentences have unique characteristics for promoters vs inhibitors that BERT could learn to recognize 
3. Remove the known molecules in these sentences so BERT will not focus on molecule names 
4. Train the BERT model with input of these promoter sentences, inhibitor sentences, and also sentences with neither and output as the class of molecule that sentence is representing (or just promoter and inhibitor sentences for 2 class version) 
5. After training, go through the corupus and pull sentences with the novel molecules and using the sentences have our trained BERT model label those sentences as promoter or inhibitor or neither (or promoter or inhibitor for 2 class version) 
6. Based on their respective sentence labels classify unknown molecules as promoter or inhibitor or neither by taking the average score across all the sentences of a given molecule 

Embeddings generated from the BERT model were also used as node values for GraphSAGE (see above) rather than Gensim Word Vectors. However, the results with the BERT embeddings were found to be worse due to the face that BERT embeddigns were not meant to be compared (such as us using cosine similarity) like traditional word embedding are. 








 
