# OpticNerveRegenNLP

Using NLP methods to gain insights from optic nerve regeneration literature.

NOTE: Some files are not uploaded due to file size limits of github. The files not included are:

- Balanced_Supervised_2_classes_no_molecules_10_epochs.pth - BERT model trained on balanced dataset
- Supervised_10_epochs.pth
- Supervised_3_classes_no_molecules_10_epochs.pth - BERT model trained for 3 classes on unbalanced dataset
- Supervised_3_classes_random_neither_10_epochs.pth - BERT model trained for 3 classes on unbalanced dataset without removal of - molecules
- FastText.wordvectors.vectors_ngrams.npy
- glove.6b.50d.txt
- glove.6b.100d.txt
- glove.6b.200d.txt
- glove.6b.300d.txt

## Previous Work

This work is an expansion of the lab's previous paper found here: \_  
The lab's previous paper used Dynamic Topic Modeling with Latent Dimension Analysis to cluster research papers from optic nerve regeneration literature.

![LDAPaper](https://github.com/Varun-Krishnan1/OpticNerveRegenNLP/assets/19865419/9ff2cc79-3589-4246-978b-6a493be1f96c)

## Dataset

Our corpus consisted of _ papers ranging from years _ - \_ chosen by members of the optic nerve regeneration wet lab team.

The research papers were in PDF format. We applied optical character recognition (OCR) to extract the text.

## Known Promoter and Inhibitors

To generate supervised learning sentences as well as to test accuracy of the various methods below we needed to have a set of known molecules that promoted optic nerve regeneration as well as molecules that inhibited optic nerve regeneration. These we will refer throughout this as **known promoters** and **known inhibitors** or simply **known molecules**. This was manually created by a domain expert in optic nerve regeneration and can be found at KnownPromotersInhibitors. The original list consisted of 23 promoters and 20 inhibitors but 5 promoters were added later on for better class balancing since inhibitors had a lot more sentences in the literature than promoters.

## Novel Molecules

One of the main goals of this work is to gain insight into new molecules that can be used for promoting optic nerve regeneration and inhibiting optic nerve regeneration. These will will refer throughout this as **novel promoters** or **novel inhibitor** or simply **novel molecules**.

Before classifying molecules however we need a way to extract molecules from the literature. SciSpacy models for Named Entity Recognition (https://scispacy.apps.allenai.org/) was attempted to be used and all 4 named entity recognition models (craft_md, jnlpba_md, bc5cdr_md, bionlp13cg_md) were tried. The results can be found at \_\_ but were determined to be too noisy and not enough molecules were able to be extracted.

We came up with another method of extracting molecules from the literature using the scispacy abbreviation detector (https://github.com/allenai/scispacy). This was found to be a much more robust method generating much higher-quality candidate texts and also provided the full name for a molecule abbrevation. Having the full name and the abbrevation allowed us to use a domain expert to easily curate the generated list of text from the abbreviation detector to only include molecules. The fully curated list can be found at \_ and consists of 822 novel molecules.

## Determining Model for Word Embeddings from Literature

The first step was to generate high-quality word embeddings from the corpus. These word embeddings were later used for inputs to GraphSage. Choices had to be made on how to pre-process the data prior to generating the word embeddings including using or not using stopwords, the use of stemming vs lemmatization, and the choice of the word embedding model (Gensim vs GloVe). Different combinations were compared in the file Determining Best Embedding Model.ipynb which **trained models on research papers from a single year** and evaluated their output. The word embeddings later used in the project were trained over the entire corpus.

Empirically training on research papers from a given year and comparing the different models outputs of most_similar to various options of words we determined that lemmatization combined with _not_ filtering out stopwords and the gensim word embedding model provided the best results.

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

For the word embedding models trained individually on each year we used the gensim word vector embedding model with lemmatization and _filtering_ out stopwords
For the word embedding model trained on all the research papers we used the gensim word vector embedding model with lemmatization and _not_ filtering out stopwords

### Single Year Analysis

We tried to characterize trends across years in the litreature by training individual models for each year we had in our time span. For the single year analysis we used the gensim model with lemmatization but filtered out the stopwords. Full analysis can be found at SingleYearAnalysis.ipynb

Examples of trends included how the similarity score between our known promoters and inhibitors and the words 'promote' and 'inhibit' changed across the different years. We also looked at how empirically chosen optic nerve regeneration words similarity scores between each others changed over years such as 'neuron' and 'lipid'. Examples shown below:

Similarity between 'neuron synpase' and 'bipolar cells' across years:
![image](https://github.com/Varun-Krishnan1/OpticNerveRegenNLP/assets/19865419/cf16c151-3ae4-48b2-a58d-5f91cf143ea8)

### Binned Year Analysis

Our original paper found \_ binned papers into time periods based on when they were written. We used our gensim model with lemmatization and _not_ filtering out stopwords and trained individual models across each binned time period. We then tried to characterize trends across the years. Full analysis can be found at BinnedTimePeriodAnalysis.ipynb

Similarity between 'neuron' and 'axon' across binned time periods:
![image](https://github.com/Varun-Krishnan1/OpticNerveRegenNLP/assets/19865419/70007890-4f0b-495c-9289-cc129591238a)

### All Year Analysis

We trained a gensim model with lemmatization and _not_ filtering out stopwords across all research papers to get a better representation of words across all contexts. Full analysis can be found at GensimModelEntireCorpusAnalysis.ipynb

To determine how much the word embedding model captured we looked at similar words between our known promoters and inhibitors. Most similar words for the molecule 'socs3'. As you can see the words are meaningful meaning the model has a good vector representation of 'socs3':  
![image](https://github.com/Varun-Krishnan1/OpticNerveRegenNLP/assets/19865419/6a61f0b4-53b7-4fb9-b52c-a44dae4f1dad)

We also plotted the word embeddings using PCA to determine if any meaningful clustering could be seen:  
![image](https://github.com/Varun-Krishnan1/OpticNerveRegenNLP/assets/19865419/45d412b3-2c15-41de-ac1c-aabdfd69db8a)

We compared this word embedding model trained on all research papers to the gensim model just trained on the latest years of research papers (i.e the last time period binning see above) to see if the noise from earlier papers were resulting in worse representations of words. To evaluate which model was better we used the similarity scores between each known promtoer and inhibitor to the words 'promote' and 'inhibit'. If the 'promote' similarity score was greater than 'inhibit' we classified that molecule as a promoter and vice versa. We then compared these to the true labels of the molecule. Using the gensim model trained over the entire dataset we got an accuracy of 62%. With the gensim model trained just over the most recent years we got an accuracy of 65%.   

With results of 62% using all years and 65% using last binned years the GloVe approach is about 10-15% worse than BERT and GPT in predicting known moleculess and inhibitors

## GraphSAGE

One of the main goals of this project was learning the relationships between different molecules. To accomplish this we used the GraphSAGE algorithm as described here: https://cs.stanford.edu/people/jure/pubs/graphsage-nips17.pdf. This algorithm allows for inductive learning of vector representations of entities that are linked together in a graph. The alogirthm then allows for using clustering these learned representations using a 2D TSNE plot. Code for analysis is at "GraphSAGE on all molecules.ipynb"

### Creating the Graph

To construct the graph we first started with our known promoters and inhibitors. The values of the nodes of the graph were the previously learned Gensim Word Embeddings for the molecules. Links between molecules were if the molecule occurred in the same sentence as another. The relationships empirically seen on the graph were found to match known pathways of how the known promoters and inhibitors relate to each other validating this approach.

Promoters are in green and inhibitors are in red. Relationships between molecules such as neigbors a molecule has matches known pathways for these promoters and inhibitors.
![image](https://github.com/Varun-Krishnan1/OpticNerveRegenNLP/assets/19865419/12a6ff5a-6348-4be4-a44e-c074224d8423)

A similar method was performed to graph all novel molecules and the known promoters or inhibitors in one graph (too big to be shown). However, because Gensim word embeddings were not available for all molecules, due to some molecules being too sparse in the corpus for an embedding to be learned, a new embedding approach had to be used.

### Byte-Pair Encoding

Because Gensim word embeddings were not available for all molecules, due to some molecules being too sparse in the corpus for an embedding to be learned, not all novel molecules could be included in the graph for GraphSAGE. To overcome this, we used Byte Pair Encoding to generate embedding for _all_ novel molecules. Code found at GensimModeloverEntireCorpusGeneration.ipynb

### Clustering using the GraphSAGE algorithm

GraphSAGE clustering using just the known promoters/inhibitors graph did not have enough molecules to see any reasonable clustering. Hence, the algorithm was doing using the graph that consisted of all novel and known molecules. The GraphSAGE algorithm was performed to generate GraphSAGE-learned word embeddings and these word embeddings were then graphed using TSNE onto a 2D plot. Known promoters and inhibitors are labeled in the graph.

![image](https://github.com/Varun-Krishnan1/OpticNerveRegenNLP/assets/19865419/721a1998-3ede-4395-82a7-ea2cd3ffcc65)

Known promoters and inhibitors were seen to cluster in known optic nerve regeneration pathways validating this approach.

## Classification

It was important to not only determine the relationships between novel molecules in our project but also to be able to classify these molecules as promoters, inhibitors, or neither.

### Using Linguistic Causation

We were interested in using the natural language subfield of linguistic causation in helping us determine classifying our novel molecules as promoters or inhibitors. Specifically we wanted to see if these molecules _caused_ promotion or inhibition of optic nerve regeneration.

Our first step was building a model that could successfully characterize sentences into causal sentences or not causal sentences. We used the SemEval 2010 Task 8 dataset which consists of sentences labeled as causal or not suggested by Yang et al (https://arxiv.org/abs/2101.06426). We used this dataset to train a decision tree model inspired from Girju et al (https://dl.acm.org/doi/10.3115/1119312.1119322) which consisted of transforming sentences into <noun phrase, verb, noun phrase 2> for training. However, we departured from the paper by turning our noun phrases into word embeddings rather than semantic features. We used the pre-trained 100-dimesion GloVe model (glove.6B.100d) for transforming the noun phrases into word embeddings. However, this resulted in some noun phrases not being found. This could later be improved by using byte pair encoding. Our decision tree resulted in a weighted avg F1 score of 0.92 on the SemEval 2010 Task 8 dataset. Full analysis can be found at Decision Tree on SemEval Dataset.ipynb

Classification report of our decision tree model (using sklearn.tree)
![image](https://github.com/Varun-Krishnan1/OpticNerveRegenNLP/assets/19865419/ad59040b-ef9c-4aff-938c-5b66337cc2fe)

### Using Causal Verbs

Asghar showed in his linguistic causation survey (https://arxiv.org/abs/1605.07895) the use of causal verbs in determining causal sentences. Causal verbs can be categorized in terms of frequency and ambiguity with high frequency, low ambiguity verbs being the easiest to differentiate causal sentences vs non-causal sentences by. We used the verbs described in the paper combined with our known promoters and inhibitors to extract sentences with a known molecule and one of the causal verbs. However, not enough percentage of sentences for the molecules had causaul verbs in them for this approach to be useful and hence, we did not further pursue it.

Percentage of sentences which mentioned a known molecule that had a causal verb:  
![image](https://github.com/Varun-Krishnan1/OpticNerveRegenNLP/assets/19865419/cb244f83-616c-4d61-b1d0-a99df289e6ff)

### Creating a Supervised Learning Dataset

To use machine learning models to predict whether the novel molecules were promoters or inhibitors we needed to have a training dataset. Our training set was constructed using the known promoters and inhibitors described above. The corpus was iterated through and each sentence that containing a _known_ promoter or inhibitor was extracted. This allows for sentences that could be presumed to be about promotion (sentences with a promoter) or sentences that could be presumed to be about inhibition (sentence with an inhibitor) of optic nerve regeneration.

To ensure that there was sentences couldn't be about promotion and inhibition meaning it contained both a promoter and an inhibitor about it we plotted the number of promoters and inhibitors in each sentence we extracted.
![image](https://github.com/Varun-Krishnan1/OpticNerveRegenNLP/assets/19865419/9eac9a54-2c00-4a91-81a8-dfcae75c0591)

You can notice from the figure that the vast majority of sentences have only one promoter or one inhibitor. In fact this consists of 97% of the sentences in the corpus. The other 3% that contained both were discarded from the supervised learning dataset.

The unbalanced version of the dataset consisted of 1279 promoter sentences and 2631 inhibitor sentences and 5000 _randomly_ chosen neutral sentences.  
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

We fine-tuned the roberta-base model on our supervised dataset (workflow described below). Note that the supervised molecules were removed from the sentence prior to training so that BERT could not over-train on the supervised molecule's names. We originally trained BERT on the unbalanced version of the dataset and validated it by its ability to classify promoters, inhibitors, and neither from a set of manually labeled molecules, different from the ones it were trained on. These manually labeled molecuels were taken from the wet team and labels from the team were extracted by taking the first letter of their respones ("P" vs "I" vs "N"). About 141 molecules were able to evalauted but it achieved an accuracy of only 34% (essentially random guessing). It's most common mistake was labelling a promoter as an inhibitor - most likely due to the unbalanced nature of the dataset. Code is at BERT Training for 3 Classes.ipynb and Evaluating BERT 3 Classes on Manually Labelled Data.ipynb.

We tried again using the balanced version of the dataset and trained it to classify novel molecules as only 2 classes, promoter or inhibitor, yet when validated with the manually labelled data it achieved an accuracy on the wet lab labelled molecules of only 48% again essentially random. Empirically giving BERT sentences that should be easy to label as promotory or inhibitory, BERT did not show an ability to acurrately classify the sentences. Code is at BERT 2 Class Training on Balanced Dataset.ipynb.

Bert Workflow:

1. Go through corpus and use known molecules to get sentences with a promoter and sentences with an inhibitor
2. Presumably, these sentences have unique characteristics for promoters vs inhibitors that BERT could learn to recognize
3. Remove the known molecules in these sentences so BERT will not focus on molecule names
4. Train the BERT model with input of these promoter sentences, inhibitor sentences, and also sentences with neither and output as the class of molecule that sentence is representing (or just promoter and inhibitor sentences for 2 class version)
5. After training, go through the corupus and pull sentences with the novel molecules and using the sentences have our trained BERT model label those sentences as promoter or inhibitor or neither (or promoter or inhibitor for 2 class version)
6. Based on their respective sentence labels classify unknown molecules as promoter or inhibitor or neither by taking the average score across all the sentences of a given molecule

Embeddings generated from the BERT model were also used as node values for GraphSAGE (see above) rather than Gensim Word Vectors. However, the results with the BERT embeddings were found to be worse due to the face that BERT embeddigns were not meant to be compared (such as us using cosine similarity) like traditional word embedding are.

### GPT

GPT has the potential to provide even better results than BERT. To test the use of a GPT model we used the GPT interface available from OpenAi at chat.openai.com. We chose to use the web interface rather than making api calls due to our high token size and therefore, high cost associated with the api. To use GPT we did not do any fine-tuning on the model and instead did a zero-shot classificaiton approach and classified molecules into two classes, promoter or inhibitor. We did not add a third class due to a lack of a ground truth dataset with neutral molecules to evalute how well the model was predicting neutral molecules.

**Generating Masked Training Sentences**  
The first goal is to generate masked sentences for molecules. This is done in Generating Masked Training Sentences.ipynb.

First, this file uses KnownPromotersInhibitors.csv to read in the PrimaryName, OtherNames, and Class of our known molecules. It then defines a Molecule class which stores these names and class for each molecule. We then iterate through the sentences in our corpus. For each sentence we remove the punctuation and then check for any molecule names (primary or other names) for each molecule. If there is only one class of molecules in the sentence (so excluding sentences with both a promoter and inhibitor in it) then we mask all the primary and other names for each molecule of the same class we found. We do **not** mask any other molecule names in the sentence. Finally, we add this sentence to a dictionary with the key being the primaryname of the molecule (or molecules) that were in the sentence.

The output is saved in Sentence Classification/Output/Combined_Sentences_Per_Molecule/masked_known_molecules.json

**Creating GPT Prompts**
We then need to create prompts to provide GPT for classifying a molecule as promoter or inhibitor. Two types of prompts were tried: one-answer and justification. The prompts were prepended to the masked sentences for each molecule.  
One Answer Prompt:

> I will give you a set of sentences from research articles that have to do with optic nerve regeneration. These sentences will have a molecule, represented by "[MASK1]", which is either a promoter or inhibitor of optic nerve regeneration. Please respond only with if this molecule is a promoter or inhibitor of optic nerve regeneration. No explanation is necessary. You can only respond with the word promoter or inhibitor. Here are the sentences from the research articles:
> Justification Prompt:  
> I will give you a set of sentences from research articles that have to do with optic nerve regeneration. These sentences will have a molecule, represented by "[MASK1]", which is either a promoter or inhibitor of optic nerve regeneration. Please respond with if this molecule is a promoter or inhibitor of optic nerve regeneration. Please justify your answer. Here are the sentences from the research articles:

The original purpose of the one-answer prompt was to save some tokens on the response if we were to use an API call in the future. However, after testing the responses we found that the justification prompt was able to get more accurate classifications of the molecules. Therefore, we used the justification prompt moving forward. Ocassionaly, GPT would not give an answer saying there was not enough information or it was too ambigious. In these cases we would respond with "Please pick an option" which forced it to choose a class.

**Token Size**  
The GPT model we first began with was GPT 3.5 which has a max token size of 4906 tokens. Therefore, we had to ensure the prompts plus all masked sentences fit under the token limit. To determine the token size without making an api call we can use the tiktoken library. To begin we only began with prompts + masked sentences for molecules < 4905 tokens. The responses were fed through the web interface for GPT 3.5 and responses were manually saved in dictionaries with the key being the primary name of the molecule and the value being the label from GPT. The results for the one-answer prompts and justify prompts for these molecules < 4905 tokens are below:

GPT one-answer prompt results:  
<img width="400" alt="image" src="https://github.com/Varun-Krishnan1/OpticNerveRegenNLP/assets/19865419/86b2b20f-fb83-4285-970e-bed28867784c">

GPT justify-answer prompt results:  
<img width="405" alt="image" src="https://github.com/Varun-Krishnan1/OpticNerveRegenNLP/assets/19865419/f3eef0e6-719f-4519-8fed-6ac22fa0b018">


To test the molecules that had a prompt size + masked sentence size > 4905 tokens we split the sentences for each given molecule into separate files that were each under the token limit. The files were split by sentences so a sentence did not get cut off between files. To get the results for a given molecule we gave separate GPT instances the sentences from each file for a molecule. We then took the mode of the class label from each file that GPT outputted and used that as the class label for the molecule. We only used the **justify answer** prompt due to it superior performance on the molecules with sentences <4905 tokens. The results for these molecules > 4095 tokens are below.      
<img width="390" alt="image" src="https://github.com/Varun-Krishnan1/OpticNerveRegenNLP/assets/19865419/cbe992bb-516a-4506-92ea-8f6fc14e833b">      


You can see the f-1 score is excellent here and surpasses that of the molecules < 4095 tokens. This is perhaps due to the molecules with more tokens are more studied in literature and therefore, have more clear categorizations as promoter or inhibitor. These molecules also perhaps have greater representation from recent years where recent years have higher-quality data for GPT to interpret. Also more sentences allows GPT to have more data to work with.

**Overall Results from Known Molecules**

The results for both molecules <4905 tokens and molecules >4905 tokens are below.

All Molecules (41 molecules) Results:  
<img width="411" alt="image" src="https://github.com/Varun-Krishnan1/OpticNerveRegenNLP/assets/19865419/9778eadf-88e4-40fb-a1e2-f89e5801098f">

**Evaluating GPT on Wet-Lab Labelled Molecules**  
Just like the BERT model we used GPT to see how well it could evaluate molecules manually labelled from the wet lab team. These molecules would be more challenging than the known molecules because these would be lesser known and most likely have less literature associated with it.

We first went through the wet lab team labels and manually converted each label as: "P" - promoter, "I" - Inhibitor, "N" - Neither, and "B" - Both. These were saved in new excel files with the wet lab team member and "Curated" appended at the end of the file name (eg, CK_Molecules_To_Label_Curated.xlsx). We extracted 111 molecules with 69 being promoters and 42 being inhibitors for this evaluation. However, after going through the corpus only 78 molecules were able to be found using their primary name and other names. This is most likely due to difficulty of finding the molecules in a sentence due to things like punctuation, end of sentence handling, etc. From these we had to remove any molecules that were also known molecules which brought our molecules down to 73. This consisted of 44 promoters and 29 inhibitors. These masked sentences were saved as a json at: Code/Sentence Classification/Output/Combined_Sentences_Per_Molecule/masked_wet_lab_73_molecules.json

Now we created a prompt in the same manner as above for the known molecules. The molecules were also split by tokens < 4905 and tokens > 4095 (including the prompt). 60 molecules were <4905 tokens and 13 molecules were >4905 tokens. Just like above if GPT did not provide an answer we would say "Please pick an option". The results are below:

Results for Wet Lab Labeled Molecules <4905 (57 molecules):  
<img width="425" alt="image" src="https://github.com/Varun-Krishnan1/OpticNerveRegenNLP/assets/19865419/33909758-4253-43ac-9d40-6ace57793f9c">  
You can see the overall accuracy is no better than random guessing. However, the f-1 for the promoter labels were relatively good meaning GPT is good at labelling promoters but not good at labelling inihibitors.

Results for Wet Lab Labeled Molecules >4905 (13 molecules):  
<img width="408" alt="image" src="https://github.com/Varun-Krishnan1/OpticNerveRegenNLP/assets/19865419/b87be481-2d9a-4b64-b0f5-41e3924af4c9">  
You can see the f-1 and accuracy is way better for molecules with more sentences which is similar to the known molecules results we got. Again predicting promoters were better than predicting inhibitors.

Results for All Molecules (70 molecules):  
<img width="421" alt="image" src="https://github.com/Varun-Krishnan1/OpticNerveRegenNLP/assets/19865419/93c8db38-8204-4875-896d-0fa7ead5f498">

### GPT with Confidence Scores

Could we somehow quantitively evaluate how "much" of a promoter or inhibitor a molecule is? To answer this question we can ask GPT to give us a confidence score on its label for a molecule. The following prompt was used for these purposes:     
> Given the following sentences from a scientific study in the field of optic nerve regeneration, where a specific molecule is masked, can you determine whether the masked molecule acts as a promoter or inhibitor of optic nerve regeneration? Provide your best guess, a confidence score from 0 (no confidence) to 100 (absolute confidence), and justify your answer based on the context given in the sentences. Here are the sentences:

This prompt was followed by the masked sentences for the molecules. Here are the results of the prompt with molecules <4905 tokens: 
<img width="406" alt="image" src="https://github.com/Varun-Krishnan1/OpticNerveRegenNLP/assets/19865419/739ed206-c3f0-4708-bb25-ac5812174c22">        


**It actually performs exactly equivalent to our previous justify answer prompt demonstrating both the reliability of GPT's responses as well as no change in effectiveness by adding confdience scores** 

Here are the results of the prompt with molecules >4905 tokens:      
<img width="406" alt="image" src="https://github.com/Varun-Krishnan1/OpticNerveRegenNLP/assets/19865419/a6ce7578-555f-4676-9c19-8f75404980f9">      

**It actually performs a little bit worse to our previous justify answer prompt but still demonstrates both the reliability of GPT's responses as well as limited change in effectiveness by adding confdience scores** 

Total Molecules:     
<img width="386" alt="image" src="https://github.com/Varun-Krishnan1/OpticNerveRegenNLP/assets/19865419/5025bb2f-5e26-4c44-b173-22c834fb219f">     

**Again Very Similar to without confidence score**

### BERT Revisited

After creating the masked molecule dataset above for GPT we can use this as a better dataset to train BERT. Recall BERT was previously trained on a per-sentence basis with the molecules from the sentence removed rather than on groups of sentences with the molecules masked. The code to train a BERT model with this new dataset is found in BERT/Masked Molecule Sentences - BERT 2 Class Training.ipynb.

The steps to train BERT on this dataset is as follows:

1. Load in the masked molecule sentence dataset which consists of the molecule name as the key and the sentences of the molecule as the value.
2. For BERT we need the keys to be the sentences and the values to be the label as 1 or 0. Conver the above dictionary into this format.
3. With this we get the 24 promoters and 17 inhibitors with an average character length of 27,479 for a molecule's sentences.
4. For BERT the maximum token-size allowed is 512. We will have to split our sentences's for a given molecule, using sentence boundaries so as not to cut the sentence in half, into chunks with < 512 tokens. This is similar with what we had to do with GPT but with a token size of 4906. We now have 247 promoter sentence chunks and 348 inhibitor sentence chunks.
5. We then create our BERT model from huggingface. The tokenizer used is the 'bert-base-uncased' tokenizer with max_lenth=512. The model used is the BertForSequenceClassification which adds a classifier layer on top of the BERT model to help us classify our sentences.
6. We then train our model with the below paramaters and the results on the validation sentences are shown below.

Training arguments for BertForSequenceClassification:  
<img width="639" alt="image" src="https://github.com/Varun-Krishnan1/OpticNerveRegenNLP/assets/19865419/fd7237cd-736a-4c67-b837-387988f34eea">

Training and Validation Loss:  
<img width="648" alt="image" src="https://github.com/Varun-Krishnan1/OpticNerveRegenNLP/assets/19865419/2a851013-8ded-4262-bdf2-43436bba5820">

Results on Validation Sentence Chunks for BERT:  
<img width="388" alt="image" src="https://github.com/Varun-Krishnan1/OpticNerveRegenNLP/assets/19865419/4fb31f4a-a39d-4990-9491-0506101b22cb">  
As you can see the results are very good on the validation dataset but it's overfitted.

### Combatting Overfitting

As you can see from the previous results the model definitely overfits. The BERT model from huggingface already implements dropout with a default value of .1 drouput. L2 regularization is also implemented by the weight_decay parameter. Let's keep it simple then and only do 5 epochs so it stops halfway through (which is about where it started to overfit).

Training arguments for early stopping:  
<img width="650" alt="image" src="https://github.com/Varun-Krishnan1/OpticNerveRegenNLP/assets/19865419/066b440c-6be2-4d85-8982-f56ed60a7784">

Training and Validation Loss:  
<img width="624" alt="image" src="https://github.com/Varun-Krishnan1/OpticNerveRegenNLP/assets/19865419/a6a1e403-9a5a-4e68-bf22-56a5d4beb32c">

Results on Validation Sentence Chunks:  
<img width="410" alt="image" src="https://github.com/Varun-Krishnan1/OpticNerveRegenNLP/assets/19865419/822b59c3-605a-4672-88b2-762c5d72b0e6">    
As you can see the f-1 is way worse when you don't allow it to overfit. Looks like it's just guessing 0 a lot.           

TO DO: Evaluate with wet lab molecules but have to figure out how to compare predicted_labels and true_labels because predicted_labels are based off off sentence chunks whereas true labels are just the molecule labels. Need to find a way to figure out which molecule is in the sentence chunk and assign that sentence chunk that correct label...

### Data Curation for better BERT Results 
A potential problem with the above BERT approach is that it is training on not good quality sentences and therefore, it is learning erroneous associations that cause it to perform poorly. Furthermore, the classes are unbalanced when splitting into sentence chunks. To fix the low-quality sentences issue, we manually inspected the data and found the promoters 'l1' and 'c3' were picked up in a lot of sentences that didn't have to do with those molecules. Therefore we removed these molecules from the training sentences and to counteract the loss of promoters we removed 'mag' and 'rock' from the inhibitors. Notably, 'rock' is also picked up in a lot of sentences where it is not actually referring to the molecule. These erroneous sentences for 'l1', 'c3', and 'rock' were primarily from earlier papers.    

After removing these low-quality sentences we got much better results. We first trained it using 10 epochs which resulted in some overfitting:    
<img width="604" alt="image" src="https://github.com/Varun-Krishnan1/OpticNerveRegenNLP/assets/19865419/e5ed22a7-1eb1-45e1-ba76-f904e0947883">      

Therefore we trained it for only 8 epochs resulting in the following training and validation loss:      
<img width="577" alt="image" src="https://github.com/Varun-Krishnan1/OpticNerveRegenNLP/assets/19865419/f11ebaa5-ebd9-4bd4-b603-28970ff89ec1">     

The f-1 scores for the test holdout sentences after training were pretty good and comparable to that of GPT on the known molecules/inhibitors:     
 <img width="428" alt="image" src="https://github.com/Varun-Krishnan1/OpticNerveRegenNLP/assets/19865419/a60b5195-348b-4688-95b5-4957bd7522ea">     

We then tested our model on the wet lab team sentences. This was done on a per-sentence basis meaning the molecule's masked sentences were split into <512 tokens for BERT to handle and each sentence was evaluated to see if BERT correctly labeled that sentence as belonging to a promoter or inhibitor. There were 142 sentences from inhibitor molecules and 312 from promoter molecules:       
<img width="401" alt="image" src="https://github.com/Varun-Krishnan1/OpticNerveRegenNLP/assets/19865419/88b9fce2-d7e5-4da7-a898-0ca3e4cd7bfc">      

Then, to see how BERT could label entire molecules rather than sentences we took the mode of all the predicted labels for each sentence of a molecule and use that mode predicted label as the predicted label for that molecule. The results are very good:     
<img width="383" alt="image" src="https://github.com/Varun-Krishnan1/OpticNerveRegenNLP/assets/19865419/2d047387-5ec1-4047-86bd-103a3bf6f12c">      

We then stratified molecules by having total sentences >4905 GPT tokens vs <4905 GPT tokens to see how it compared to GPT since GPT performed much better on molecules with more sentences. **It actually outperformed GPT in lower token total sentences but worse in higher token total sentences.**      
<img width="440" alt="image" src="https://github.com/Varun-Krishnan1/OpticNerveRegenNLP/assets/19865419/9ac448a3-1621-4f9d-9020-f0bb9dc635f8">      
Greater than 4905 tokens:      
<img width="387" alt="image" src="https://github.com/Varun-Krishnan1/OpticNerveRegenNLP/assets/19865419/ef1aa779-2e2e-4297-b75c-4952acead315">

### BERT Confidence Scores 
The BERT labeling of sentences was based on the biggest probability of the softmax output. However we can use the raw probabilties to get a confidence score for a predicted label. To see if higher confidence scores resulted in higher accuracy predicted_labels we can do a T-test between the means of the confidence scores that were not accurate and confidence scores for sentences that were accurate.   

These were done a per-sentence basis. 

Histogram of all confidence scores on a per-sentence basis:     
<img width="495" alt="image" src="https://github.com/Varun-Krishnan1/OpticNerveRegenNLP/assets/19865419/2e58c0da-6946-41e8-93fd-924aa50df05c">     

Histogram of all confidence scores on a per-sentence basis:         
<img width="482" alt="image" src="https://github.com/Varun-Krishnan1/OpticNerveRegenNLP/assets/19865419/d70c916a-e986-47be-aba4-b76da8852bc6">      

T-test on a per-sentence for known molecules/inhibitors:      
<img width="232" alt="image" src="https://github.com/Varun-Krishnan1/OpticNerveRegenNLP/assets/19865419/4c98a663-bd86-46ff-9165-a6ed7c6cc872">          

T-test on a per-sentence for wet lab molecules:      
<img width="267" alt="image" src="https://github.com/Varun-Krishnan1/OpticNerveRegenNLP/assets/19865419/3efc3218-32cc-473c-870a-00c1b5aa7f04">     

**Was better at using confidence scores for the known molecules/inhibitors than the wet lab team ones on a per-sentence basis**

When looking at confidence scores on a **per-molecule basis** however, then it looks like the scores became significant. We used the average confidence scores across all of a molecule's sentences to determine that molecule's confidence score. 

Histogram of confidence scores on a per-molecule basis:     
<img width="479" alt="image" src="https://github.com/Varun-Krishnan1/OpticNerveRegenNLP/assets/19865419/431d60b2-7df5-45bb-b613-d83e25595aea">     

T-test on a per-molecule basis for wet-lab molecules: 
<img width="237" alt="image" src="https://github.com/Varun-Krishnan1/OpticNerveRegenNLP/assets/19865419/cae3cdce-8f40-4db2-b3c0-0e641712a507">

Even though, it is significant you can see the means are 0.8 vs 0.84 so practically speaking not sure how useful it is....


## Future Steps

- [✔️] Regraph Word2Vec embeddings with known molecules in graphsage and see if you get similar clusters that group on pathways
  - [✔️] Get F1 when comparing known molecules to “promoter” and “inhibitor”
  - [✔️] Get F1 when comparing wet lab labelled molecules to “promoter” and “inhibitor”
- [✔️] Graph BPE and see how clusters are
  - [✔️] Create a BPE model trained just on recent year and see how that does 
  - [✔️] Get F1 when comparing known molecules to “promoter” and “inhibitor”
  - [✔️] Get F1 when comparing wet-label molecules to “promoter” and “inhibitor”
- [✔️] Use confidence scores with known molecules using GPT and see if they correlate to accuracy of predicted label - they do NOT 
- [✔️] Check BERT model confidence scores on a per molecule basis
- [ ] Create Slides for results and convert Readme.md to word document 
- [ ] Meet with Dr. J to see next steps for final GPT model for known molecules and 73 wet lab labeled 
- [ ] For final decided GPT model remove l1, c3, mag, and rock from results since BERT did not use those for known molecules
- [ ] Test BERT model on explicit sentences to see how it does
- [ ] If you really want to evaluate performance you need to have manual graders for the masked sentences that GPT and BERT are given and see how well they do to a manual labeler. Because right now we are seeing its accuracy when compared to someone that has access to all literature to make a classification.
- [❌] Using Logistic Regression with Wet Lab Molecules how is the F1? -> However would have to convert wet lab molecule sentences to freq vectors so not straightforward will take time probably not worth it since won't be included in final paper
- [❌] Using Naive-Bayes with Wet Lab Molecules how is the F1? -> However would have to convert wet lab molecule sentences to smoothed freq vectors so not straightforward will take time probably not worth it since won't be included in final paper
- [❌] Using GPT4 which would allow for longer token sizes for input. Unfortunately the cost is very high and the web API has a cap of 25 messages every 3 hours.
