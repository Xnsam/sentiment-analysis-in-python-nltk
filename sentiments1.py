#necessary imports

from nltk.classify import NaiveBayesClassifier
from nltk.corpus import subjectivity
from nltk.sentiment import SentimentAnalyzer
from nltk.sentiment.util import *
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk import tokenize


#to download the required from nltk
nltk.download('subjectivity')
nltk.download('vader_lexicon')



n_instances = 100

# load subjective and objective texts
subj_docs = [(sent, 'subj') for sent in subjectivity.sents(categories='subj')[:n_instances]]
obj_docs = [(sent, 'obj') for sent in subjectivity.sents(categories='obj')[:n_instances]]


# splitting into train and test sets
train_subj_docs = subj_docs[:80]
test_subj_docs = subj_docs[80:100]
train_obj_docs = obj_docs[:80]
test_obj_docs = obj_docs[80:100]
training_docs = train_subj_docs + train_obj_docs
testing_docs = test_subj_docs + test_obj_docs

#creating a sentiment analyzer object
sentim_analyzer = SentimentAnalyzer()

all_words_neg = sentim_analyzer.all_words([mark_negation(doc) for doc in training_docs])

unigram_feats = sentim_analyzer.unigram_word_feats(all_words_neg, min_freq=4)
sentim_analyzer.add_feat_extractor(extract_unigram_feats, unigrams = unigram_feats)

#apply the features
training_set = sentim_analyzer.apply_features(training_docs)
test_set = sentim_analyzer.apply_features(testing_docs)

#train the classifier
trainer = NaiveBayesClassifier.train
classifier = sentim_analyzer.train(trainer, training_set)

#get the user input
user_input = str(input("Enter your text\n"))
user_input = tokenize.sent_tokenize(user_input)
#print(user_input)
sid = SentimentIntensityAnalyzer()
for s in user_input:
 print(s)
 ss = sid.polarity_scores(s)
 for k in sorted(ss):
  print('{0}: {1}, '.format(k, ss[k]), end='')
 print()
