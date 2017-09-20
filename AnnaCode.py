from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import numpy as np
import string
from gensim.models.word2vec import Word2Vec
from sklearn.svm import LinearSVC,SVC
from sklearn.metrics import accuracy_score,classification_report
from sklearn.feature_extraction.text import normalize
from scipy.sparse import *
from scipy import *
import cPickle
from nltk.util import ngrams
from nltk import sent_tokenize,word_tokenize
vect_mode = 'tfidf'
# glove_fname = 'E:/glove.6B.50d.txt'
# glove_vect_dim = int([p[:-1] for p in os.path.basename(glove_fname.split('.')) if p.endswith('d')][0])
# glove_vect_dim = 50
max_features = None
deep_kernel_mode = 'diag'
w2v_vect_dim = 100
num_of_docs = 20000

def write_new_file(start,end,sentences):
    with open('C:/Users/admin/FYP/r8-test-bigrams.txt','a') as f:
        for s in sentences:
            f.write(s+'\n')
            for i in range(start,end+1):
                if i != 1:
                    grams = ngrams(word_tokenize(s),i)
                    f.write(' '.join(['_'.join(gram) for gram in grams]) +'\n')


def get_Ngrams(sen,start,end):
    output = []
    for i in range(start,end+1):
        grams =  ngrams(word_tokenize(sen),i)
        output.append(['_'.join(gram) for gram in grams])
    return output

def display_results (y_pred,y_true):
    print 'classification report: ', classification_report(y_pred=y_pred, y_true=y_true)
    print 'acc: ', accuracy_score(y_true=y_true, y_pred=y_pred)

def bow_classify (X_train,train_labels,X_test,test_labels):
    model = LinearSVC()
    model.fit(X_train,train_labels)
    y_pred = model.predict(X_test)
    print '***bow report***'
    display_results(y_pred=y_pred, y_true=test_labels)
    print '***end of bow report***'

def kernel_svm_classify(deep_doc_kernel_train,train_labels,deep_doc_kernel_test,test_labels):
    model = SVC(kernel='precomputed')
    model.fit(deep_doc_kernel_train, train_labels)
    y_pred = model.predict(deep_doc_kernel_test)
    print '***deep doc kernel report***'
    display_results(y_pred=y_pred, y_true=test_labels)
    print '***end of deep doc kernel report***'

train_labels = [l.split('\t')[0] for l in open('C:/Users/admin/FYP/DataSet/r8-train-all-terms.txt')]
train_labels_bigrams =[]
# train_labels_trigrams =[]
wv2_train = [l.strip().translate(string.maketrans('\n',' ')) for l in open('r8-train-test-bigrams.txt')]
train_sents = [l.split('\t')[1].strip() for l in open('C:/Users/admin/FYP/DataSet/r8-train-all-terms.txt')]
train_sents_bigrams = [l.strip().translate(string.maketrans('\n',' ')) for l in open('r8-train-all-terms-bigrams.txt')]
# train_sents_trigrams = [l.strip().translate(string.maketrans('\n',' ')) for l in open('C:/Users/admin/FYP/r8-train-all-terms-trigrams.txt')]
test_labels = [l.split('\t')[0] for l in open('./DataSet/r8-test-all-terms.txt')]
test_sents = [l.split('\t')[1].strip() for l in open('./DataSet/r8-test-all-terms.txt')]
test_sents_bigrams = [l.strip().translate(string.maketrans('\n',' ')) for l in open('r8-test-bigrams.txt')]
for i in train_labels:
    train_labels_bigrams.append(i)
    train_labels_bigrams.append(i)
# for i in train_labels:
#     train_labels_trigrams.append(i)
#     train_labels_trigrams.append(i)
#     train_labels_trigrams.append(i)
# write_new_file(1,2,test_sents)
# print("DONE")
# print get_Ngrams("This sentence is for testing",1,3)
vectorizer =TfidfVectorizer(lowercase=True,max_features=max_features,ngram_range=(1,1),stop_words='english')
X_train = vectorizer.fit_transform(train_sents_bigrams[:num_of_docs])
vocab_w2v = vectorizer.get_feature_names()
X_test = vectorizer.transform(test_sents_bigrams)
print 'obtained a vocab of len: {} from the training + testing set'.format(len(vocab_w2v))
# model = Word2Vec([i.translate(string.maketrans('\n',' ')).split() for i in wv2_train[:num_of_docs]],size=w2v_vect_dim,min_count=1)
# model.save('C:/Users/admin/FYP/modelBigrams')
# print("DONE")
model = Word2Vec.load('modelBigrams')
w2v = dict(zip(model.wv.index2word, model.wv.syn0))
# bow_classify (X_train,train_labels,X_test,test_labels)

if deep_kernel_mode == 'diag':
    # word kernel is a DIAG matrix
    word_kernel = csr_matrix((len(vocab_w2v),len(vocab_w2v)))
    for i,w in enumerate(vocab_w2v):
        print i
        word_vec = w2v.get(w,np.zeros(shape=(w2v_vect_dim,)))
        word_kernel[i,i] = word_vec.dot(word_vec.T)

elif deep_kernel_mode == 'pairwise':
    # word kernel is pairwise similarity
    # word_vect_as_in_vocab = np.zeros(shape=(len(vocab_w2v), w2v_vect_dim))
    word_vect_as_in_vocab = csr_matrix((len(vocab_w2v), int(w2v_vect_dim)))
    for i,w in enumerate(vocab_w2v):
        word_vect_as_in_vocab[i] = w2v.get(w,np.zeros(shape=(w2v_vect_dim,)))
    # print word_vect_as_in_vocab.shape
    word_kernel = word_vect_as_in_vocab.dot(word_vect_as_in_vocab.T)
else:
    with open('save.p','rb') as f:
        word_kernel = cPickle.load(f)
# print 'word vectors matrix prepared acc to vocab and its shape is: ', word_vect_as_in_vocab.shape
print 'word KERNEL matrix prepared acc to vocab and its shape is: ', word_kernel.shape
cPickle.dump(word_kernel,open('word_kernel_bigrams.p','wb'))
print("DONE")
deep_doc_kernel_train = X_train.dot(word_kernel)
deep_doc_kernel_train = deep_doc_kernel_train.dot(X_train.T).todense()
deep_doc_kernel_test = X_test.dot(word_kernel)
print 'deep doc kernel test matrix shape: ',deep_doc_kernel_test.shape
deep_doc_kernel_test = deep_doc_kernel_test.dot(X_train.T).todense()

print 'deep doc kernel train matrix shape: ',deep_doc_kernel_train.shape
print 'deep doc kernel test matrix shape: ',deep_doc_kernel_test.shape

kernel_svm_classify(deep_doc_kernel_train,train_labels_bigrams[:num_of_docs],deep_doc_kernel_test,test_labels)