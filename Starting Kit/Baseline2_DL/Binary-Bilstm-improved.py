import re
import numpy as np

GloveEmbeddings = {}
emb_dim = 300
max_query_words = 12
max_passage_words = 100

def loadEmbeddings(embeddingfile):
    global GloveEmbeddings,emb_dim

    fe = open(embeddingfile,"r",encoding="utf-8",errors="ignore")
    for line in fe:
        tokens= line.strip().split()
        word = tokens[0]
        try:
            vec = np.asarray(tokens[1:], dtype='float32')
            GloveEmbeddings[word]=vec
        except ValueError:
            print('Invalid value!')    
    #Add Zerovec, this will be useful to pad zeros, it is better to experiment with padding any non-zero constant values also.
    GloveEmbeddings["zerovec"] = [0.0]*emb_dim
    fe.close()
    print("embeddings loaded")


vocab_size = 0
num_classes = 1

from keras.models import Sequential
from keras.layers import LSTM, GRU
from keras.layers import Dense, Flatten
from keras.layers import TimeDistributed
from keras.layers import Bidirectional, Embedding
from keras.utils import np_utils
from keras import backend as K

def model(embedding_matrix):
    global vocab_size
    model_glove = Sequential()
    model_glove.add(Embedding(vocab_size, 300, input_length=112, weights=[embedding_matrix], trainable=False))
    model_glove.add(Bidirectional(GRU(112, dropout_U = 0.2, dropout_W = 0.2), merge_mode='concat'))
    model_glove.add(Dense(num_classes, activation='sigmoid'))
    model_glove.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model_glove




import nltk

def BiLSTM(trainFileName, evaluationFileName, submissionfile):
    global GloveEmbeddings,emb_dim, vocab_size
    f = open(trainFileName,"r",encoding="utf-8",errors="ignore")
    word_index = {}
    index = 0
    for line in f:
        tokens = line.strip().lower().split("\t")
        query_id,query,passage,label = tokens[0],tokens[1],tokens[2],tokens[3]

        words = nltk.word_tokenize(query)
        words = [x for x in words if x] # to remove empty words 
        for word in words:  
            if word not in word_index:
                word_index[word] = index                #will it not change if words are duplicate
                index+=1
                vocab_size+=1

        words = nltk.word_tokenize(passage)
        words = [x for x in words if x] # to remove empty words 
        for word in words:
            if word not in word_index:                      #same as above this code
                word_index[word] = index
                index+=1
                vocab_size+=1
                
    word_index["zerovec"] = index
    vocab_size+=1
    f.close()
    
    
    
    embedding_matrix = np.zeros((vocab_size,emb_dim))
    for key in word_index:
        if key in GloveEmbeddings:
            embedding_matrix[word_index[key]] = GloveEmbeddings[key]
        else:     
            embedding_matrix[word_index[key]] = GloveEmbeddings["zerovec"]

    f = open(trainFileName,"r",encoding="utf-8",errors="ignore")
    input_to_bilstm = []
    labels = []
    for line in f:
        tokens = line.strip().lower().split("\t")
        query_id,query,passage,label = tokens[0],tokens[1],tokens[2],tokens[3]
        input_vector = []

        labels.append(label)

        #****Query Processing****
        words = nltk.word_tokenize(query)
        words = [x for x in words if x] # to remove empty words 
        word_count = len(words)
        remaining = max_query_words - word_count  
        if(remaining>0):
            words = ["zerovec"]*remaining + words # Pad zero vecs if the word count is less than max_query_words
        words = words[:max_query_words] # trim extra words

        for word in words:
            input_vector.append(word_index[word])
        
        #***** Passage Processing **********
        words = nltk.word_tokenize(passage)
        words = [x for x in words if x] # to remove empty words 
        word_count = len(words)
        remaining = max_passage_words - word_count
        if(remaining>0):
            words = ["zerovec"]*remaining + words # Pad zero vecs if the word count is less than max_passage_words
        words = words[:max_passage_words] # trim extra words

        for word in words:
            input_vector.append(word_index[word])

        input_to_bilstm.append(input_vector)
    f.close()
    n_batch = 1000
    model_bilstm = model(embedding_matrix)
    model_bilstm.fit(np.array(input_to_bilstm),np.array(labels), batch_size=n_batch, validation_split=0.1, epochs = 20)
    
    f = open(evaluationFileName,"r",encoding="utf-8",errors="ignore")
    eval_input = []
    queries = []
    for line in f:
        tokens = line.strip().lower().split("\t")
        query_id,query,passage = tokens[0],tokens[1],tokens[2]
        input_vector = []
        #****Query Processing****
        words = nltk.word_tokenize(query)
        words = [x for x in words if x] # to remove empty words 
        word_count = len(words)
        remaining = max_query_words - word_count  
        if(remaining>0):
            words = ["zerovec"]*remaining + words # Pad zero vecs if the word count is less than max_query_words
        words = words[:max_query_words] # trim extra words

        for word in words:
            if word in word_index:
                input_vector.append(word_index[word])
            else:
                input_vector.append(word_index["zerovec"])
        
        #***** Passage Processing **********
        words = nltk.word_tokenize(passage)
        words = [x for x in words if x] # to remove empty words 
        word_count = len(words)
        remaining = max_passage_words - word_count  
        if(remaining>0):
            words = ["zerovec"]*remaining + words# Pad zero vecs if the word count is less than max_passage_words
        words = words[:max_passage_words] # trim extra words

        for word in words:
            if word in word_index:
                input_vector.append(word_index[word])
            else:
                input_vector.append(word_index["zerovec"])

        eval_input.append(input_vector) 
        queries.append(query_id)
    f.close()
    
    # inp = model_bilstm.input                                           # input placeholder
    # outputs = [layer.output for layer in model_bilstm.layers]          # all layer outputs
    # functor = K.function([inp]+ [K.learning_phase()], outputs )        # evaluation function
    # all_scores = {}
    # print(len(eval_input))
    # print(len(queries))

    temp = model_bilstm.predict(np.array(eval_input))
    print(temp[0])
    all_scores = {}

    for it in range(len(temp)):
        if(it%10000 == 0):
            print(it)
        score = temp[it]
        query_id = queries[it]
        if(query_id in all_scores):
            all_scores[query_id].append(score)
        else:
            all_scores[query_id] = [score]

    fw = open(submissionfile,"w",encoding="utf-8")
    for query_id in all_scores:
        scores = all_scores[query_id]
        scores_str = [str(sc) for sc in scores] # convert all scores to string values
        scores_str = "\t".join(scores_str) # join all scores in list to make it one string with  tab delimiter.  
        fw.write(query_id+"\t"+scores_str+"\n")
    fw.close()


embeddingFileName = "glove.840B.300d.txt"
loadEmbeddings(embeddingFileName)


trainFileName = "30-Binary.tsv"
evaluationFileName= "../eval1_unlabelled.tsv"

submissionFileName = "answer.tsv"

BiLSTM(trainFileName, evaluationFileName, submissionFileName)