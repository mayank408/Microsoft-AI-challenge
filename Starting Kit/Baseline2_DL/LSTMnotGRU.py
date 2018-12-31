from keras.models import Sequential
from keras.layers import LSTM, GRU
from keras.layers import Dense
from keras.layers import TimeDistributed
from keras.layers import Bidirectional, Embedding
from keras.utils import np_utils
from keras import backend as K

import re
import numpy as np
GloveEmbeddings = {}
max_query_words = 12
max_passage_words = 50
emb_dim = 50
num_classes= 1
vocab_size = 0
#The following method takes Glove Embedding file and stores all words and their embeddings in a dictionary
def loadEmbeddings(embeddingfile):
    global GloveEmbeddings,emb_dim

    fe = open(embeddingfile,"r",encoding="utf-8",errors="ignore")
    for line in fe:
        tokens= line.strip().split()
        word = tokens[0]
        vec = np.asarray(tokens[1:], dtype='float32')
        GloveEmbeddings[word]=vec
    #Add Zerovec, this will be useful to pad zeros, it is better to experiment with padding any non-zero constant values also.
    GloveEmbeddings["zerovec"] = [0.0]*emb_dim
    fe.close()

def model(embedding_matrix):
	global vocab_size
	model_glove = Sequential()
	model_glove.add(Embedding(vocab_size, 50, input_length=62, weights=[embedding_matrix], trainable=False))
	model_glove.add(Bidirectional(LSTM(62, dropout_U = 0.2, dropout_W = 0.2), merge_mode='concat'))

	model_glove.add(Dense(num_classes, activation='sigmoid'))
	model_glove.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model_glove

def BiLSTM(trainFileName, evaluationFileName, submissionfile):
	global GloveEmbeddings,emb_dim, vocab_size
	f = open(trainFileName,"r",encoding="utf-8",errors="ignore")
	word_index = {}
	index = 0
	for line in f:
		tokens = line.strip().lower().split("\t")
		query_id,query,passage,label = tokens[0],tokens[1],tokens[2],tokens[3]


		words = re.split('\W+', query)
		words = [x for x in words if x] # to remove empty words 
		for word in words:	
			if word not in word_index:
				word_index[word] = index				#will it not change if words are duplicate
				index+=1
				vocab_size+=1

		words = re.split('\W+', passage)
		words = [x for x in words if x] # to remove empty words 
		for word in words:
			if word not in word_index:						#same as above this code
				word_index[word] = index
				index+=1
				vocab_size+=1
                
	word_index["zerovec"] = index
	vocab_size+=1
	f.close()

# 	print(vocab_size)
# 	print(len(word_index))
 	   
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
		words = re.split('\W+', query)
		words = [x for x in words if x] # to remove empty words 
		word_count = len(words)
		remaining = max_query_words - word_count  
		if(remaining>0):
			words += ["zerovec"]*remaining # Pad zero vecs if the word count is less than max_query_words
		words = words[:max_query_words] # trim extra words

		for word in words:
			input_vector.append(word_index[word])
        
		#***** Passage Processing **********
		words = re.split('\W+', passage)
		words = [x for x in words if x] # to remove empty words 
		word_count = len(words)
		remaining = max_passage_words - word_count
		if(remaining>0):
			words += ["zerovec"]*remaining # Pad zero vecs if the word count is less than max_passage_words
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
		words = re.split('\W+', query)
		words = [x for x in words if x] # to remove empty words 
		word_count = len(words)
		remaining = max_query_words - word_count  
		if(remaining>0):
			words += ["zerovec"]*remaining # Pad zero vecs if the word count is less than max_query_words
		words = words[:max_query_words] # trim extra words

		for word in words:
			if word in word_index:
				input_vector.append(word_index[word])
			else:
				input_vector.append(word_index["zerovec"])
        
		#***** Passage Processing **********
		words = re.split('\W+', passage)
		words = [x for x in words if x] # to remove empty words 
		word_count = len(words)
		remaining = max_passage_words - word_count  
		if(remaining>0):
			words += ["zerovec"]*remaining # Pad zero vecs if the word count is less than max_passage_words
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
	all_scores = {}

	for it in range(len(temp)):
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



	# for it in range(len(eval_input)):
		
	# 	row = eval_input[it]
		
	# 	query_id = queries[it]
		
		
	# 	if(query_id in all_scores):
	# 		all_scores[query_id].append(temp)
	# 	else:
	# 		all_scores[query_id] = [temp]
	# fw = open(submissionfile,"w",encoding="utf-8")
	# for query_id in all_scores:
	# 	print("S")
	# 	scores = all_scores[query_id]
	# 	scores_str = [str(sc) for sc in scores] # convert all scores to string values
	# 	scores_str = "\t".join(scores_str) # join all scores in list to make it one string with  tab delimiter.  
	# 	fw.write(query_id+"\t"+scores_str+"\n")
	



if __name__ == "__main__":

    trainFileName = "30splitdata.tsv"
    evaluationFileName= "../eval1_unlabelled.tsv"
    #validationFileName = "validationdata.tsv"
    #EvaluationFileName = "eval1_unlabelled.tsv"

    embeddingFileName = "glove.6B.50d.txt"
    submissionFileName = "answer.tsv"

    loadEmbeddings(embeddingFileName) 
    BiLSTM(trainFileName, evaluationFileName, submissionFileName)
