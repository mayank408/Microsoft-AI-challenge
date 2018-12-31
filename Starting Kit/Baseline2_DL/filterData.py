import math
import pickle
import string
import re
#Initialize Global variables
docIDFDict = {}
avgDocLength = 0


def GetCorpus(inputfile,corpusfile):
    f = open(inputfile,"r",encoding="utf-8")
    fw = open(corpusfile,"w",encoding="utf-8")
    for line in f:
        passage = line.strip().lower().split("\t")[2]
        fw.write(passage+"\n")
    f.close()
    fw.close()

# The following IDF_Generator method reads all the passages(docs) and creates Inverse Document Frequency(IDF) scores for each unique word using below formula
# IDF(q_i) = log((N-n(q_i)+0.5)/(n(q_i)+0.5)) where N is the total number of documents in the collection and n(q_i) is the number of documents containing q_i
# After finding IDF scores for all the words, The IDF dictionary will be saved in "docIDFDict.pickle" file in the current directory

def IDF_Generator(corpusfile, delimiter=' ', base=math.e) :

    global docIDFDict,avgDocLength

    docFrequencyDict = {}
    numOfDocuments = 0
    totalDocLength = 0

    for line in open(corpusfile,"r",encoding="utf-8") :
        doc = line.strip().split(delimiter)
        totalDocLength += len(doc)

        doc = list(set(doc)) # Take all unique words

        for word in doc : #Updates n(q_i) values for all the words(q_i)
            if word not in docFrequencyDict :
                docFrequencyDict[word] = 0
            docFrequencyDict[word] += 1

        numOfDocuments = numOfDocuments + 1
        if (numOfDocuments%5000==0):
            print(numOfDocuments)

    for word in docFrequencyDict:  #Calculate IDF scores for each word(q_i)
        docIDFDict[word] = math.log((numOfDocuments - docFrequencyDict[word] +0.5) / (docFrequencyDict[word] + 0.5), base) #Why are you considering "numOfDocuments - docFrequencyDict[word]" instead of just "numOfDocuments"

    avgDocLength = totalDocLength / numOfDocuments


    pickle_out = open("docIDFDict.pickle","wb") # Saves IDF scores in pickle file, which is optional
    pickle.dump(docIDFDict, pickle_out)
    pickle_out.close()


    print("NumOfDocuments : ", numOfDocuments)
    print("AvgDocLength : ", avgDocLength)



#The following GetBM25Score method will take Query and passage as input and outputs their similarity score based on the term frequency(TF) and IDF values.
def GetBM25Score(Query, Passage, k1=1.5, b=0.75, delimiter=' ') :

    global docIDFDict,avgDocLength

    query_words= Query.strip().lower().split(delimiter)
    passage_words = Passage.strip().lower().split(delimiter)
    passageLen = len(passage_words)
    docTF = {}
    for word in set(query_words):   #Find Term Frequency of all query unique words
        docTF[word] = passage_words.count(word)
    commonWords = set(query_words) & set(passage_words)
    tmp_score = []
    for word in commonWords :
        numer = (docTF[word] * (k1+1))   #Numerator part of BM25 Formula
        denom = ((docTF[word]) + k1*(1 - b + b*passageLen/avgDocLength)) #Denominator part of BM25 Formula
        if(word in docIDFDict) :
            tmp_score.append(docIDFDict[word] * numer / denom)

    score = sum(tmp_score)
    return score

#The following line reads each line from testfile and extracts query, passage and calculates BM25 similarity scores and writes the output in outputfile
def RunBM25OnEvaluationSet(testfile,outputfile):

    lno=0
    tempscores=[]  #This will store scores of 10 query,passage pairs as they belong to same query
    labels = []
    passages = []
    passage_id = []
    label1 = -1
    f = open(testfile,"r",encoding="utf-8")
    fw = open(outputfile,"w",encoding="utf-8")
    for line in f:
        # re.sub( '\s+', ' ', mystring ).strip()
        # s.translate(None, string.punctuation)
        # line = line.translate(None, string.punctuation)
        tokens = line.strip().lower().split("\t")
        Query = re.sub('\s+', ' ', tokens[1].translate(str.maketrans('','',string.punctuation+'0123456789'))).strip()
        Passage = re.sub('\s+', ' ', tokens[2].translate(str.maketrans('','',string.punctuation+'0123456789'))).strip()
        label = tokens[3]
        pid = tokens[4]
        if int(label) == 1:
            label1 = lno%10
        score = GetBM25Score(Query,Passage)
        tempscores.append(score)
        labels.append(label)
        passages.append(Passage)
        passage_id.append(pid)
        lno+=1
        if(lno%10==0):
            top4 = sorted(range(len(tempscores)), key=lambda i: tempscores[i])[-4:]
            is_label_1_present = False
            qid = tokens[0]
            for index in top4:
                if int(labels[index]) == 1:
                    is_label_1_present = True
                    break
            if is_label_1_present:
                for index in top4:
                    fw.write(qid+"\t"+Query+"\t"+passages[index]+"\t"+labels[index]+"\t"+passage_id[index]+"\n")
            else:
                for index in top4[1:]:
                    fw.write(qid+"\t"+Query+"\t"+passages[index]+"\t"+labels[index]+"\t"+passage_id[index]+"\n")
                fw.write(qid+"\t"+Query+"\t"+passages[label1]+"\t"+labels[label1]+"\t"+passage_id[label1]+"\n")
            tempscores=[]
            labels = []
            passages = []
            passage_id = []
            lno = 0
        if(lno%5000==0):
            print(lno)
    print(lno)
    f.close()
    fw.close()


if __name__ == '__main__' :

    inputFileName = "data.tsv"   # This file should be in the following format : queryid \t query \t passage \t label \t passageid
    corpusFileName = "corpus.tsv"
    outputFileName = "binarydata.tsv"

    GetCorpus(inputFileName,corpusFileName)    # Gets all the passages(docs) and stores in corpusFile. you can comment this line if corpus file is already generated
    print("Corpus File is created.")
    IDF_Generator(corpusFileName)   # Calculates IDF scores.
    print("IDF Dictionary Generated.")
    RunBM25OnEvaluationSet(inputFileName,outputFileName)
    print("Submission file created. ")
