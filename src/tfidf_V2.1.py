#importing necessary libraries
import pandas as pd
import matplotlib as plt
from pathlib import Path
import math
#importing nlp libraries
import string 
import spacy 
from spacy import displacy 
from spacy.lang.en.stop_words import STOP_WORDS
from nltk.tokenize import word_tokenize
from gensim.parsing.preprocessing import STOPWORDS
import nltk
nltk.download('stopwords')
from nltk.tokenize import word_tokenize
#importing libraries for TF-IDF calculations, NLP libraries 
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity


#assigning variables (file1 file2 ... file5) for all text files and putting them into the list "file_lst"
file1 = "Chapter1.txt"
file2 = "Chapter2.txt"
file3 = "Chapter3.txt"
file4 = "Chapter4.txt"
file5 = "Chapter5.txt"

file_lst = [file1,file2,file3,file4,file5]


#appending all the texts into one file named AllChapters.txt
#call this function only if you need everyfile to be combined
def allChapter_combiner():
    with open('AllChapters.txt', 'w',encoding="utf8") as outfile:
        for fname in file_lst:
            with open(fname,encoding="utf8") as infile:
                for line in infile:
                    outfile.write(line)
    with open('AllChapters.txt','r',encoding="utf8") as allfltxt:
               return allfltxt.read()


#in case if you want to use all chapter files text all in one use this variable 
all_chap_lines = allChapter_combiner()



#opening the AllChapters.txt file and assigning the texts to lines variable
def DataReader(fl):
    with open(fl,'r',encoding="utf8") as finalfl:
            temp = finalfl.read()
            lines = temp
    return lines

         


#reading all documents and individually 
f1 = DataReader(file1)
f2 = DataReader(file2)
f3 = DataReader(file3)
f4 = DataReader(file4)
f5 = DataReader(file5)




#assigning variables to and initiating nlp libraries to english language
nlp = spacy.load('en_core_web_sm')
stopwords = STOPWORDS
pun = string.punctuation



#as if we see the text files(documents) it's very messy, we need to clean the data 
#writing several functions for cleaning the data 


def text_cleaner(contents):
    global fnl_filtered_lst,tokens_sw,lower_tokens,numbers,num,pun,punctuation,text_tokens,lines,texts,temp2,temp4,txt_pun
    text_tokens = word_tokenize(contents)
    lower_tokens = []
    for text in text_tokens:
        temp2 = text.lower()
        lower_tokens.append(temp2)
    punctuation = []
    for i in pun:
        punctuation.append(i)
    numbers = list(range(0,101))
    pun_append()
    num = []
    for j in numbers:
        temp3 = str(j)
        num.append(temp3)
    tokens_sw = []
    for word in lower_tokens:
        if word not in stopwords:
            tokens_sw.append(word)
    filtered_lst = []
    for text in tokens_sw:
        if text not in num:
            filtered_lst.append(text)
    txt_pun = []
    for c in filtered_lst:
        if c not in punctuation:
            txt_pun.append(c)
    return txt_pun
#the documents has some special punctuations that are not included in default pun in the nlp libraries
def pun_append():
    punctuation.append("'")
    punctuation.append("’")
    punctuation.append("“")
    punctuation.append("”")
    punctuation.append("...")
    punctuation.append("-")
    punctuation.append("__")
    punctuation.append("_")
    punctuation.append("-")
    punctuation.append("—")
    
#furthermore we are writing a function to lammatize and clean properly
def txt_data_clean(sentence):
    doc = nlp(sentence)
    
    
    tokens = []
    for token in doc:
        
        if token.lemma_ !="-PRON-": 
            temp = token.lemma_.lower().strip()
        else:  
            temp = token.lower_
        tokens.append(temp) 
    clean_tok =[]
    for token in tokens:
        if token not in stopwords and token not in  punctuation: 
            clean_tok.append(token)
    return clean_tok 
#calling a function to get a meaningful list in return
def final_cleaner(lst):
    fnl_filtered_lst = []
    for y in lst:
        temp8 = txt_data_clean(y)
        for q in temp8:
            fnl_filtered_lst.append(q)
    return fnl_filtered_lst
    


#now extrating clean texts form all douments 

#chapter1
clntxt1 = text_cleaner(f1)
pun_append()
document1 = final_cleaner(clntxt1)

#chapter2
clntxt2 = text_cleaner(f2)
document2 = final_cleaner(clntxt2)

#chapter3
clntxt3 = text_cleaner(f3)
document3 = final_cleaner(clntxt3)

#chapter4
clntxt4 = text_cleaner(f4)
document4 = final_cleaner(clntxt4)

#chapter5
clntxt5 = text_cleaner(f5)
document5 = final_cleaner(clntxt5)




AllDocumentsList = [document1, document2, document3, document4, document5]


#EVALUATING ALL UNIQUE WORDS IN ALL THE DOCUMENTS 
uniqueWords = set().union(*AllDocumentsList)



numOfWords1 = dict.fromkeys(uniqueWords, 0)
for word in document1:
    numOfWords1[word] += 1

numOfWords2= dict.fromkeys(uniqueWords, 0)
for word in document2:
    numOfWords2[word] += 1 

numOfWords3 = dict.fromkeys(uniqueWords, 0)
for word in document3:
    numOfWords3[word] += 1 

numOfWords4 = dict.fromkeys(uniqueWords, 0)    
for word in document4:
    numOfWords4[word] += 1 

numOfWords5 = dict.fromkeys(uniqueWords, 0)
for word in document5:
    numOfWords5[word] += 1 
    
    


#These dictonaries tells how many times each unique words appreadred in the paritucular document
num_of_wrd_lst = [numOfWords1,numOfWords2,numOfWords3,numOfWords4,numOfWords5]
for var in num_of_wrd_lst:
    print(pd.DataFrame(var.items(), columns=['terms','frequency']))


"""Term Frequency also known as TF, TF is the number of times a word appears
in a document divided by the total number of words in the document. Every document has its own term frequency"""

def TF_calc(word_dict, document):
    tf_Dict = {}
    document_count = len(document)
    for word, count in word_dict.items():
        tf_Dict[word] = count / float(document_count)
    return tf_Dict


#term frequency calculation aka TF calculation
tf1 = TF_calc(numOfWords1, document1)
tf2 = TF_calc(numOfWords2, document2)
tf3 = TF_calc(numOfWords3, document3)
tf4 = TF_calc(numOfWords4, document4)
tf5 = TF_calc(numOfWords5, document5)



#showing the term frequencies 
tfList = [tf1, tf2, tf3, tf4, tf5]
for freq in tfList:
    tfdf = pd.DataFrame(tf1.items(), columns=['terms','term frequency'])
    print(tfdf)

#Inverse Data/Document Frequency(IDF)
"""the log of the number of documents divided by the number of documents that
contains the word w. Inverse data frequency determines the weight of rare words across
all documents in the corpus"""

def IDF_calc(documents):
    n = len(documents)
    
    idfDict = dict.fromkeys(documents[0].keys(),0)
    for document in documents:
        for word, val in document.items():
            if val > 0:
                idfDict[word] += 1
    
    for word, val in idfDict.items():
        idfDict[word] = math.log(n / float(val))
    return idfDict


#claculating the inverse document frequencies of all documents
idfs = IDF_calc([numOfWords1, numOfWords2, numOfWords3, numOfWords4, numOfWords5])

print(pd.DataFrame(idfs.items(), columns=['terms','IDF']))



#in Conclusion we need to calculate the tf-idf which is nothing but the multiplicaiton of TF and IDF
def TFIDF_calc(tfBOW, idfs):
    tfidf = {}
    for word, val in tfBOW.items():
        tfidf[word] = val * idfs[word]
    return tfidf
    


#calculating TFIDF for all documents
tfidf1 = TFIDF_calc(tf1, idfs)
tfidf2 = TFIDF_calc(tf2, idfs)
tfidf3 = TFIDF_calc(tf3, idfs)
tfidf4 = TFIDF_calc(tf4, idfs)
tfidf5 = TFIDF_calc(tf5, idfs)




#framing them into a dataframe for better understaing and visualization
AllTfidfs = [tfidf1, tfidf2, tfidf3, tfidf4, tfidf5]
#framing the dataframe of TF-IDF-VECTORIZER for reference 
for tfid in AllTfidfs:
    df = pd.DataFrame(tfid.items(),columns=['terms','TF_IDF'])
    print(df)


#Saving all list inot a string to do calculation in TfidfVec
StrDoc1 = ' '.join(document1)
StrDoc2 = ' '.join(document2)
StrDoc3 = ' '.join(document3)
StrDoc4 = ' '.join(document4)
StrDoc5 = ' '.join(document5)



#saving those filtered strings in a text document for future calculation
strList = [StrDoc1,StrDoc2,StrDoc3,StrDoc4,StrDoc5]

text_file1 = open("txt/StrDoc1.txt","w")
strn1 = text_file1.write(StrDoc1)
text_file1.close()

text_file2 = open("txt/StrDoc2.txt","w")
strn2 = text_file2.write(StrDoc2)
text_file2.close()

text_file3 = open("txt/StrDoc3.txt","w")
strn3 = text_file3.write(StrDoc3)
text_file3.close()

text_file4 = open("txt/StrDoc4.txt","w")
strn4 = text_file4.write(StrDoc4)
text_file4.close()

text_file5 = open("txt/StrDoc5.txt","w")
strn5 = text_file5.write(StrDoc5)
text_file5.close()



"""even though we've calculated TF-IDF without any library we are going to use 
TfidfVectorized and compare both results"""

vectorizer = TfidfVectorizer()
vectors = vectorizer.fit_transform([StrDoc1, StrDoc2, StrDoc3, StrDoc4, StrDoc5])
feature_names = vectorizer.get_feature_names()
dense = vectors.todense()
denselist = dense.tolist()
df1 = pd.DataFrame(denselist, columns=feature_names)


#framing the dataframe of TF-IDF-VECTORIZER for reference 
print(df1)



#create a txt folder
Path("./txt").mkdir(parents=True, exist_ok=True)
all_txt_files =[]
for file in Path("txt").rglob("*.txt"):
     all_txt_files.append(file.parent / file.name)
# counts the length of the list
n_files = len(all_txt_files)
print(n_files)



all_txt_files.sort()



all_docs = []
for txt_file in all_txt_files:
    with open(txt_file) as f:
        txt_file_as_string = f.read()
    all_docs.append(txt_file_as_string)


vectorizer = TfidfVectorizer(max_df=.65, min_df=1, stop_words=None, use_idf=True, norm=None)
transformed_documents = vectorizer.fit_transform(all_docs)


transformed_documents_as_array = transformed_documents.toarray()
# use this line of code to verify that the numpy array represents the same number of documents that we have in the file list
len(transformed_documents_as_array)



# make the output folder if it doesn't already exist
Path("./tf_idf_output").mkdir(parents=True, exist_ok=True)

# construct a list of output file paths using the previous list of text files the relative path for tf_idf_output
output_filenames = [str(txt_file).replace(".txt", ".csv").replace("txt/", "tf_idf_output/") for txt_file in all_txt_files]

# loop each item in transformed_documents_as_array, using enumerate to keep track of the current position
for counter, doc in enumerate(transformed_documents_as_array):
    # construct a dataframe
    tf_idf_tuples = list(zip(vectorizer.get_feature_names(), doc))
    one_doc_as_df = pd.DataFrame.from_records(tf_idf_tuples, columns=['term', 'rank']).sort_values(by='rank', ascending=False).reset_index(drop=True)
    #printing the term and it's rank in all document based 
    print(one_doc_as_df)
    # output to a csv using the enumerated value for the filename
    #one_doc_as_df.to_csv(output_filenames[counter])



"""Text Similarity has to determine how the two text documents close to each other in terms of their context or meaning. 
There are various text similarity metric exist such as Cosine similarity, Euclidean distance and Jaccard Similarity. 
All these metrics have their own specification to measure the similarity between two queries."""


#perfoming cosin similarities using count vectorizer function
count_vectorizer = CountVectorizer()
vector_matrix_cosin = count_vectorizer.fit_transform(strList)



tokens = count_vectorizer.get_feature_names()



#convertin everyting into a matrix/array
matrix_cosin_vector = vector_matrix_cosin.toarray()


#writing a function for creating dataframe 
def dataFramer(matrix, tokens):
    DocName = [f'doc_{i+1}' for i, _ in enumerate(matrix)]
    dataframe = pd.DataFrame(data=matrix, index=DocName, columns=tokens)
    return(dataframe)    


#vector representation of document
print(dataFramer(matrix_cosin_vector,tokens))

cosine_similarity_matrix = cosine_similarity(matrix_cosin_vector)

#framing the cosine similarites betweent those 5 documents 
docList = ['doc_1','doc_2','doc_3','doc_4','doc_5']
dataFramer(cosine_similarity_matrix, docList)

