import numpy as np
import json, re, nltk, string
from nltk.corpus import wordnet
from gensim.models import Word2Vec
import csv
import random

def clean_word_list(item):
    current_text = item.replace("\r", " ")
    current_text = re.sub(r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+", "", current_text,)

    start_loc = current_text.find("Stack trace:")
    current_text = current_text[:start_loc]
    current_text = re.sub(r"(\w+)0x\w+", "", current_text)
    current_text = current_text.lower()
    current_text_tokens = nltk.word_tokenize(current_text)

    current_text_filter = [word.strip(string.punctuation) for word in current_text_tokens]
    
    current_data = current_text_filter
    current_data = [x for x in current_data if x]

    return current_data

def Run_Preprocess(_Dataset_Name, Config_W2V_MinCNT, Config_W2V_Size, Config_W2V_Window):
    print("Preprocessing {0} dataset: Start".format(_Dataset_Name))
    MyCSV = "./data/{0}.csv".format(_Dataset_Name)

    My_Feature = []
    My_Class = []
    with open(MyCSV) as f:
        reader = csv.reader(f, delimiter=',', quotechar="'")
        
        for row in reader:
            if (len(row) > 1):
                tmp = clean_word_list(row[1])
                My_Feature.append(tmp) #Class, Feature1, ... #Feature1
                My_Class.append(row[0])
                
    print("Preprocessing {0} dataset: Finished".format(_Dataset_Name))
    
    print("Preprocessing {0} Word2Vec model: Start".format(_Dataset_Name))
    wordvec_model = Word2Vec(
        My_Feature,
        min_count = Config_W2V_MinCNT,
        size = Config_W2V_Size,
        window = Config_W2V_Window,
    )

    wordvec_model.save("./data/{0}_word2vec.model".format(_Dataset_Name))
    print("Preprocessing {0} Word2Vec model: Finished".format(_Dataset_Name))
    
    print("Preprocessing {0} Classifier data: Start".format(_Dataset_Name))
    np.save("./data/{0}_Feature.npy".format(_Dataset_Name), My_Feature,) # all_data.npy -> Feature.npy
    np.save("./data/{0}_Class.npy".format(_Dataset_Name), My_Class,) #all_severity -> Class.npy
    print("Preprocessing {0} Classifier data: Finished".format(_Dataset_Name))
