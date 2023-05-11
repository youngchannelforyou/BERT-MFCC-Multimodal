import numpy as np
from gensim.models import Word2Vec
from keras.utils import np_utils
import random

def filter_with_vocabulary(sentences, labels, vocabulary, min_sentence_length=15):
    """Remove all the words that is not present in the vocabulary"""
    updated_sentences = []
    updated_labels = []
    
    for j, item in enumerate(sentences):
        current_train_filter = [word for word in item if word in vocabulary]
        if len(current_train_filter) >= min_sentence_length:
            updated_sentences.append(current_train_filter)
            updated_labels.append(labels[j])

    return updated_sentences, updated_labels


def filter_with_labels(sentences, labels, known_labels):
    """Remove data from test set that is not there in train set"""
    known_labels_unique = set(known_labels)
    labels_unique = set(labels)
    unwanted_labels = list(labels_unique - known_labels_unique)
    updated_sentences = []
    updated_labels = []
    for j in range(len(labels)):
        if labels[j] not in unwanted_labels:
            updated_sentences.append(sentences[j])
            updated_labels.append(labels[j])

    return updated_sentences, updated_labels


def load_data(_Dataset_Name):
    W2V_Model = Word2Vec.load("./data/{0}_word2vec.model".format(_Dataset_Name))
    Features = np.load("./data/{0}_Feature.npy".format(_Dataset_Name), allow_pickle=True,)
    Classes = np.load("./data/{0}_Class.npy".format(_Dataset_Name), allow_pickle=True,)

    return W2V_Model, Features, Classes


def embedding(sentences, labels, unique_labels, wordvec_model, vocabulary, max_sentence_len=50, embed_size_word2vec=200,):
    """ Create the data matrix and labels required for the deep learning model training and softmax classifier"""

    X = np.empty(
        shape=[len(sentences), max_sentence_len, embed_size_word2vec], dtype="float32"
    )
    Y = np.empty(shape=[len(labels), 1], dtype="int32")
    # 1 - start of sentence, # 2 - end of sentence, # 0 - zero padding. Hence, word indices start with 3
    for j, curr_row in enumerate(sentences):
        sequence_cnt = 0
        for item in curr_row:
            if item in vocabulary:
                X[j, sequence_cnt, :] = wordvec_model[item]
                sequence_cnt = sequence_cnt + 1
                if sequence_cnt == max_sentence_len - 1:
                    break
        for k in range(sequence_cnt, max_sentence_len):
            X[j, k, :] = np.zeros((1, embed_size_word2vec))
        Y[j, 0] = unique_labels.index(labels[j])

    return X, Y


def chronological_cv(_Dataset_Name, _Min_Samples_Per_Class, _Num_CV, _W2V_Size, _W2V_Sent_Len):
    wordvec_model, sentences, labels = load_data(_Dataset_Name)

    # chronological cross validation split is performed
    vocabulary = wordvec_model.wv.vocab
    splitLength = len(sentences) // (_Num_CV + 1)
    
    Temp_List = []

    for i in range(len(sentences)): #161433
        Temp_List.append([labels[i], sentences[i]])
    
    random.shuffle(Temp_List)
    
    _sentences = []
    _labels = []
    
    for _str in Temp_List:
        _labels.append(_str[0])
        _sentences.append(_str[1])
        
    labels = np.array(_labels)
    sentences = np.array(_sentences)
    
    for i in range(1, _Num_CV + 1):
        train_x = sentences[: i * splitLength - 1]
        train_y = labels[: i * splitLength - 1]
        test_x = sentences[i * splitLength : (i + 1) * splitLength - 1]
        test_y = labels[i * splitLength : (i + 1) * splitLength - 1]

        # Remove all the words that is not present in the vocabulary
        updated_train_x, updated_train_y = filter_with_vocabulary(train_x, train_y, vocabulary)
        tmp_test_x, final_test_y = filter_with_vocabulary(test_x, test_y, vocabulary)

        # Remove those classes from the test set, for whom the train data is not available.
        updated_test_x, updated_test_y = filter_with_labels(tmp_test_x, final_test_y, updated_train_y)

        # Create the data matrix and labels required for the deep learning model training and softmax classifier
        unique_train_label = list(set(updated_train_y))
        classes = np.array(unique_train_label)
        X_train, Y_train = embedding(
            updated_train_x,
            updated_train_y,
            unique_train_label,
            wordvec_model,
            vocabulary,
        )
        
        X_test, Y_test = embedding(
            updated_test_x,
            updated_test_y,
            unique_train_label,
            wordvec_model,
            vocabulary,
        )


        y_train = np_utils.to_categorical(Y_train, len(unique_train_label))
        y_test = np_utils.to_categorical(Y_test, len(unique_train_label))
        
        yield X_train, y_train, X_test, y_test, classes