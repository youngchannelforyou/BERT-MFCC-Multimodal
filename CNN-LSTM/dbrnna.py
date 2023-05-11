import numpy as np
from keras.preprocessing import sequence
from keras.models import Model
from keras.layers import (
    Dense,
    Dropout,
    Embedding,
    LSTM,
    GRU,
    BatchNormalization,
    Flatten,
    Input,
    RepeatVector,
    Permute,
    multiply,
    Lambda,
    Activation,
    Conv1D, 
    MaxPooling1D,
)
from keras.layers.merge import concatenate
from tensorflow.keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from keras import backend as K
from dataset import chronological_cv
from sklearn import metrics
from sklearn.metrics import precision_recall_fscore_support
import tensorflow as tf
import csv

# np.random.seed(1337)

def _Recall(y_target, y_pred):
    y_target_yn = K.round(K.clip(y_target, 0, 1)) # 실제값 0(Negative) / 1(Positive)
    y_pred_yn = K.round(K.clip(y_pred, 0, 1)) # 예측값 0(Negative) / 1(Positive)
    
    count_true_positive = K.sum(y_target_yn * y_pred_yn) 
    count_true_positive_false_negative = K.sum(y_target_yn)

    recall = count_true_positive / (count_true_positive_false_negative + K.epsilon())
    
    # print("[recall] count_true_positive = ", count_true_positive, ",  count_true_positive_false_negative = ", count_true_positive_false_negative, ",  K.epsilon() = ", K.epsilon())
    
    return recall

def _Precision(y_target, y_pred):
    #print("y_target: ", y_target)
    #print("y_pred: ", y_pred)
    
    y_pred_yn = K.round(K.clip(y_pred, 0, 1)) # 예측값 0(Negative) / 1(Positive)
    y_target_yn = K.round(K.clip(y_target, 0, 1)) # 실제값 0(Negative) / 1(Positive)
    
    count_true_positive = K.sum(y_target_yn * y_pred_yn) 
    count_true_positive_false_positive = K.sum(y_pred_yn)

    precision = count_true_positive / (count_true_positive_false_positive + K.epsilon())
    
    # print("[precision] count_true_positive = ", count_true_positive, ",  count_true_positive_false_positive = ", count_true_positive_false_positive, ",  K.epsilon() = ", K.epsilon())
    
    return precision

def _F1score(y_target, y_pred):
    recall = _Recall(y_target, y_pred)
    precision = _Precision(y_target, y_pred)
    _f1score = (2 * recall * precision) / (recall + precision+ K.epsilon())
    
    # print("[f1score] _f1score = ", _f1score, "  recall = ", recall, ",  K.precision = ", precision, ",  K.epsilon() = ", K.epsilon())
    
    return _f1score

def _Matthews_correlation(y_true, y_pred):
    y_pred_pos = K.round(K.clip(y_pred, 0, 1))
    y_pred_neg = 1 - y_pred_pos

    y_pos = K.round(K.clip(y_true, 0, 1))
    y_neg = 1 - y_pos
    
    tp = K.sum(y_pos * y_pred_pos)
    tn = K.sum(y_neg * y_pred_neg)

    fp = K.sum(y_neg * y_pred_pos)
    fn = K.sum(y_pos * y_pred_neg)

    numerator = (tp * tn - fp * fn)
    denominator = K.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))

    return numerator / (denominator + K.epsilon())

#Model_CLSTM((_W2V_Sent_Len, _W2V_Size), len(classes), _Model_Type, _LR)
def Model_CLSTM(Input_Shape=None, Num_Output=None, Num_RNN_Unit=512, Num_Dense_Unit=1000, _Model_Type=None, LR=None):
    input_1 = Input(shape=Input_Shape, dtype="float32")
    
    conv1d = Conv1D(filters=128, kernel_size=3, padding='same', activation='relu')(input_1)
    maxpooling1d = MaxPooling1D(pool_size=2)(conv1d)
    
    forwards_1 = LSTM(Num_RNN_Unit, return_sequences=True, dropout=0.2)(input_1) 
    attention_1 = Dense(1, activation="tanh")(forwards_1)
    attention_1 = Flatten()(attention_1)  # squeeze (None,50,1)->(None,50)
    attention_1 = Activation("softmax")(attention_1)
    attention_1 = RepeatVector(Num_RNN_Unit)(attention_1)   #LSTM에서 사용?
    attention_1 = Permute([2, 1])(attention_1)
    attention_1 = multiply([forwards_1, attention_1])
    attention_1 = Lambda(lambda xin: K.sum(xin, axis=1), output_shape=(Num_RNN_Unit,))(attention_1)
    
    last_out_1 = Lambda(lambda xin: xin[:, -1, :])(forwards_1)
    sent_representation_1 = concatenate([last_out_1, attention_1])

    after_dp_forward_5 = BatchNormalization()(sent_representation_1)
    backwards_1 = LSTM(Num_RNN_Unit, return_sequences=True, dropout=0.2, go_backwards=True)(input_1)


    attention_2 = Dense(1, activation="tanh")(backwards_1)
    attention_2 = Flatten()(attention_2)
    attention_2 = Activation("softmax")(attention_2)
    attention_2 = RepeatVector(Num_RNN_Unit)(attention_2)
    attention_2 = Permute([2, 1])(attention_2)
    attention_2 = multiply([backwards_1, attention_2])
    attention_2 = Lambda(lambda xin: K.sum(xin, axis=1), output_shape=(Num_RNN_Unit,))(attention_2)

    last_out_2 = Lambda(lambda xin: xin[:, -1, :])(backwards_1)
    sent_representation_2 = concatenate([last_out_2, attention_2])

    after_dp_backward_5 = BatchNormalization()(sent_representation_2)
    merged = concatenate([after_dp_forward_5, after_dp_backward_5])
    after_merge = Dense(Num_Dense_Unit, activation="relu")(merged)
    after_dp = Dropout(0.4)(after_merge)
    
    output = Dense(Num_Output, activation="softmax")(after_dp)
    model = Model(input_1, output)
    model.compile(loss="categorical_crossentropy", optimizer=Adam(lr=LR), metrics=['accuracy', _F1score, _Precision, _Recall, _Matthews_correlation])

    return model

def Model_CNN(Input_Shape=None, Num_Output=None, LR=None):
    input_1 = Input(shape=Input_Shape, dtype="float32")
    conv1d = Conv1D(filters=128, kernel_size=1, padding='same', activation='relu')(input_1)
    maxpooling1d = MaxPooling1D(pool_size=2)(conv1d)
            
    attention_1 = Dense(1, activation="tanh")(maxpooling1d)
    attention_1 = Flatten()(attention_1)
    attention_1 = Activation("softmax")(attention_1)
    output = Dense(Num_Output, activation="softmax")(attention_1)
    
    model = Model(input_1, output)
    model.compile(loss="categorical_crossentropy", optimizer=Adam(lr=LR), metrics=['accuracy', _F1score, _Precision, _Recall, _Matthews_correlation])
    
    print(model)
    
    return model

def Model_LSTM(Input_Shape=None, Num_Output=None, Num_RNN_Unit=512, Num_Dense_Unit=1000, _Model_Type=None, LR=None):    
    input_1 = Input(shape=Input_Shape, dtype="float32")
    
    forwards_1 = LSTM(Num_RNN_Unit, return_sequences=True, dropout=0.0)(input_1) 
    attention_1 = Flatten()(forwards_1)
    output = Dense(Num_Output, activation="softmax")(attention_1)
    
    model = Model(input_1, output)
    model.compile(loss="categorical_crossentropy", optimizer=Adam(lr=LR), metrics=['accuracy', _F1score, _Precision, _Recall, _Matthews_correlation])

    return model

def topk_accuracy(prediction, y_test, classes, rank_k=1):
    accuracy = []
    sortedIndices = []
    pred_classes = []
    for ll in prediction:
        sortedIndices.append(
            sorted(range(len(ll)), key=lambda ii: ll[ii], reverse=True)
        )
    for k in range(1, rank_k + 1):
        id = 0
        trueNum = 0
        for sortedInd in sortedIndices:
            pred_classes.append(classes[sortedInd[:k]])
            if np.argmax(y_test[id]) in sortedInd[:k]:
                trueNum += 1
            id += 1
        accuracy.append((float(trueNum) / len(prediction)) * 100)

    return accuracy


def Run_Model(_Dataset_Name=None, _Min_Samples_Per_Class=None, _Num_CV=None, _Model_Type="LSTM", _Epochs=1, _W2V_Size=None, _W2V_Sent_Len=None, _Batch_Size=None, _TopK=None, _LR=None):
    slices = chronological_cv(_Dataset_Name, _Min_Samples_Per_Class, _Num_CV, _W2V_Size, _W2V_Sent_Len)
    
    slice_results = {}
    top_rank_k_accuracies = []
    
    My_Result = []
    My_Result.append(['CV', 'Precision', 'Recall', 'F1_score', 'Accuracy', 'Mcc', 'Loss'])
    
    for i, (X_train, y_train, X_test, y_test, classes) in enumerate(slices):   
        if _Model_Type == "CNN":
            model = Model_CNN(Input_Shape = (_W2V_Sent_Len, _W2V_Size), Num_Output = len(classes), LR = _LR)
            
        elif _Model_Type == "LSTM" or _Model_Type == "GRU":
            model = Model_LSTM(Input_Shape = (_W2V_Sent_Len, _W2V_Size), Num_Output = len(classes), _Model_Type = _Model_Type, LR = _LR)
            
        elif _Model_Type == "CLSTM":
            model = Model_CLSTM(Input_Shape = (_W2V_Sent_Len, _W2V_Size), Num_Output = len(classes), _Model_Type = _Model_Type, LR = _LR)

        #model.summary()
        
        # Train the deep learning model and test using the classifier
        early_stopping = EarlyStopping(monitor="val_loss", patience=3)
        hist = model.fit(X_train, y_train, validation_data=(X_test, y_test), batch_size=_Batch_Size, epochs=_Epochs, callbacks=[early_stopping],)
        
        My_Precision = hist.history['_Precision']
        My_Recall = hist.history['_Recall']
        My_F1score = hist.history['_F1score']
        #My_Accuracy = hist.history['acc']
        My_Mcc = hist.history['_Matthews_correlation']
        My_Loss = hist.history['loss']
        # 'accuracy', _F1score, _Precision, _Recall
        Loss, Accuracy, F1_score, Precision, Recall, Mcc = model.evaluate(X_test, y_test, verbose=1)
        
        prediction = model.predict(X_test)
        accuracy = topk_accuracy(prediction, y_test, classes, rank_k=_TopK)
        
        y_test_arg = np.argmax(prediction, axis=1)
        #Mcc = _Matthews_correlation(y_test, t)
        
        print("\n[" + _Model_Type + "] Precision : ", Precision)
        print("[" + _Model_Type + "] Recall : ", Recall)
        print("[" + _Model_Type + "] F-Measure : ", F1_score)
        print("[" + _Model_Type + "] Accuracy : ", Accuracy)
        #print("[" + _Model_Type + "] Mcc : ", Mcc)
        print("[" + _Model_Type + "] Loss : ", Loss)
        
        My_Result.append([str(i + 1), Precision, Recall, F1_score, Accuracy, Mcc, Loss])
        
        train_result = hist.history
        train_result["test_topk_accuracies"] = accuracy
        slice_results[i + 1] = train_result
        top_rank_k_accuracies.append(accuracy[-1])
    
    output = './result/' + _Dataset_Name + '_' + _Model_Type + '_' + str(_Epochs) + '_' + str(_LR) + '.csv'
    with open(output,'w', newline='') as file:
        write = csv.writer(file)
        for _str_ in My_Result:
            write.writerow(_str_)
    
    print("Top{0} accuracies for all CVs: {1}".format(_TopK, top_rank_k_accuracies))
    print("Average top{0} accuracy: {1}".format(_TopK, sum(top_rank_k_accuracies)/_TopK))
    return slice_results
