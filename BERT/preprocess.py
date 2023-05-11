import os
import argparse
import subprocess
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from imblearn.under_sampling import RandomUnderSampler
from pydub import AudioSegment
from pathlib import Path
import chardet
import numpy as np
import re
import math
from sklearn.metrics.pairwise import cosine_similarity
#이 프리프로세싱 코드는 typeA코드로 대표데이터 not under sampling
#프리프로세싱 코드를 돌리기 전에 set.py를 돌려주세요

def split_df(df, train_size):
    df['person_idx'] = range(1, len(df) + 1)
    
    train_df, temp_df = train_test_split(
            df, test_size=1 - train_size#, random_state=42
        )
    val_df, test_df = train_test_split(
            temp_df, test_size=.5#, random_state=42
        )
    
    print("데이터 분할 완료")

    return train_df, val_df, test_df


def extract_audio(path, df):
    
    columns = ['person_idx', 'audio', 'sentence', 'emotion']
    new_df = pd.DataFrame(columns=columns)
    
    personlist = []
    audiolist = []
    sentencelist = []
    emotionlist = []
    EDAlist = []
    TEMPlist = []
    IBIlist = []
    
    for index, row in df.iterrows():
        
        person_idx = row['person_idx']
        sentence = row['sentence']
        emotion = row['emotion']
        EDA = row['EDA']
        TEMP = row['TEMP']
        IBI = row['IBI']
    
        # convert wav file to 1 channel
        audio = AudioSegment.from_wav(row['audio_path'])
        audio = audio.set_channels(1)
        audio = audio.get_array_of_samples()

        # save in dataframe
        personlist.append(person_idx)
        audiolist.append(audio)
        sentencelist.append(sentence)
        emotionlist.append(emotion)
        EDAlist.append(EDA)
        TEMPlist.append(TEMP)
        IBIlist.append(IBI)
        
    new_df['person_idx']=personlist
    new_df['audio']=audiolist
    new_df['sentence']=sentencelist
    new_df['emotion']=emotionlist
    new_df['EDA']=EDAlist
    new_df['TEMP']=TEMPlist
    new_df['IBI']=IBIlist


    new_df['person_idx'] = new_df['person_idx'].astype('object')

    new_df.dropna(subset=['audio'], inplace=True)
    
    print("오디오 데이터 로드 완료")
    
    #from imblearn.over_sampling import RandomOverSampler

    # # 샘플링
    # X_train = new_df.drop('emotion', axis=1)
    # y_train = new_df['emotion']
    # # RandomUnderSampler 객체 생성
    # rus = RandomOverSampler(random_state=42)

    # #  샘플링 수행
    # X_train_res, y_train_res = rus.fit_resample(X_train, y_train)

    # new_df = pd.concat([X_train_res, y_train_res], axis=1)

    return new_df#.sort_values('sentence')

def make_annotationdf(path,sess):
    new_df = pd.DataFrame()
    files = []
    
    for dirpath, dirnames, filenames in os.walk(os.path.join(path, "annotation")):
            for filename in filenames:
                if filename.endswith(f'{sess:02}'+"_eval.csv"):
                    files.append(os.path.join(dirpath, filename))

    print("annotation 데이터 프레임 주소 수집 완료")

    
    df_annotation = pd.concat((pd.read_csv(f, skiprows=1) for f in files))
    print("annotation 데이터 프레임 생성 완료")


    
    new_df['person_idx'] = df_annotation["Segment ID"]
    new_df['emotion'] = df_annotation["Emotion"]


    new_df = new_df.replace(['', ' ', 'N/A', 'NaN', 'nan', 'Nan'], np.nan)
    new_df = new_df.dropna().reset_index(drop=True)

    audio = []
    text = []
    for index, row in new_df.iterrows():
        if(row["person_idx"] != 'nan'):
            temp= "Session" + str(row["person_idx"])[4:6]
            src = os.path.join(path, temp)

            temp = str(row["person_idx"]) + ".wav"
            audio_src = os.path.join(src, temp)
            audio.append(audio_src)
        
            temp = str(row["person_idx"]) + ".txt"
            txt_src = os.path.join(src, temp)
            text.append(txt_src)
            
    print("annotation 데이터 프레임 데이터 주소 저장 완료")

    new_df["audio_path"] = audio
    new_df['text_path'] = text

    sentences = []
    runtype = ""
    
    def cut_stranger(text):
        text = text.replace("n/","").replace("b/","",).replace("l/","").replace("c/","").replace("N/","").replace("u/","").replace("+","").replace("*","")
        return text
    
    for index, row in new_df.iterrows():
        file_contents = ""
        with open(row['text_path'], "rb") as file:
            result = chardet.detect(file.read())
            runtype = result['encoding']
        
        if(runtype == 'EUC-KR'):
            with open(row['text_path'], "r", encoding='EUC-KR') as file:
                file_contents = (file.read())#데이터지우기cut_stranger
                sentences.append(file_contents)
        
        elif(runtype == 'CP949'):
            with open(row['text_path'], "r", encoding='CP949') as file:
                file_contents = (file.read())#데이터지우기cut_stranger
                sentences.append(file_contents)
        if(file_contents == ""):
            sentences.append(np.nan)
    
    print("annotation 데이터 프레임 텍스트 데이터 로드 완료")
    
    new_df['sentence'] = sentences 

    
    new_df = new_df.dropna(subset=['person_idx']).reset_index(drop=True)
    new_df = new_df.dropna(subset=['emotion']).reset_index(drop=True)
    new_df = new_df.dropna(subset=['audio_path']).reset_index(drop=True)
    new_df = new_df.dropna(subset=['sentence']).reset_index(drop=True)

     # 막코드
    for index, row in new_df.iterrows():

        # convert wav file to 1 channel
        audio = AudioSegment.from_wav(row['audio_path'])
        audio = audio.set_channels(1)
        audio = audio.get_array_of_samples()

        left, right = None, None
        for idx in range(len(audio)):
            if np.float32(0) != np.float32(audio[idx]):
                left = idx
                break
        for idx in reversed(range(len(audio))):
            if np.float32(0) != np.float32(audio[idx]):
                right = idx
                break
        
        if (left == None or right == None):
            row['audio_path'] = np.nan

    new_df = new_df.dropna(subset=['audio_path']).reset_index(drop=True)
    
    print("annotation 데이터 프레임 오디오 데이터 검사 완료")
   
    #막코드
    return new_df

def do_sampling(new_df):
    # 샘플링
    X_train = new_df.drop('emotion', axis=1)
    y_train = new_df['emotion']
    # RandomUnderSampler 객체 생성
    rus = RandomUnderSampler(random_state=42)

    #  샘플링 수행
    X_train_res, y_train_res = rus.fit_resample(X_train, y_train)

    new_df = pd.concat([X_train_res, y_train_res], axis=1)
    
    print("샘플링 완료")
    
    return new_df


def do_preprotestdev(trn_df,sess):
    for index, row in trn_df.iterrows():
        #대표 데이터 쪼개기
        if ';' in row["emotion"]:
            for i in row["emotion"].split(';'):
                if (row["emotion"].split(';')[0]==i):
                    trn_df.at[index, 'emotion'] = i
                    continue
                new_row = row.copy()
                new_row['emotion'] = i
                trn_df = trn_df.append(new_row)
    for index, row in trn_df.iterrows():
        if 'disgust' in row['emotion']:
            trn_df.at[index, 'emotion'] = 'disqust'

    print("annotation 데이터 프레임 Emotion 전처리 완료")
    
    
    print(trn_df["emotion"].unique())

    filesEDA = []
    filesIBI = []
    filesTEMP = []

    
    if ((sess == 12)|(sess == 17)|(sess == 5)):
        sess-=1
        
    for dirpath, dirnames, filenames in os.walk(os.path.join("./data/EDA/",f"Session{sess:02}")):
            for filename in filenames:
                if filename.endswith(".csv"):
                    filesEDA.append(os.path.join(dirpath, filename))

    for dirpath, dirnames, filenames in os.walk(os.path.join("./data/IBI/",f"Session{sess:02}")):
            for filename in filenames:
                if filename.endswith(".csv"):
                    filesIBI.append(os.path.join(dirpath, filename))

    for dirpath, dirnames, filenames in os.walk(os.path.join("./data/TEMP/",f"Session{sess:02}")):
            for filename in filenames:
                if filename.endswith(".csv"):  
                    filesTEMP.append(os.path.join(dirpath, filename))

    print("annotation 데이터 프레임 주소 수집 완료")

    df_annotationEDA = []
    for f in filesEDA:
        try:
            csvf = pd.read_csv(f, skiprows=1)
            csvf = csvf.dropna()#[csvf['Unnamed: 2'].str.contains('Sess')]

            for index, row in csvf.iterrows():
                newinfo = []
                for keys in row.keys():
                    newinfo.append(row.loc[keys])
                df_annotationEDA.append(newinfo)#, error_bad_lines=False
        except:
            print(f+"파일이 비어있음")

    df_annotationIBI = []
    for f in filesIBI:
        try:
            csvf = pd.read_csv(f, skiprows=1)
            csvf = csvf.dropna()#[csvf['Unnamed: 2'].str.contains('Sess')]

            for index, row in csvf.iterrows():
                newinfo = []
                for keys in row.keys():
                    newinfo.append(row.loc[keys])
                df_annotationIBI.append((newinfo))#, error_bad_lines=False
        except:
            print(f+"파일이 비어있음")

    df_annotationTEMP = []
    for f in filesTEMP:
        try:
            csvf = pd.read_csv(f, skiprows=1)
            csvf = csvf.dropna()#[csvf['Unnamed: 2'].str.contains('Sess')]

            for index, row in csvf.iterrows():
                newinfo = []
                for keys in row.keys():
                    newinfo.append(row.loc[keys])
                df_annotationTEMP.append(newinfo)#, error_bad_lines=False
        except:
            print(f+"파일이 비어있음")

    # df_annotationEDA = pd.concat((pd.read_csv(f, skiprows=1) for f in filesEDA))
    # df_annotationIBI = pd.concat((pd.read_csv(f, skiprows=1) for f in filesIBI))
    # df_annotationTEMP = pd.concat((pd.read_csv(f, skiprows=1) for f in filesTEMP))
    
    df_annotationEDA = pd.DataFrame(df_annotationEDA)
    df_annotationTEMP = pd.DataFrame(df_annotationTEMP)
    df_annotationIBI = pd.DataFrame(df_annotationIBI)
    
    EDAeve = df_annotationEDA[0].mean()
    print(EDAeve)
    TEMPeve = df_annotationTEMP[0].mean()
    print(TEMPeve)
    try:
        IBIeve = df_annotationIBI[1].mean()
    except:
        #df_annotationIBI[1] = df_annotationIBI[1].select_dtypes(include=['float']).apply(pd.to_numeric, errors='coerce')
        df_annotationIBI[1] = df_annotationIBI[1].astype(float)
        df_annotationIBI = df_annotationIBI.dropna()
        IBIeve = df_annotationIBI[1].mean()
    print(IBIeve)
    df_annotationEDA[0] = df_annotationEDA[0].fillna(EDAeve, inplace=False)#.fillna(df_annotationEDA.mean())
    df_annotationTEMP[0] = df_annotationTEMP[0].fillna(TEMPeve, inplace=False)#.fillna(df_annotationTEMP.mean())
    df_annotationIBI[1] = df_annotationIBI[1].fillna(float(IBIeve), inplace=False)#.fillna(df_annotationIBI.mean())

    print("annotation 데이터 프레임 생성 완료")

    EDA=[]
    TEMP=[]
    IBI=[]
    for index, row in trn_df.iterrows():
        idx = row["audio_path"]
        idx = idx.split("/")
        idx = idx[3].split(".wav")[0]
        #print(idx)
        #print(df_annotationIBI)#[df_annotationEDA[2]])#.str.contains(idx)])
        #print(df_annotationEDA[df_annotationEDA[2].str.contains(idx)])#[0].mean())
        try:
            EDA.append(df_annotationEDA[df_annotationEDA[2].str.contains(idx)][0].mean())
        except:
            EDA.append(0)

        try:
            TEMP.append(df_annotationTEMP[df_annotationTEMP[2].str.contains(idx)][0].mean())
        except:
            TEMP.append(0)
            
        try:
            IBI.append(df_annotationIBI[df_annotationIBI[3].str.contains(idx)][1].mean())
        except:
            IBI.append(0)
        if EDA[len(EDA)-1] is None:
            trn_df.at[index, 'sentence'] += f'{EDAeve:0.3f} ' 
        else:
            trn_df.at[index, 'sentence'] += f'{EDA[len(EDA)-1]:0.3f} '
        if TEMP[len(TEMP)-1] is None:
            trn_df.at[index, 'sentence'] += f'{TEMPeve:0.3f} '
        else:
            trn_df.at[index, 'sentence'] += f'{TEMP[len(TEMP)-1]:0.3f} '
        if IBI[len(IBI)-1] is None:
            trn_df.at[index, 'sentence'] += f'{IBIeve:0.3f} '
        else:
            trn_df.at[index, 'sentence'] += f'{IBI[len(IBI)-1]:0.3f} '
            
    columns = ['person_idx', 'audio_path', 'sentence', 'emotion','TEMP','IBI','EDA']
    trans_trn_df = pd.DataFrame(columns=columns)
    
    trans_trn_df["person_idx"] = trn_df["person_idx"]
    trans_trn_df["audio_path"] = trn_df["audio_path"]
    trans_trn_df["emotion"] = trn_df["emotion"]
    trans_trn_df["sentence"] = trn_df["sentence"]
    trans_trn_df["TEMP"] = TEMP
    trans_trn_df["IBI"] = IBI
    trans_trn_df["EDA"] = EDA

    trans_trn_df["TEMP"] = trans_trn_df["TEMP"].fillna(TEMPeve, inplace=False)
    trans_trn_df["IBI"] = trans_trn_df["IBI"].fillna(IBIeve, inplace=False)
    trans_trn_df["EDA"] = trans_trn_df["EDA"].fillna(EDAeve, inplace=False)
    
    trans_trn_df
    
    return trans_trn_df

def calculate_cosine_similarity(neutral_text, other_texts):
    # CountVectorizer를 사용하여 텍스트 데이터를 벡터화
    from sklearn.feature_extraction.text import CountVectorizer
    vectorizer = CountVectorizer()
    vectorizer.fit_transform([neutral_text] + other_texts)
    vectors = vectorizer.transform([neutral_text] + other_texts).toarray()
    
    # 코사인 유사도 계산
    cosine_similarities = cosine_similarity(vectors)
    
    return cosine_similarities[0, 1:]

def do_precos(trn_df):
    neutral_df = trn_df[trn_df['emotion'] == 'neutral']
    happy_df = trn_df[trn_df['emotion'] == 'happy']
    angry_df = trn_df[trn_df['emotion'] == 'angry']
    surprise_df = trn_df[trn_df['emotion'] == 'surprise']
    disqust_df = trn_df[trn_df['emotion'] == 'disqust']
    sad_df = trn_df[trn_df['emotion'] == 'sad']
    fear_df = trn_df[trn_df['emotion'] == 'fear']
    
    df_list = { "happy" : happy_df,
              "angry" : angry_df,
              "surprise" : surprise_df,
              "disqust" : disqust_df,
              "sad" : sad_df,
              "fear" : fear_df}
    
    pre = len(happy_df["emotion"])
    
    RET_df = pd.DataFrame()
    RET_df = neutral_df.copy()
    
    for emotion_label, emotion_df in tqdm(df_list.items()):
        
        for text in emotion_df['sentence']:
            # 해당 감정 레이블을 가진 데이터의 텍스트 데이터와 neutral 데이터의 코사인 유사도 계산
            similarities = calculate_cosine_similarity(text, neutral_df['sentence'].tolist())
            # 유사도가 70 이상인 행 필터링하여 새로운 데이터프레임에 추가
            similar_indices = np.where(similarities >= 0.6)[0]
            similar_rows = neutral_df.iloc[similar_indices].copy()
            similar_rows['emotion'] = emotion_label

            RET_df = RET_df.reset_index(drop=True)

            for i, row_forrem in RET_df.iterrows():
                if len(similar_rows['sentence'].values) != 0:
                    if similar_rows['sentence'].values[0] in row_forrem['sentence']:
                        RET_df.at[i, 'sentence'] = None
                        RET_df = RET_df.dropna()
                    
            emotion_df = emotion_df.append(similar_rows)

        RET_df = RET_df.append(emotion_df)
        
    print(pre ," ",len(happy_df["emotion"]))
    
    
    RET_df = RET_df.reset_index(drop=True)

    return RET_df

def main(args):
    for sess in range(1,41):
        data_df = make_annotationdf(args.raw_path,sess) 

        # train-dev-test split
        trn1_df, dev1_df, tst1_df = split_df(data_df, args.train_size)

        trn_df = do_preprotestdev(trn1_df,sess)
        
        #trn_df = do_preprotrian(trn1_df)
        #trn_df = do_preprotestdev(trn1_df)
        
        dev_df = do_preprotestdev(dev1_df,sess)
        tst_df = do_preprotestdev(tst1_df,sess)
        #trn_df = do_precos(trn_df)
        print(tst_df['sentence'])
        # add audio features
        for df, split in zip([trn_df, dev_df, tst_df], ['train', 'dev', 'test']):
            df = extract_audio(args.raw_path, df)

            df.to_pickle(os.path.join(args.save_path, f'{split}_{sess:02}.pkl'))
            print(f"saved {split}_{sess:02}.pkl in {args.save_path}")
            
        print(trn_df['emotion'].value_counts())
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw_path', type=str, default='./data')
    parser.add_argument('--script_path', type=str, default='./data/pkls/Script.txt')
    parser.add_argument('--save_path', type=str, default='./data/pkls')
    parser.add_argument('--train_size', type=float, default=.8)
    args_ = parser.parse_args()
    main(args_)