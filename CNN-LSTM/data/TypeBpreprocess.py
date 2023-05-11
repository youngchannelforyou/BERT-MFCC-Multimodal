import os
import argparse
import subprocess
import pandas as pd
from tqdm import tqdm
from dataset import LABEL_DICT
from sklearn.model_selection import train_test_split
from imblearn.under_sampling import RandomUnderSampler
from pydub import AudioSegment
from pathlib import Path
import chardet
import numpy as np

#TEST를 위한 데이터 크기 조절 0은 전체 데이터 1은 sess01 하나
DBUG = 0

def split_df(df, train_size):
    df['person_idx'] = range(1, len(df) + 1)
    
    train_df, temp_df = train_test_split(
            df, test_size=1 - train_size, random_state=42
        )
    val_df, test_df = train_test_split(
            temp_df, test_size=.5, random_state=42
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

    for index, row in df.iterrows():
        
        person_idx = row['person_idx']
        sentence = row['sentence']
        emotion = row['emotion']
    
        # convert wav file to 1 channel
        audio = AudioSegment.from_wav(row['audio_path'])
        audio = audio.set_channels(1)
        audio = audio.get_array_of_samples()

        # save in dataframe
        personlist.append(person_idx)
        audiolist.append(audio)
        sentencelist.append(sentence)
        emotionlist.append(emotion)

    new_df['person_idx']=personlist
    new_df['audio']=audiolist
    new_df['sentence']=sentencelist
    new_df['emotion']=emotionlist


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

def make_annotationdf(path):
    new_df = pd.DataFrame()
    files = []

    if DBUG == 0:
        for dirpath, dirnames, filenames in os.walk(os.path.join(path, "annotation")):
                for filename in filenames:
                    if filename.endswith(".csv"):
                        files.append(os.path.join(dirpath, filename))
    else :
        for dirpath, dirnames, filenames in os.walk(os.path.join(path, "annotation")):
                for filename in filenames:
                    if filename.endswith("01_eval.csv"):
                        files.append(os.path.join(dirpath, filename))
                    
    print("annotation 데이터 프레임 주소 수집 완료")

    
    df_annotation = pd.concat((pd.read_csv(f, skiprows=1) for f in files))
    print("annotation 데이터 프레임 생성 완료")


    new_df['person_idx'] = df_annotation[" .1"]
    new_df['emotion'] = df_annotation["Emotion"]
    new_df['Emotion.1'] = df_annotation["Emotion.1"]
    new_df['Emotion.2'] = df_annotation["Emotion.2"]
    new_df['Emotion.3'] = df_annotation["Emotion.3"]
    new_df['Emotion.4'] = df_annotation["Emotion.4"]
    new_df['Emotion.5'] = df_annotation["Emotion.5"]
    new_df['Emotion.6'] = df_annotation["Emotion.6"]
    new_df['Emotion.7'] = df_annotation["Emotion.7"]
    new_df['Emotion.8'] = df_annotation["Emotion.8"]
    new_df['Emotion.9'] = df_annotation["Emotion.9"]
    new_df['Emotion.10'] = df_annotation["Emotion.10"]
    

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

def do_preprotrian(trn_df):
    for index, row in trn_df.iterrows():
        #첫 번째 평가자 를 대표로
        trn_df.at[index, 'emotion'] = row['Emotion.1']
        #나머지 평가자들의 평가를 추가
        for i in range(2,11):
            new_row = row.copy()
            new_row['emotion'] = row['Emotion.'+str(i)]
            trn_df = trn_df.append(new_row)
    for index, row in trn_df.iterrows():
        if 'disgust' in row['emotion']:
            trn_df.at[index, 'emotion'] = 'disqust'
    print("annotation 데이터 프레임 Emotion 전처리 완료")
    
    
    print(trn_df["emotion"].unique())

    columns = ['person_idx', 'audio_path', 'sentence', 'emotion']
    trans_trn_df = pd.DataFrame(columns=columns)

    trans_trn_df["person_idx"] = trn_df["person_idx"]
    trans_trn_df["emotion"] = trn_df["emotion"]
    trans_trn_df["audio_path"] = trn_df["audio_path"]
    trans_trn_df["sentence"] = trn_df["sentence"]
    
    return trans_trn_df

def do_preprotestdev(trn_df):
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

    columns = ['person_idx', 'audio_path', 'sentence', 'emotion']
    trans_trn_df = pd.DataFrame(columns=columns)

    trans_trn_df["person_idx"] = trn_df["person_idx"]
    trans_trn_df["audio_path"] = trn_df["audio_path"]
    trans_trn_df["emotion"] = trn_df["emotion"]
    trans_trn_df["sentence"] = trn_df["sentence"]
    
    return trans_trn_df

def main(args):
    data_df = make_annotationdf(args.raw_path)

    # train-dev-test split
    trn1_df, dev1_df, tst1_df = split_df(data_df, args.train_size)
        
    trn_df = do_preprotrian(trn1_df)
    trn_df = do_sampling(trn_df)

    #trn_df = do_preprotestdev(trn1_df)
    dev_df = do_preprotestdev(dev1_df)
    tst_df = do_preprotestdev(tst1_df)

    # add audio features
    for df, split in zip([trn_df, dev_df, tst_df], ['train', 'dev', 'test']):
        df = extract_audio(args.raw_path, df)

        df.to_pickle(os.path.join(args.save_path, f'B{split}.pkl'))
        print(f"saved {split}.pkl in {args.save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw_path', type=str, default='./data')
    parser.add_argument('--script_path', type=str, default='./data/Script.txt')
    parser.add_argument('--save_path', type=str, default='./data')
    parser.add_argument('--train_size', type=float, default=.8)
    args_ = parser.parse_args()
    main(args_)