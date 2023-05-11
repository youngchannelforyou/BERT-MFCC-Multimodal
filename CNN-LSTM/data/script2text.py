import os
import argparse
import subprocess
import pandas as pd
from tqdm import tqdm
from pydub import AudioSegment
from pathlib import Path
import chardet
import numpy as np
#이 프리프로세싱 코드는 typeA코드로 대표데이터 not under sampling
#구분자 ?!11!?

def make_annotationdf(path):
    new_df = pd.DataFrame()
    files = []
     
    for dirpath, dirnames, filenames in os.walk(os.path.join(path, "annotation")):
            for filename in filenames:
                if filename.endswith("_eval.csv"):
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
    
    for index, row in new_df.iterrows():
        file_contents = ""
        with open(row['text_path'], "rb") as file:
            result = chardet.detect(file.read())
            runtype = result['encoding']
        
        if(runtype == 'EUC-KR'):
            with open(row['text_path'], "r", encoding='EUC-KR') as file:
                file_contents = file.read()
                sentences.append(file_contents)
        
        elif(runtype == 'CP949'):
            with open(row['text_path'], "r", encoding='CP949') as file:
                file_contents = file.read()
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

def do_preprotestdev(trn_df):

    filtered_df = trn_df#[trn_df['audio_path'].str.contains(f'_User0{human_num:02}')]
    #filtered_df = trn_df[trn_df['audio_path'] == f'Sess{sess:02}_script01_User002M_001']
    #trn_df = trn_df.drop(trn_df[trn_df['person_idx'] in 'Sess01_script06_User002M'].index)
    #trn_df = trn_df[~trn_df.apply(lambda x: x.str.contains('Sess01_script06_User002M')).any(axis=1)]
    print(trn_df)
    trn_df = pd.DataFrame()
    
    trn_df['1'] = filtered_df['emotion']
    trn_df['2'] = filtered_df['sentence']
    print(trn_df)
    return trn_df

def do_prepro(trn_df):
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

    data_df = do_prepro(data_df)
    tst_df = do_preprotestdev(data_df)
    tst_df.to_csv(os.path.join(args.save_path, '_person0.csv'), index=False)
        # with open(os.path.join(args.save_path, f'Sess{sess:02}_person1.txt'), "w") as f:
    #     f.write(tst_df[script])
    print(f"Seerson1.csv in {args.save_path}")
            
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw_path', type=str, default='./data')
    parser.add_argument('--script_path', type=str, default='./data/pkls/Script.txt')
    parser.add_argument('--save_path', type=str, default='./pppppppppp')
    parser.add_argument('--train_size', type=float, default=.8)
    args_ = parser.parse_args()
    main(args_)