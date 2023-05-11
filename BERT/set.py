import os
import csv
import argparse
#set.py 설명 read_csv를 원활하게 하기 위한 코드 
# EDA IBI TEMP 폴더 안에 csv파일들을 read_csv파일로 읽을 수 있는 형태로 바꿈

def main(args):
    for sess in range(args.start,args.end+1):
        for root, dirs, files in os.walk(f'{args.data_path}/Session{sess:02}'):
            for filename in files:
                if filename.endswith('.csv'):
                    filepath = os.path.join(root, filename)
                    with open(filepath, 'r') as csv_file:
                        reader = csv.reader(csv_file)
                        lines = list(reader)

                    #del lines[1]  # 2번째 줄 삭제

                    with open(filepath, 'w', newline='') as csv_file:
                        writer = csv.writer(csv_file)
                        for line in lines:
                            line_str = ",".join(line)  # 각 셀을 쉼표로 구분한 문자열로 변환
                            line_str += ","  # 각 행의 마지막에 쉼표 추가
                            line_str = line_str.replace(",,",",")
                            if 'Sess' in line_str:
                                line_str = line_str[:-1]
                            csv_file.write(line_str + '\n')  # 각 행을 CSV 형태로 기존 파일에 쓰기

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # settings
    parser.add_argument('--data_path', type=str, default='./data/EDA')
    parser.add_argument('--start', type=int, default=1)
    parser.add_argument('--end', type=int, default=41)
    args_ = parser.parse_args()
    main(args_)
    
#python set.py --data_path=./EDA --start=1 --end=41
