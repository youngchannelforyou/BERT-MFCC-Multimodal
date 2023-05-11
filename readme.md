###BERT모델, CNN-LSTM모델을 활용한 감정 인식 모델

사용한 데이터 [KEMDy20](https://nanum.etri.re.kr/share/kjnoh/KEMDy20?lang=ko_KR)


### 데이터 디렉토리 구조

├── KoBERT  
│&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;└── __pycache__  
├── data  
│   ├── EDA  
│   ├── IBI  
│   ├── Session01  
│   ├── Session02  
│   ├── Session03  
│   ├── Session04  
│   ├── Session05  
│   ├── Session06  
│   ├── Session07  
│   ├── Session08  
│   ├── Session09  
│   ├── Session10  
│   ├── Session11  
│   ├── Session12  
│   ├── Session13  
│   ├── Session14  
│   ├── Session15  
│   ├── Session16  
│   ├── Session17  
│   ├── Session18  
│   ├── Session19  
│   ├── Session20  
│   ├── Session21  
│   ├── Session22  
│   ├── Session23  
│   ├── Session24  
│   ├── Session25  
│   ├── Session26  
│   ├── Session27  
│   ├── Session28  
│   ├── Session29  
│   ├── Session30  
│   ├── Session31  
│   ├── Session32  
│   ├── Session33  
│   ├── Session34  
│   ├── Session35  
│   ├── Session36  
│   ├── Session37  
│   ├── Session38  
│   ├── Session39  
│   ├── Session40  
│   ├── TEMP  
│   ├── annotation  
│   ├── pkls  
│   └── wav  
└── result  
  
###실행 방법
set.py - 총 3회실행 <EDA IBI TEMP 데이터에 대하여 폴더 지정후 3번 실행 할 것>
이후 preprocess 실행
이후 train 실행


아래의 소스코드를 활용하여 구성하였습니다.
[코드 출처](https://github.com/youngbin-ro/audiotext-transformer)
  
