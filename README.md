student-gameplay
==============================

This project is about Predict Student Performance from Game Play from Kaggle.

Project Organization
------------

    Gameplay Prediction Project 
        ├── data
        │   ├── processed      <- processing 된 train, test 및 encoder, scaler 데이터
        │   └── raw            <- Kaggle 에서 제공하는 Raw 데이터셋
        │ 
        ├── models             <- 훈련시킨 DNN, LSTM 모델 .pt 파일
        ├── EDA_notebook       <- Exploratory Data Analysis (EDA) 진행한 Jupyter notebook 
        │
        ├── requirements.txt   <- 프로젝트 관련 가상환경 설정 (패키지)               
        │
        └── src                <- Source code for use in this project.
            │
            ├── data           
            │   └── dataloader.py  <- 모델에 넣기 위한 데이터 전처리 관련 코드
            │
            ├── models         
            │   │                      
            │   ├── criterion.py   <- loss 계산 
            │   ├── metric.py      <- 성능지표 계산 
            │   ├── model.py       <- import 할 모델의 모음
            │   ├── optimizer.py   <- Adam 등 optimizer 모음
            │   ├── scheduler.py   <- 학습 스케쥴러 모음
            │   └── trainer.py     <- 모델 훈련, 검증 스크립트
            │
            ├── args.py            <- parse 할 argument 모음                            
            │
            ├── train.py           <- 실행 코드 (main)
            │
            └── utils.py           <- 부가 기능

--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
