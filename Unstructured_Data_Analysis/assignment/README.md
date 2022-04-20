# assignment code
## [various data analysis with text data]

### ```1_Preprocessing.ipynb```
text 데이터를 전처리하는 과정 실습
- Tokenizing
- POS tagging
- Stopword 제거
- Stemming
- matplotlib을 활용한 단어 빈도 수 시각화

### ```2_Network Analysis.ipynb```
text 데이터로 동시 출현 단어 그래프 시각화 실습
- 노드와 엣지 추가
- 다양한 레이아웃 활용하기
- 네트워크의 특징 파악
- 이웃 노드 파악
- 각 노드의 중심성 지표 파악
- Gephi에서 활용하기 위한 coocurrence matrix 생성

### ```3_Document_Level_Analysis(TF-IDF).ipynb```
text 데이터 활용 - vector space model 생성, TF-IDF 벡터 생성

### ```4_Document_Level_Analysis(Word2vec,Doc2vec).ipynb```
text 데이터 활용 - Word2vec,Doc2vec 학습

### ```5_Document_Level_Analysis(Elmo).ipynb```, ```5_Document_Level_Analysis(LDA, Glove).ipynb```
- Elmo pretrained model을 활용하여 spam mail 분류하기
- LDA를 활용하여 로이터 뉴스 데이터 분류하기
- Glove를 활용하여 IMDB 단어 임베딩 하기

### ```6_Clustering_Classification.ipynb```
- 이진분류(감성분석): IMDB 영화 리뷰 긍/부정 분류 -> 문장 극성 계산하기
  - Vader 감성분류기 사용
  - textblob 사용: pattern anlyzer, naive bayesian analyzer
- 다중분류: 로이터 뉴스 기사 분류
  - 단순 NN(Neural Network) 활용
- text clustering
  - 연합뉴스 데이터 유사 문서 클러스터링 (kmeans clustering 활용)

### ```7_Text_Crawling.ipynb```
- http://media.daum.net/digital 에서 뉴스 리스트와 URL을 불러오기
    -> BeautifulSoup, html구조 활용

### ```8_Sequence_modeling_1.ipynb```
- IMDB 영화 리뷰 분류 문제
  - SimpleRNN, LSTM, 1D CNN 사용
- 기온 예측 문제
  - 1D CNN, 1D CNN + RNN, GRU, bidirectional RNN 사용 

### ```9_Sequence_modeling_2.ipynb```
Transformer 구현
- Positional Encoding
- Self Attention
- Encoder/Decoder
- Transformer
- Self Attention을 활용한 텍스트 분류
