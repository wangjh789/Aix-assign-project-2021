## 자연어 처리를 이용한 리뷰 감성분석(Natural language processing)


### Members :
2016003254/소프트웨어학과/고동현
2016003690/소프트웨어학과/왕종휘 

### I. Proposal 
 인터넷의 발달과 소셜미디어의 활성화로 대중이 정보를 소비하는 주체에서 생산하는 주체로 변화하기 시작했습니다. 온라인 커머스 사이트를 통해 상품을 구매한 고객들은 구매 후기, 또는 상품평을 통해 자신들의 구매 경험을 공유합니다. 개인의 주관이 반영된 상품평은 브랜드, 가격 등 객관적 정보 못지않게 사람들의 구매 의사결정에 영향을 줍니다. 이와 같이 주관적인 데이터를 수집하고 분석해 마케팅, 고객관리에 사용한다면 상품의 구매율과 브랜드에 대한 가치를 향상시키는 데에 도움을 줄 수 있을 거라 생각합니다.
 
우선 레퍼런스가 많은 영문으로 자연어 처리를 진행하면서 NLP에 대한 전반적 과정과 방법을 익히고, 한글과 영문의 구조적 차이를 인지하면서 한글에 대한 자연어 처리를 진행합니다.
 
### II. Datasets
Bag of Words Meets Bags of Popcorn
-	id: 각 데이터의 id
-	sentiment : review의 Sentiment 결과 값 (1:긍정 0:부정)
-	review : 영화평의 텍스트


네이버 영화 평점 데이터
-	id : 각 데이터의 id (네이버가 설정)
-	document : 영화평의 텍스트
-	label : document 의 Sentiment 결과 값 (1:긍정 0:부정)

### III. Methodology 
텍스트를 머신러닝에 적용하기 위해선 비정형 텍스트 데이터를 피처 형태로 추출하고 추출된 피처에 의미 있는 값을 부여하는 피처 벡터화를 거쳐야 합니다. 피처 벡터화에는 대표적으로 Bag Of Words, Word2Vec 이렇게 두가지 방법이 있습니다.
#### 순서
1.	텍스트 사전 준비 : 텍스트를 피처로 만들기 전에 클렌징, 토큰화, 스탑워드 제거를 통해 텍스트 정규화 작업을 수행합니다.
2.	피처 벡터화 : 가공된 텍스트에 피처를 추출하고 여기에 벡터값을 할당합니다. 
 - Count 기반 벡터화 : 각 문서에서 해당 단어가 나타나는 횟수가 높을수록 중요한 단어로 인식함.
 - TF-IDF 기반 벡터화 : 자주 나타나는 단어에 높은 가중치를 주되, 모든 문서에서 전반적으로 자주 나타나는 단어에 대해서는 패널티를 줌.
3.	ML모델 수립 및 학습/예측 평가 : 피처 벡터화된 데이터 세트에 ML모델을 적용해 학습/예측 및 평가를 수행합니다.

영문 자연어 처리는 사이킷런에 내장된 BOW 방식의 피처 벡터화를 진행 후, 로지스틱 회귀 모델로 지도학습을 수행합니다. 
한글 자연어 처리도 영문과 동일하게 처리하되 피처 벡터화의 토큰화 작업은 KoNLPy 한글 형태소 패키지의 Twitter 클래스를 이용합니다.

평가는 사이킷런의 accuracy_score()로 정확도를 측정합니다. 추가적으로 모델이 이진분류를 수행하기 때문에 roc_auc_score()도 진행하겠습니다.


#### 1.영문 자연어 처리
캐글에서 가져온 Bag of Words Meets Bags of Popcorn 로 진행합니다.
(https://www.kaggle.com/c/word2vec-nlp-tutorial)

##### 1.1 텍스트 사전 준비
```
review_df = pd.read_csv('./labeledTrainData.tsv',header=0,sep='\t',quoting=3)
review_df.head(10)
print(review_df['review'][0])
```
<img width="506" alt="스크린샷 2021-12-05 오후 5 27 59" src="https://user-images.githubusercontent.com/19744909/144739335-bf1c50d2-b0e4-4b94-a433-c492d0cd55f0.png">
<img width="1002" alt="스크린샷 2021-12-05 오후 5 28 46" src="https://user-images.githubusercontent.com/19744909/144739358-639810e9-efb8-4052-a67d-e06f893f6881.png">
첫번째 리뷰는 위의 사진과 같이 HTML 형식에서 추출해 <br />태크가 존재하는 것을 볼 수 있습니다.

```
import re

review_df['review'] = review_df['review'].str.replace('<br />',' ')

review_df['review'] = review_df['review'].apply(lambda x: re.sub("[^a-zA-Z]"," ",x))
```
다음과 같이 태그와 영문이 아닌 숫자/특수문자는 공백으로 대체합니다.

```
from sklearn.model_selection import train_test_split

class_df = review_df['sentiment']
feature_df = review_df.drop(['id','sentiment'],axis=1,inplace=False)

X_train,X_test,y_train,y_test = train_test_split(feature_df,class_df,test_size=0.3, random_state=100)
```
모델은 학습하고 평가하는데 필요한 sentiment와 review 컬럼을 제외한 나머지 컬럼들을 제거합니다.
그 후 데이터를 학습데이터는 70, 평가데이터는 30의 비율로 나눕니다.
```
X_train.shape, X_test.shape
```
<img width="1008" alt="스크린샷 2021-12-05 오후 5 38 23" src="https://user-images.githubusercontent.com/19744909/144739642-2b17be47-dfe8-485c-8bd3-b0b7442b1efc.png">
데이터 셋이 학습데이터 17500, 평가데이터 7500으로 나뉜 것을 확인 할 수 있습니다.

##### 1.2 피처 벡터화

```
CountVectorizer(stop_words='english',ngram_range=(1,2))
```
사이킷런 sklearn.feature_extraction.text의 클래스 CountVectorizer를 통해 학습/평가에 불 필요한 스탑워드 제거를 수행합니다. stop_words 파라미터 'english'에 해당하는 단어는 아래의 링크에서 확인가능합니다.
(https://www.ranks.nl/stopwords)



