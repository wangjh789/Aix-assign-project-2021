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
텍스트를 머신러닝에 적용하기 위해선 비정형 텍스트 데이터를 피처 형태로 추출하고 추출된 피처에 의미 있는 값을 부여하는 피처 벡터화를 거쳐야 합니다. 피처 벡터화에는 대표적으로 Bag Of Words, Word2Vec 이렇게 두가지 방법이 있습니다. 여기선 사이킷런에 내장된 BOW 방식을 이용합니다.
#### 순서
1.	텍스트 사전 준비 : 텍스트를 피처로 만들기 전에 클렌징, 토큰화, 스탑워드 제거를 통해 텍스트 정규화 작업을 수행합니다.
2.	피처 벡터화 : 가공된 텍스트에 피처를 추출하고 여기에 벡터값을 할당합니다. 
 - Count 기반 벡터화 : 각 문서에서 해당 단어가 나타나는 횟수가 높을수록 중요한 단어로 인식함.
 - TF-IDF 기반 벡터화 : 자주 나타나는 단어에 높은 가중치를 주되, 모든 문서에서 전반적으로 자주 나타나는 단어에 대해서는 패널티를 줌.
3.	ML모델 수립 및 학습/예측 평가 : 피처 벡터화된 데이터 세트에 ML모델을 적용해 학습/예측 및 평가를 수행합니다.

영문 자연어 처리는 사이킷런에 내장된 BOW 방식의 피처 벡터화를 진행 후, 로지스틱 회귀 모델로 지도학습을 수행합니다. 
한글 자연어 처리도 영문과 동일하게 처리하되 피처 벡터화의 토큰화 작업은 KoNLPy 한글 형태소 패키지의 Okt 클래스를 이용합니다.

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

위의 사진과 같이 reivew는 HTML 형식에서 추출해 <br />태크가 존재하는 것을 볼 수 있습니다.


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
사이킷런 sklearn.feature_extraction.text의 클래스 CountVectorizer를 통해 학습/평가에 불 필요한 스탑워드 제거를 수행합니다. stop_words 파라미터 'english'에 해당하는 단어는 다음의 링크에서 확인가능합니다. (https://www.ranks.nl/stopwords)

BOW 방식의 피처 벡터화는 단어의 순서를 고려하지 않기 때문에 문장 내에서 단어의 문맥적인 의미가 무시됩니다. 이러한 단점을 보완하기위해 ngram_range 파라미터를 이용합니다. 위와 같이 (1,2)로 설정하면 토큰화된 단어를 1개씩, 그리고 순서대로 2개씩 묶어서 피처로 추출합니다.

##### 1.3 ML모델 수립 및 학습/예측 평가

```
from sklearn.linear_model import LogisticRegression

LogisticRegression(C=10)
```
ML모델은 사이킷런 sklearn.linear_model의 클래스 LogisticRegression를 사용합니다.

파라미터 C는 coefficient를 뜻합니다. C가 높을수록 학습을 더욱 복잡하게, 다시말해 학습데이터에 대해 적합하게 학습을 수행합니다. 그렇기 때문에 C가 너무 높으면 평가 데이터셋에 성능이 좋지 않은 과적합 문제를 야기할 수 있습니다. 

```
pipeline = Pipeline([
    ('cnt_vect', CountVectorizer(stop_words='english',ngram_range=(1,2))),
    ('lr_clf',LogisticRegression(C=10))
])
```
평가 데이터를 적용할 때도 같은 과정을 거쳐야하므로, 피처 벡터화, ML모델 수립의 2가지 과정을 Pipline으로 하나로 통합해 사용하겠습니다.

```
pipeline.fit(X_train['review'],y_train)
pred = pipeline.predict(X_test['review'])
pred_probs = pipeline.predict_proba(X_test['review'])[:,1]
```
로지스틱회귀 모델을 학습시키고 성능을 평가하는 과정입니다. 

predict()는 단순히 1,0 값을 반환하지만, predict_proba는 다음과 같이 확률을 반환하므로 ROC 곡선그래프와 roc_auc_score()로 적절한 threshold를 찾을 때에 유용합니다.
<img width="980" alt="스크린샷 2021-12-05 오후 6 13 40" src="https://user-images.githubusercontent.com/19744909/144740650-9df9d86c-0ccb-4ec0-a96d-748bb69886d5.png">

```
print('예측 정확도: {0:.4f}\n ROC-AUC : {1:.4f}\n'.format(accuracy_score(y_test,pred),roc_auc_score(y_test,pred_probs[:,1])))
```
<img width="983" alt="스크린샷 2021-12-05 오후 6 27 11" src="https://user-images.githubusercontent.com/19744909/144741046-c9a63b17-b588-410e-8a49-8dfe07be0128.png">

여기서 ROC-AUC가 의미하는 것은 결과를 얼마나 신뢰할 수 있는지를 의미합니다. TPR가 클수록 FPR이 작을수록 그래프의 아래면적(AUC)이 증가합니다.

<img width="407" alt="스크린샷 2021-12-05 오후 6 18 35" src="https://user-images.githubusercontent.com/19744909/144741095-6ce40d89-115f-4327-9158-effac3527295.png">

위 모델은 정확도가 84%, AUC 94%로 리뷰의 감성분석에 쓸만하고, 신뢰할 만한 모델이라고 볼 수 있습니다.

#### 2.한글 자연어 처리
한글은 띄어쓰기와 조사 때문에 영문보다 자연어 처리가 어렵습니다. 잘못된 띄어쓰기는 단어의 의미를 왜곡시키거나 없는 단어로 인식하게 합니다. 또 한글의 조사는 경우의 수가 많아 전처리 시 제거하기가 어렵습니다.
그렇기에 KoNLPy를 이용해 한글의 피처 백터화를 수행합니다.

Lucy Park의 깃허브에 있는 네이버 영화리뷰 데이터셋을 사용합니다. (https://github.com/e9t/nsmc)
해당 데이터셋은 학습을 위한 데이터셋과 평가를 위한 데이터셋이 파일로 분리되어 있어 train_test_split()는 생략합니다.

#### 2.1 텍스트 사전 준비
```
train_df = pd.read_csv('ratings_train.txt',sep='\t')
train_df.head(10)
```
<img width="599" alt="스크린샷 2021-12-05 오후 6 54 25" src="https://user-images.githubusercontent.com/19744909/144741812-7637cdad-55dd-4d2b-8ca1-b465e3d0372b.png">
해당 데이터셋의 구조는 위와 같습니다.

```
train_df = train_df.fillna(' ')
train_df['document'] = train_df['document'].apply(lambda x: re.sub(r"\d+"," ",x))

test_df = pd.read_csv("ratings_test.txt",sep='\t')
test_df = test_df.fillna(' ')
test_df['document'] = test_df['document'].apply(lambda x: re.sub(r"\d+"," ",x))

train_df.drop('id',axis=1,inplace=True)
test_df.drop('id',axis=1,inplace=True)
```
학습 데이터셋과 평가 데이터셋 중 Null과 숫자를 공백값으로 대체합니다. 그 후 학습, 평가하는데에 필요없는 id컬럼을 제거합니다.

```
from konlpy.tag import Okt

okt = Okt()
def tw_tokenizer(text):
    tokens_ko = okt.morphs(text)
    return tokens_ko
```
위 함수는 토큰화를 담당하는 함수입니다. 

```
print(otk.morphs("아빠가 방에 들어가신다"))
print(otk.morphs("아빠가방에 들어가신다"))
```
<img width="1000" alt="스크린샷 2021-12-05 오후 7 04 41" src="https://user-images.githubusercontent.com/19744909/144742111-ca4db398-ec6c-40e4-be78-5f5c8e995dc4.png">

morphs()함수는 파라미터로 받는 text를 형태소 단위로 끊어 리스트 형태로 반환합니다. 

##### 2.2 피처 벡터화

```
tfidf_vect = TfidfVectorizer(tokenizer=tw_tokenizer, ngram_range=(1,2),min_df=3,max_df=0.9)
tfidf_vect.fit(train_df['document'])
tfidf_matrix_train = tfidf_vect.transform(train_df['document'])
```
BOW의 TF-IDF방식의 피처 벡터화를 수행합니다. TfidfVectorizer를 이용해 TF-IDF 피처 모델을 생성합니다. tokenizer 파라미터는 별도의 커스텀함수를 이용시 적용합니다. 한글의 토큰화를 위해 구현한 tw_tokenizer()함수를 넣습니다.
min_df와 max_df는 각각 빈도수가 너무 낮은, 너무 높은 단어를 제외하기 위한 파라미터입니다. 너무 낮은 빈도수를 가진 단어는 크게 중요하지 않은 단어이고, 너무 높은 빈도수를 가진 단어는 스톱워드와 비슷한 문법적인 특성으로 반복적인 단어일 가능성이 높습니다.

```
tfidf_vect.vocabulary_.items()
```
<img width="992" alt="스크린샷 2021-12-05 오후 7 37 38" src="https://user-images.githubusercontent.com/19744909/144743088-212cd453-35e2-4ba9-aa2c-e1b8a0f70e23.png">
피처 모델이 학습한 단어 사전입니다.

```
print(tfidf_matrix_train)
```
<img width="993" alt="스크린샷 2021-12-05 오후 7 42 19" src="https://user-images.githubusercontent.com/19744909/144743235-daedb38c-b97a-42e8-831f-d652680e5081.png">
학습 데이터의 (document_id,tokne_id) tf-idf Score 를 볼 수 있습니다.

```
tfidf_matrix_test = tfidf_vect.transform(test_df['document'])
```
학습 데이터로 학습된 TF-IDF 피처 모델을 이용해 평가 데이터셋 또한 '(document_id,tokne_id) tf-idf Score'의 형태로 만듭니다.


##### 2.3 ML모델 수립 및 학습/예측 평가
```
lg_clf = LogisticRegression(C=3)
lg_clf.fit(tfidf_matrix_train,train_df['label'])
preds = lg_clf.predict(tfidf_matrix_test)
pred_probs = lg_clf.predict_proba(tfidf_matrix_test)
```
ML모델로 로지스틱회귀 모델을 선택했습니다. 그 후 로지스틱회귀 모델에 학습데이터 셋으로 학습시키고 평가 데이터 셋으로 평가합니다.


