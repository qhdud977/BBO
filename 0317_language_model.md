# 0317_Language_Model

## Language Model 

- 문장이 말이 되는지 안 되는지를 판단하는 모델 => 문장(단어 시퀀스)에 확률 할당 

- 통계적 모델 
- 인공지능을 활용한 모델 (최근 ↑)

##### Statistical Language Model 

- 조건부 확률 사용 => 이전 단어 주어졌을 때 다음 단어로 등장할 확률이 높은 단어 
- a probability distribution over sequences of words. 
- 각 단어들이 이전의 단어가 주어졌을 때 다음 단어로 등장할 확률의 곱으로 구성
  - **P**(x1, x2, ... , xn) = **P**(x1)**P**(x2|x1)...**P**(xn|x1x2...xn-1) 
- 확률 count 기반으로 계산 
  - 사용하기 위해서는 학습 데이터가 많아야 함
    => 인공지능의 영역으로 넘어감.
- n-gram : 두 개/ 세 개 씩 끊어가면서 확률을 구함. 

## Word Representation

- 문장 표현하기

- 단어들 간의 관계를 잘 나타내기 위한 표현법 

- continuous representation : 주변 단어 참고 

- local representation : 해당 단어만 보고 표현 => one hot vector / n gram 

  ### 1. Bag of Words : 

  - 문장을 단어들의 순서 고려 없이 *출현 빈도*에 집중 (**토큰화**)하여 표현 
  -  ('정부': 0, '가': 1, '발표': 2, '하는': 3, '물가상승률': 4, '과': 5, '소비자': 6, '느끼는': 7, '은': 8, '다르다': 9) 
    → count : [1, 2, 1, 1, 2, 1, 1, 1, 1, 1] 

  ### 2. DTM (Document-Term Matrix) 

  - BoW의 집합을 행렬로 → 공간적 낭비 심함
  - 출현 빈도 높다고 중요한 단어 아님

  ### 3. TF-IDF (Term Frequency – Inverse Document Frequency) 

  - = **TF * IDF**

  - 단어의 빈도를 사용하여 DTM 내 **단어들의 중요도**로 가중치 주는 방식 

  - 특정 문서 내에서 단어 빈도가 높을수록, 전체 문서들 중 단어를 포함한 문서가 적을 수록 높아짐 
    → **모든 문서에 흔하게 나타나는 단어를 걸러냄**

  - 문서의 유사도, 중요도, 단어의 중요도 

  - 단어의 의미를 고려하지는 못함

  - **TF** : 특정 문서에서 특정 단어의 등장 횟수 (=DTM)

    ```python
    def tf(t, d):
    
      return d.count(t)
    ```

  - **DF** : 특정 단어가 포함된 문서의 개수 (문서 빈도)

    - 한 단어가 문서 전제 집합에서 얼마나 공통적으로 나타나는지 

    ```python
    for doc in docs:
            df += t in doc
    ```

  - **IDF**  = *log (n/1+DF)* => 희귀한 단어에 대한 가중치 ↑

    - 역문서빈도 : **특정 문서에서만 자주 나오는 단어**가 그 문서에서 진짜 **중요한 단어**라고 생각 
      => DF의 반비례하는 값으로 계산
    - n : 문서의 개수 

    ```python
    def idf(t):
        df = 0
        for doc in docs:
            df += t in doc
        return math.log(N/(df + 1))
    
    # 값들이 너무 커지지 않도록 log 취함. 
    # 분모 0 되지 않도록 df+1 사용 
    ```

  - **Sklearn 활용**

    ```python
    corpus = [
        'you know I want your love',
        'I like you',
        'what should I do ',    
    ]
    
    # tf
    from sklearn.feature_extraction.text import CountVectorizer
    vector = CountVectorizer()
    print(vector.fit_transform(corpus).toarray()) 
         # fit : 코퍼스로부터 각 단어의 빈도 수를 기록
    print(vector.vocabulary_) 
         # vector.vocabulary_ : 각 단어의 인덱스가 어떻게 부여되었는지를 보여줌
    
    #tf-idf
    from sklearn.feature_extraction.text import TfidfVectorizer
    tfidfv = TfidfVectorizer().fit(corpus)
    
    ```

  ### 4. LSA (Latent Semantic Analysis)

  - https://wikidocs.net/24949

  - 잠재 의미 분석 : 추상적인 **주제 발견** 위한 통계적 모델 (**topic modeling**의 아이디어 제공)

  -  단어의 의미 고려 

  - **SVD(Singular Value Decomposition 특이값 분해)**를 통해 **DTM의 잠재된 의미**를 이끌어내는 방법
    : 행렬을 아래와 같이 세 개 행렬의 곱으로 분해하여 (특이값 분해)
    차원 축소를 하면서 A를 잘 대표할 수 있는 행렬(Σ)로 만들어 냄. 

    **A = U Σ V전치**      U : mxm 직교 / V : nxn 직교 / Σ : mxn 직사각 대각행렬)

    Σ 행렬의 대각 원소 값을 행렬 A의 특이값이라 함. 

    - 직교 : A x A전치 = I
    - 대각 : A\[i]\[j] = an  if i==j / otherwise : 0      
      ex) [[a 0 0], [0 b 0], [0 0 c], [0 0 0]] 

  - **Truncated SVD**

    - SVD의 대각 행렬에서 상위 t개만 남겨두는 방식  (t = 찾고자 하는 토픽의 개수)
    - 나머지를 지움으로써 계산 비용을 낮춤 (필요없는 정보 제거)

  ```python
  # X : TF-IDF 행렬 
  
  from sklearn.decomposition import TruncatedSVD
  svd_model = TruncatedSVD(n_components=20, algorithm='randomized', n_iter=100, random_state=122)
  # n_components = 토픽 개수
  
  svd_model.fit(X)
  np.shape(svd_model.components_)
  # svc_model.components_ : 직사각 대각행렬  (토픽의 수 x 단어의 수)
  
  terms = vectorizer.get_feature_names() # 단어 집합. 1,000개의 단어가 저장됨.
  
  # 각 행의 1000개의 단어 중 가장 값이 큰 값 단어로 출력
  def get_topics(components, feature_names, n=5):
      for idx, topic in enumerate(components):
          print("Topic %d:" % (idx+1), [(feature_names[i], topic[i].round(5)) for i in topic.argsort()[:-n - 1:-1]])
  get_topics(svd_model.components_,terms)
```
  

  
### 5. LDA (Latent Dirichlet Allocation)
  
- https://wikidocs.net/30708
  
  - 문서 : 토픽의 혼합 / 각 토픽은 확률 분포에 기반하여 단어를 생성한다고 가정 
  - 문서 작성 과정 (가정)
    1. 문서에 사용한 단어의 개수 선정 
    2. 문서를 구성할 토픽의 혼합을 확률 분포에 기반하여 결정
    3. 문서에 사용할 단어 :
       토픽 분포에 기반하여 **토픽 선정 후** 
       해당 토픽에서 단어 출현 확률 분포에 따라 사용할 **단어 선정**
  - **LDA는 주어진 데이터를 가지고 위 과정(토픽 → 단어)을 역추적하는 방식(단어 → 토픽)으로 토픽을 찾아냄**  
  - 입력 : DTM 또는 TF-IDF 행렬 
  - LDA 수행 과정 
    1. 사용자가 알고리즘에 토픽 개수 k 입력 
    2. 모든 단어를 k개의 토픽 중 하나에 랜덤으로 할당 
    3. 어떤 문서의 각 단어 w가 자신은 잘못된 토픽에 할당됐지만 다른 단어들은 잘 할당되어있는 걸로 안다면 두 가지 기준에 따라 토픽 재할당 받음.
       - 문서 d의 단어들 중 토픽 t에 해당하는 단어의 비율 
         - 같은 문서 안에서 토픽의 비율 확인 
           (한 문서 안에서 토픽의 분포 확인)
       - 토픽 t에서 단어 w의 분포 
         - 전체 문서에서 해당 단어가 제일 많이 할당되어 있는 토픽 확인 
         (전체 문서의 토픽에서 해당 단어의 분포 확인)
  
  ```python
  from gensim import corpora
  dictionary = corpora.Dictionary(tokenized_doc)
  # 각 단어를 (word_id, word_frequency)의 형태로 corpus에 저장 
  corpus = [dictionary.doc2bow(text) for text in tokenized_doc]
  print(corpus[1]) # 수행된 결과에서 두번째 뉴스 출력. 첫번째 문서의 인덱스는 0
```
  
  ```python
  # LDA 모델 훈련 => model : ladmodel 
  import gensim
  NUM_TOPICS = 20 #20개의 토픽, k=20
  ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics = NUM_TOPICS, id2word=dictionary, passes=15)
  # passes : 알고리즘의 동작 횟수
  topics = ldamodel.print_topics(num_words=4)
  # num_words : 4개의 단어만 출력
  for topic in topics:
      print(topic)
      
  # output : (18, '0.011*"year" + 0.009*"last" + 0.007*"first" + 0.006*"runs"')
  # (해당 토픽 번호, 해당 단어의 해당 토픽에 대한 기여도*단어)
```
  
  ```python
  #시각화
  pip install pyLDAvis
  
  import pyLDAvis.gensim
  pyLDAvis.enable_notebook()
  vis = pyLDAvis.gensim.prepare(ldamodel, corpus, dictionary)
  pyLDAvis.display(vis)
  
  # output : 원이 겹친다면 유사한 토픽이라는 뜻 
# 토픽의 번호 1부터 시작
  ```
  
  ```python
  # 각 문서의 토픽 분포는 이미 훈련된 LDA 모델인 ldamodel[]에 전체 데이터가 정수 인코딩 된 결과를 넣은 후에 확인이 가능
  for i, topic_list in enumerate(ldamodel[corpus]):
      if i==5:
          break
    print(i,'번째 문서의 topic 비율은',topic_list)
  ```

  

  ##### LSA vs. LDA (topic modeling 위한 것이긴 하지만 다른 방법임..)
  
- LSA : 차원을 축소하여 (SVD) 근접 단어들 토픽으로 묶음
  - LDA : 단어가 특정 토픽에 존재할 확률과 문서에 특정 토픽이 존재할 확률의 결합확률로 토픽 추정

  

  ## NLP Metrics (evaluation)

  다양한 task가 존재하기 때문에 각각에 맞는 metrics 필요 

  - cosine similarity : 벡터 간 코사인 각도로 벡터의 유사도 계산 

    - 가까울수록 1, 멀수록 -1, 관련 없다면 0

  - perplexity : 테스트 데이터에 대한 확률의 역수 
  
    - ppl = {**P**(x1, x2, ... , xn) = **P**(x1)**P**(x2|x1)...**P**(xn|x1x2...xn-1) } ^ (-1/n)
  - 최소화 : 문장의 확률 최대화 
    - 낮을수록 좋은 모델 

  - BLEU Score (Bilingual Evaluation Understudy Score) 

    - https://donghwa-kim.github.io/BLEU.html

    - 번역 모델에서 가장 많이 사용됨.

    - 기계 번역 결과와 사람이 번역한 결과의 유사도 (**얼마나 겹치는지**)=> n-gram으로 측정

      1. 단어 개수로 측정 

         : count(기계 | 사람) / count (기계)

      2. 중복 단어 보정 (같은 단어가 연속적으로 나올 때 과적합 보정)

         : count (기계 번역 문장의 각 유니그램에 대해 min(count, 유니그램 몇번 등장)) / count (기계 유니그램 수)

      3. 순서 고려한 n-gram

  - ROUGE(Recall-Oriented Understudy for Gisting Evaluation) Score 

    - https://huffon.github.io/2019/12/07/rouge/

    - 문장 요약 모델의 Metric으로 자주 사용

    - Rouge-N: n-gram으로 recall 점수를 측정 

    - Precision/Recall/f—score와 같은 방식으로 계산 

    - Rouge-L: 두 문장 중 겹치는 가장 긴 n-gram으로 계산 

    - Rouge-S: skip-gram을 허용해서 계산 
  
      

# 실습

데이터 전처리 

- 데이터 불러오기

  ```python
  import pandas as pd
  from sklearn.datasets import fetch_20newsgroups
  dataset = fetch_20newsgroups(shuffle=True, random_state=1, remove=('headers', 'footers', 'quotes'))
  documents = dataset.data
  ```

- 데이터 전처리

  - 특수 문자 제거 
  - 짧은 단어 제거 
  - 소문자 변환
  - 불용어 제거 

  ```python
  news_df = pd.DataFrame({'document':documents})
  # 특수 문자 제거
  news_df['clean_doc'] = news_df['document'].str.replace("[^a-zA-Z]", " ")
  # 길이가 3이하인 단어는 제거 (길이가 짧은 단어 제거)
  news_df['clean_doc'] = news_df['clean_doc'].apply(lambda x: ' '.join([w for w in x.split() if len(w)>3]))
  # 전체 단어에 대한 소문자 변환
  news_df['clean_doc'] = news_df['clean_doc'].apply(lambda x: x.lower())
  
  from nltk.corpus import stopwords
  stop_words = stopwords.words('english') # NLTK로부터 불용어를 받아옵니다. (제공)
  tokenized_doc = news_df['clean_doc'].apply(lambda x: x.split()) # 토큰화
  tokenized_doc = tokenized_doc.apply(lambda x: [item for item in x if item not in stop_words])
  # 불용어를 제거합니다.
  # 불용어 : 관사, 전치사, 조사, 접속사 등
  ```

- TF-IDF 행렬 만들기 (=> tokenized_doc)

=> 이후 LSA 혹은 LDA에 적용