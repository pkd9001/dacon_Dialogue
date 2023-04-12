# dacon_Dialogue
# 월간 데이콘 월간 데이콘 발화자의 감정인식 AI 경진대회
## 링크 : https://dacon.io/competitions/official/236027/overview/description
### 1. 개요
![데이콘 이미지 1](https://user-images.githubusercontent.com/100681144/231427243-394ac05d-0f87-4950-896a-ce626378dd55.PNG)

### 2. 데이터

* train.csv [파일]
  * ID : 행 별 고유 id
  * Utterance : 발화문 
  * Speaker : 발화자
  * Dialogue_ID : Dialogue 별 고유 id
  * Target : 감정 (neutral, joy, surprise, anger, sadness, disgust, fear 존재)
  
* test.csv [파일]
  * ID : 행 별 고유 id
  * Utterance : 발화문 
  * Speaker : 발화자
  * Dialogue_ID : Dialogue 별 고유 id
  
![데이콘 이미지 2](https://user-images.githubusercontent.com/100681144/231428091-ece829e8-7c97-4f0e-990e-4c8f63b8db3d.PNG)

* dialogue 특성에 따라 일반적인 분류 작업시 낮은 score를 보임

### 3. 데이터 전처리

* dialogue 특성을 활용하는 방안을 모색
   * 특정 대사 다음으로 나오는 대사는 전의 대사에 영향을 받는 것을 이용
   * A문장에 대해 B문장이 만들어 지므로 BERT 모델의 input을 2개의 sentence로 구분하여 넣기
   ![데이콘 이미지 3](https://user-images.githubusercontent.com/100681144/231434018-5fd9827b-283e-4c29-9985-988a5b341c87.PNG)
   * 성능이 향상되는 것을 보임

* 추가 방안을 모색
  * A문장은 B문장에게 영향을, B문장은 C문장에 영향을, C는 D...
  * A, B, C, D,...의 문장을 넣는 Stack 방식을 채택

**예시**

**Utterance_add**|**Utterance**|**label**
---|---|---
.|문장A.|0
문장A.|문장B.|0
문장A.문장B.|문장C.|1
문장A.문장B.문장C.|문장D.|2

### 4.결과(dacon F1 기준)
* 74등을 기록했지만 유의미한 결과를 보임

**Normal**|**Next Sentence Add**|**Sentence Stack**
---|---|---
26.16|39.43|43.46
