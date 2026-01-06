# 🧴 MyPerfume  
**LG U+ Why Not SW Camp 8기 프로젝트**  
양원선 · 이정수 · 김태화


## 📌 프로젝트 개요
향수 지식이 없어도,  
**지금의 나에게 어울리는 향을 이해하고 선택**할 수 있도록 설계했습니다.


### 📖 분석 배경
기존 향수 추천 서비스는 다음과 같은 한계를 갖고 있습니다.

- 향조·노트 중심 구조로 비전문가에게 진입 장벽이 높음  
- 추천 결과에 대한 근거가 명확하지 않음  
- 개인의 스타일, 색감, 계절 맥락이 충분히 반영되지 않음  

이러한 문제를 해결하기 위해  
사용자의 **실제 선택 맥락**을 추천의 출발점으로 설정했습니다

### 🎯 분석 목표
사람이 향수를 고를 때 자연스럽게 고려하는 요소를 기준으로  
추천 로직을 설계하는 것을 목표로 했습니다.

- **스타일**  
  의류 이미지 기반 스타일 분류 결과 활용  
- **색감**  
  의류 색상과 향조 간 연상 관계를 점수화  
- **계절**  
  향수의 계절별 적합도와 현재 계절 맥락 반영  

세 가지 기준의 점수를 통합해  
향수별 종합 점수를 계산하고, 상위 3개 향수를 추천합니다.


## 📊 추천 로직
```
사용자 의류 이미지 & 선호 정보
        ↓
스타일 점수 / 색상 점수 / 계절 점수
        ↓
가중합(myscore) 계산
        ↓
Top-3 향수 추천 + 설명 생성
```
- 모든 점수는 정규화 후 가중치 튜닝
- Precision@k 기준 성능 개선 검증

## 🛠️ 설명 생성 (Explainability)
추천 결과에는 단순한 향수 리스트가 아닌,  
**스타일·색감·계절 맥락을 연결한 자연어 설명**을 함께 제공합니다.

- LLM은 설명 생성에만 사용  
- 추천 순위와 점수 계산은 데이터 기반 로직으로 분리 설계  

이를 통해 추천 결과에 대한 **이해도와 신뢰도**를 높였습니다.


## 📈 기대 효과
- 향수 지식이 없는 사용자도 추천 결과를 쉽게 이해 가능  
- 개인의 스타일과 상황에 맞는 향수 선택 경험 제공  
- 추천 근거가 명확한 설명 가능한 추천 시스템 구현  

## 📂 파일 구조 
```
/data # 데이터셋
/code # 전처리, 모델링, LLM
/django # 웹 서비스
/docs # 보고서, 참고 자료
README.md
```
## 💻 기술 스택

### Backend & Frontend
<img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white">
<img src="https://img.shields.io/badge/Django-092E20?style=for-the-badge&logo=django&logoColor=white">
<img src="https://img.shields.io/badge/Django_REST_Framework-ff1709?style=for-the-badge&logo=django&logoColor=white">

### Data / Modeling
<img src="https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white">
<img src="https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white">
<img src="https://img.shields.io/badge/Scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white">
<img src="https://img.shields.io/badge/LightGBM-02569B?style=for-the-badge&logo=lightgbm&logoColor=white">
<img src="https://img.shields.io/badge/CatBoost-FFCC00?style=for-the-badge&logo=catboost&logoColor=black">

### Infrastructure
<img src="https://img.shields.io/badge/Docker-2496ED?style=for-the-badge&logo=docker&logoColor=white">
<img src="https://img.shields.io/badge/AWS-232F3E?style=for-the-badge&logo=amazonaws&logoColor=white">
<img src="https://img.shields.io/badge/Nginx-009639?style=for-the-badge&logo=nginx&logoColor=white">
<img src="https://img.shields.io/badge/Gunicorn-499848?style=for-the-badge&logo=gunicorn&logoColor=white">
