# Mission 17 - Streamlit

# 미션 소개

이번 미션에서는 사용자가 웹 인터페이스에서 이미지를 업로드하면 해당 이미지가 무엇인지 **분류(Classification)**하는 Streamlit 기반 웹 서비스를 개발해봅시다. 이를 위해 **Hugging Face의Transformers 라이브러리**에서 제공하는 사전학습된 이미지 분류 모델을 활용해볼거에요.

## 🎯 미션 목표

이번 미션에서는 **Streamlit**을 사용하여

**이미지 업로드 → AI 모델 추론 → 예측 결과 시각화**까지 포함한 **이미지 분류 웹 애플리케이션**을 구현합니다.

완성된 앱은 **Streamlit Cloud에 배포**하여 **외부에서 접속 가능한 웹 서비스 형태로 제출**해야 합니다.

> 💡 이 미션을 통해
> 
> - Streamlit UI 구성 능력
> - Hugging Face 모델 활용 경험
> - **AI 서비스 배포 경험**을 함께 가져가는 것이 목표입니다.

이번 미션을 해결하면, 여러분들은 모델을 구현하는 것에 끝나지않고 직접 AI 기반 서비스를 만드는 엔지니어로 성장할 수 있어요.

---

# 가이드라인

## 1. 환경 설정

다음 라이브러리들을 설치해야 합니다:

- **streamlit**: 웹 앱 프레임워크
- **transformers**: Hugging Face 모델 사용
- **torch**: PyTorch 딥러닝 프레임워크
- **pillow**: 이미지 처리

---

## 2. 사용할 모델

Hugging Face Hub에서 제공하는 이미지 분류 모델을 사용합니다:

- **모델명**: `google/vit-base-patch16-224`
- **특징**: Vision Transformer(ViT) 기반, ImageNet 1000개 클래스 분류
- **입력**: 224x224 크기의 이미지
- **출력**: 상위 N개 예측 클래스와 확률

---

## 3. 핵심 구현 요소

### 3.1 모델 로딩

- Transformers의 `pipeline` 함수를 사용하여 이미지 분류 파이프라인을 생성합니다
- `"image-classification"` 태스크를 지정하면 자동으로 적절한 모델이 로드됩니다

### 3.2 Streamlit UI 구성

다음 Streamlit 컴포넌트들을 활용하세요:

- `st.title()`: 앱 제목 표시
- `st.file_uploader()`: 이미지 파일 업로드 위젯
- `st.image()`: 업로드된 이미지 화면에 표시
- `st.button()`: 분류 실행 버튼
- `st.write()` 또는 `st.metric()`: 분류 결과 출력

### 3.3 이미지 처리

- PIL 라이브러리의 `Image.open()`을 사용하여 업로드된 파일을 이미지 객체로 변환합니다
- 변환된 이미지 객체를 그대로 분류 모델에 전달할 수 있습니다

### 3.4 성능 최적화

- `@st.cache_resource` 데코레이터를 사용하여 모델을 캐싱하세요
- 이렇게 하면 페이지가 새로고침되어도 모델을 다시 로드하지 않습니다

---

## 4. 구현 순서

1. **페이지 설정**: `st.set_page_config()`로 페이지 제목과 아이콘 설정
2. **모델 로딩 함수 작성**: 캐싱을 적용한 모델 로딩 함수 구현
3. **UI 레이아웃 구성**: 제목, 설명, 파일 업로더 배치
4. **이미지 표시**: 업로드된 이미지를 화면에 미리보기
5. **분류 실행**: 버튼 클릭 시 모델로 이미지 분류 수행
6. **결과 출력**: 분류 결과를 시각적으로 표시

---

## 5. 결과 출력 형식

분류 모델은 다음과 같은 형식으로 결과를 반환합니다:

- **label**: 예측된 클래스 이름 (예: "golden retriever")
- **score**: 해당 클래스의 확률 (0~1 사이 값)

결과를 표시할 때 다음을 고려하세요:

- 상위 1개 결과를 강조하여 표시
- `st.progress()`를 활용하여 확률을 시각적으로 표현
- 퍼센트 형식으로 신뢰도 표시 (예: 92.3%)

## 6. Streamlit Cloud 배포 (필수)

이번 미션은 **로컬 실행 제출 ❌**

**Streamlit Cloud 배포 URL 제출 ⭕** 입니다.

### 📌 배포 방법

1. 프로젝트를 **GitHub 저장소**에 업로드
2. [https://streamlit.io/cloud](https://streamlit.io/cloud) 접속
3. **New app** 클릭
4. GitHub 저장소 선택
5. Main file path:
    
    ```
    app.py
    ```
    
6. Deploy 클릭
7. 생성된 **`.streamlit.app` URL 확인**

---

## 6. 추가 도전 과제 (선택)

- [ ]  Top-5 결과를 막대 차트로 시각화 (plotly, altair)
- [ ]  카메라로 직접 촬영하는 기능 추가 (`st.camera_input()`)
- [ ]  여러 이미지를 한 번에 분류하는 기능
- [ ]  분류 결과에 따른 이모지 표시

---

# 제출 요구사항(Streamlit Cloud 배포)

1. **GitHub 저장소 URL:** 작성한 `app.py`와 `requirements.txt` 파일을 본인의 GitHub 저장소(Repository)에 업로드하세요.
2. **Streamlit Cloud 배포 URL (필수)**
    - 예: [https://kasdf12345gbasewed2v.streamlit.app/](https://kyudrya5fbrftgofwtpy7n.streamlit.app/)
3. **스크린샷**: 앱 실행 화면 캡처 (선택)

---

# 참고 자료

- [Hugging Face Image Classification](https://huggingface.co/docs/transformers/tasks/image_classification)
- [Streamlit File Uploader](https://docs.streamlit.io/library/api-reference/widgets/st.file_uploader)
- [Vision Transformer (ViT) 모델](https://huggingface.co/google/vit-base-patch16-224)
- Streamlit Cloud 배포 가이드: [Deploy an app on Streamlit Cloud](https://www.google.com/search?q=https://docs.streamlit.io/streamlit-community-cloud/get-started/deploy-an-app)

---

# 힌트

> 💡 모델 로딩 시간: 처음 실행 시 모델 다운로드에 시간이 걸립니다. @st.cache_resource로 캐싱하세요.
> 

> 💡 이미지 형식: PIL Image 객체를 그대로 분류 함수에 전달하면 됩니다.
> 

> 💡 결과 개수 조절: top_k 파라미터로 반환받을 결과 개수를 지정할 수 있습니다.
> 

> 💡 로딩 표시: st.spinner()로 분석 중임을 사용자에게 알려주세요.
> 

---

# 예상 결과물

앱 실행 시 다음과 같은 기능이 동작해야 합니다:

1. 사용자가 이미지 파일을 업로드할 수 있는 업로더
2. 업로드된 이미지 미리보기
3. "분류하기" 버튼 클릭 시 AI 분류 수행
4. 분류 결과 (클래스명 + 확률) 시각적 표시