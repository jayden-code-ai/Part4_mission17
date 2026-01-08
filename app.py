import streamlit as st
import pandas as pd
from transformers import pipeline
from PIL import Image
from deep_translator import GoogleTranslator

# 1. 페이지 초기 설정
# 페이지의 제목, 아이콘, 레이아웃을 설정합니다.
st.set_page_config(
    page_title="이미지 분류기",
    page_icon="🖼️",
    layout="wide"
)

# 2. 모델 로딩 함수 (캐싱 적용)
# @st.cache_resource: 모델을 매번 새로 로딩하면 시간이 오래 걸리므로,
# 한 번 로딩되면 캐시(메모리)에 저장해두고 재사용합니다.
@st.cache_resource
def load_model():
    """
    Hugging Face의 이미지 분류 모델을 로드합니다.
    사용 모델: google/vit-base-patch16-224 (Vision Transformer)
    """
    return pipeline("image-classification", model="google/vit-base-patch16-224")

def translate_to_korean(text):
    """
    영어 텍스트를 한국어로 번역하는 함수입니다.
    deep_translator 라이브러리를 사용하여 구글 번역기를 활용합니다.
    """
    try:
        # GoogleTranslator 객체를 생성하고 번역 수행
        # source='auto': 입력 언어 자동 감지
        # target='ko': 출력 언어를 한국어로 설정
        translator = GoogleTranslator(source='auto', target='ko')
        result = translator.translate(text)
        return result
    except Exception as e:
        # 번역 중 에러가 발생하면 원본 텍스트를 그대로 반환
        return text

def get_emoji(label, score):
    """
    분류된 라벨과 신뢰도 점수에 따라 적절한 이모지를 반환하는 함수입니다.
    label: 분류된 클래스 이름 (영어)
    score: 신뢰도 점수 (0~1)
    """
    # 신뢰도에 따른 상태 이모지 결정
    if score > 0.8:
        status = "😎" # 매우 확실함
    elif score > 0.5:
        status = "🤔" # 어느 정도 확실함
    else:
        status = "🧐" # 불확실함
        
    # 라벨 내용에 따른 동물 이모지 결정
    label_lower = label.lower()
    if 'dog' in label_lower or 'golden retriever' in label_lower or 'poodle' in label_lower or 'terrier' in label_lower:
        icon = "🐶" # 강아지 관련
    elif 'cat' in label_lower or 'tabby' in label_lower:
        icon = "🐱" # 고양이 관련
    elif 'bird' in label_lower:
        icon = "🐦" # 새 관련
    elif 'fish' in label_lower or 'shark' in label_lower:
        icon = "🐟" # 물고기 관련
    else:
        icon = "📷" # 그 외 사물 등
        
    return f"{status} {icon}"

# 3. 메인 UI 구성
def main():
    # 앱의 메인 제목과 설명
    st.title("🖼️ AI 이미지 분류 서비스")
    st.markdown("### 당신의 사진이 무엇인지 AI가 분석해드립니다!")

    # 세션 스테이트 초기화 (리셋 기능을 위해 사용)
    # Streamlit은 새로고침하면 변수가 초기화되는데, session_state에 저장하면 유지됩니다.
    if 'uploader_key' not in st.session_state:
        st.session_state.uploader_key = 0

    # 사이드바 구성 (사용 방법 및 리셋 버튼)
    with st.sidebar:
        st.header("사용 방법")
        st.info("1. **파일 업로드** 또는 **카메라**를 선택하세요.\n2. 이미지를 입력하면 자동 분석합니다.\n3. 상세한 확률 그래프를 확인해보세요!")
        st.divider()
        
        # 리셋 버튼: 누르면 session_state의 키 값을 변경하여 화면을 새로고침 효과를 줌
        if st.button("🗑️ 모든 정보 리셋", type="primary"):
            st.session_state.uploader_key += 1
            st.session_state.analysis_results = {} # 분석 결과 캐시도 초기화
            st.rerun() # 앱 재실행
            
        st.divider()
        st.caption("Powered by Hugging Face ViT Model")

    # 탭 구성: 파일 업로드 탭과 카메라 촬영 탭으로 분리
    tab1, tab2 = st.tabs(["📁 파일 업로드", "📸 카메라 촬영"])

    images_to_process = [] # 처리할 이미지들을 담을 리스트

    # 탭 1: 파일 업로드 기능
    with tab1:
        uploaded_files = st.file_uploader(
            "이미지 파일을 선택하세요 (여러 장 가능)", 
            type=["jpg", "jpeg", "png"], 
            accept_multiple_files=True,
            # key 값이 변하면 업로더가 초기화됨 (리셋 구현용)
            key=f"uploader_{st.session_state.uploader_key}"
        )
        if uploaded_files:
            images_to_process.extend(uploaded_files)

    # 탭 2: 카메라 촬영 기능
    with tab2:
        camera_image = st.camera_input(
            "카메라로 찰칵!",
            key=f"camera_{st.session_state.uploader_key}"
        )
        if camera_image:
            images_to_process.append(camera_image)

    # 이미지가 하나라도 있으면 분석 시작
    if images_to_process:
        st.divider()
        st.write(f"총 {len(images_to_process)}장의 이미지를 분석합니다.")
        
        # 모델 로드 (최초 1회만 실행되고 이후엔 캐시 사용)
        classifier = load_model()

        # 분석 결과를 저장할 딕셔너리 초기화 (없으면 생성)
        if 'analysis_results' not in st.session_state:
            st.session_state.analysis_results = {}

        # 각 이미지에 대해 반복 처리
        for i, img_file in enumerate(images_to_process):
            # 이미지 식별자 생성 (파일 이름 + 사이즈 등 조합)
            # file_uploader의 객체는 name 속성을 가짐, camera input은 불확실하므로 처리 필요
            if hasattr(img_file, 'name'):
                file_id = f"{img_file.name}_{img_file.size}"
            else:
                # 카메라 이미지는 이름이 없으므로 임의의 ID 부여 (주의: reruns 시 재생성될 수 있음)
                # 다만 여기선 간단히 바이트 길이로 구분하거나 인덱스 사용
                # 카메라 이미지는 매번 새로운 객체로 올 수 있어 캐싱이 까다롭지만,
                # 여기서는 '현재 리스트의 인덱스'와 결합하여 식별 시도
                file_id = f"camera_{st.session_state.uploader_key}_{i}"

            # 이미지 열기
            # seek(0)은 스트림 위치 초기화 (혹시 모를 에러 방지)
            if hasattr(img_file, 'seek'):
                img_file.seek(0)
            image = Image.open(img_file)
            
            # 레이아웃: 왼쪽(이미지), 오른쪽(분석 결과)으로 나누기
            col1, col2 = st.columns([1, 2])
            
            # 왼쪽 컬럼: 이미지 표시
            with col1:
                st.image(image, caption=f"이미지 #{i+1}", use_container_width=True)
            
            # 오른쪽 컬럼: 분석 결과 표시
            with col2:
                # 1. 이미 분석된 결과가 있는지 확인
                if file_id in st.session_state.analysis_results:
                    # 저장된 결과 불러오기
                    cached_data = st.session_state.analysis_results[file_id]
                    top_result = cached_data['top_result']
                    chart_data = cached_data['chart_data']
                    ko_label = cached_data['ko_label']
                    
                    # (캐시됨) 표시와 함께 결과 출력
                    en_label = top_result['label']
                    score = top_result['score']
                    emoji_str = get_emoji(en_label, score)
                    
                    st.subheader(f"{emoji_str} {ko_label}")
                    st.caption(f"({en_label}) - 신뢰도: {score*100:.2f}% (저장된 결과)")

                    # 저장된 데이터로 차트 그리기
                    df = pd.DataFrame(chart_data)
                    st.bar_chart(df.set_index("Class")['Confidence'], color="#FF4B4B", horizontal=True)
                    
                    with st.expander("상세 수치 보기"):
                         for item in chart_data:
                            # chart_data에 이미 원본 라벨 정보가 없으므로 간단히 출력
                            st.write(f"- **{item['Class']}**: {item['Confidence']*100:.2f}%")

                else:
                    # 2. 분석된 결과가 없으면 모델 실행
                    with st.spinner(f"이미지 #{i+1} 분석 중... (새로운 이미지)"):
                        # 모델 예측 수행 (상위 5개 결과 반환)
                        results = classifier(image, top_k=5)
                        
                        # 가장 높은 확률의 결과
                        top_result = results[0]
                        en_label = top_result['label']
                        score = top_result['score']
                        
                        # 라벨 번역
                        ko_label = translate_to_korean(en_label)

                        # 이모지 및 결과 출력
                        emoji_str = get_emoji(en_label, score)
                        st.subheader(f"{emoji_str} {ko_label}")
                        st.caption(f"({en_label}) - 신뢰도: {score*100:.2f}%")
                        
                        # 차트 데이터 준비
                        chart_data = []
                        for res in results:
                            translated = translate_to_korean(res['label'])
                            chart_data.append({
                                "Class": translated, 
                                "Confidence": res['score'],
                                "Original": res['label'] # 원본 라벨도 저장
                            })
                        
                        # 세션 스테이트에 결과 저장 (캐싱)
                        st.session_state.analysis_results[file_id] = {
                            'top_result': top_result,
                            'chart_data': chart_data,
                            'ko_label': ko_label
                        }
                        
                        # 차트 그리기
                        df = pd.DataFrame(chart_data)
                        st.bar_chart(df.set_index("Class")['Confidence'], color="#FF4B4B", horizontal=True)
                        
                        with st.expander("상세 수치 보기"):
                            for item in chart_data:
                                st.write(f"- **{item['Class']}** ({item['Original']}): {item['Confidence']*100:.2f}%")
            
            st.divider()  # 이미지 간 구분선

if __name__ == "__main__":
    main()
