#분류 결과 + 이미지 + 텍스트와 함께 분류 결과에 따라 다른 출력 보여주기
#파일 이름 streamlit_app.py
import streamlit as st
from fastai.vision.all import *
from PIL import Image
import gdown

# Google Drive 파일 ID
file_id = '1KSzzJDdxIQ39TBDTfyObI2m80QhY5nm-'

# Google Drive에서 파일 다운로드 함수
@st.cache(allow_output_mutation=True)
def load_model_from_drive(file_id):
    url = f'https://drive.google.com/uc?id={file_id}'
    output = 'model.pkl'
    gdown.download(url, output, quiet=False)

    # Fastai 모델 로드
    learner = load_learner(output)
    return learner

def display_left_content(image, prediction, probs, labels):
    st.write("### 왼쪽: 기존 출력 결과")
    if image is not None:
        st.image(image, caption="업로드된 이미지", use_container_width=True)
    st.write(f"예측된 클래스: {prediction}")
    st.markdown("<h4>클래스별 확률:</h4>", unsafe_allow_html=True)
    for label, prob in zip(labels, probs):
        st.markdown(f"""
            <div style="background-color: #f0f0f0; border-radius: 5px; padding: 5px; margin: 5px 0;">
                <strong style="color: #333;">{label}:</strong>
                <div style="background-color: #d3d3d3; border-radius: 5px; width: 100%; padding: 2px;">
                    <div style="background-color: #4CAF50; width: {prob*100}%; padding: 5px 0; border-radius: 5px; text-align: center; color: white;">
                        {prob:.4f}
                    </div>
                </div>
        """, unsafe_allow_html=True)

def display_right_content(prediction, data):
    st.write("### 오른쪽: 동적 분류 결과")
    cols = st.columns(3)

    # 1st Row - Images
    for i in range(3):
        with cols[i]:
            st.image(data['images'][i], caption=f"이미지: {prediction}", use_container_width=True)
    # 2nd Row - YouTube Videos
    for i in range(3):
        with cols[i]:
            st.video(data['videos'][i])
            st.caption(f"유튜브: {prediction}")
    # 3rd Row - Text
    for i in range(3):
        with cols[i]:
            st.write(data['texts'][i])

# 모델 로드
st.write("모델을 로드 중입니다. 잠시만 기다려주세요...")
learner = load_model_from_drive(file_id)
st.success("모델이 성공적으로 로드되었습니다!")

labels = learner.dls.vocab

# 스타일링을 통해 페이지 마진 줄이기
st.markdown("""
    <style>
    .reportview-container .main .block-container {
        max-width: 90%;
        padding-top: 1rem;
        padding-right: 1rem;
        padding-left: 1rem;
        padding-bottom: 1rem;
    }
    </style>
    """, unsafe_allow_html=True)

# 분류에 따라 다른 콘텐츠 관리
content_data = {
    labels[0]: {
        'images': [
            "https://i.ibb.co/ZgnbZ5f/5-Bl-YE2-B-48-Qm-DTXKOTHoc2l6-K42-Sg63apgc-F-O7l-Ngi-KEt-AOtae-jiu-Noe-JAy-Vsy-PFn2s333-Jshx-SJ-Mk-Y.webp",
            "https://i.ibb.co/0hbDp0n/images-1.jpg",
            "https://i.ibb.co/J2zVZmg/images-2.jpg"
        ],
        'videos': [
            "https://youtu.be/DAFxciBIIfI?feature=shared",
            "https://youtu.be/6zoxXfUTeug?feature=shared",
            "https://youtu.be/iYJByhrbea4?feature=shared"
        ],
        'texts': [
            "일본의 크리에이터 나가노(ナガノ) 작가의 캐릭터. https://twitter.com/ngntrtr 작가의 sns계정에서 일러스트와 만화를 볼 수 있다.",
            "농담곰과 함께 공부하기!",
            "비누로 농담곰 만들기!"
        ]
    },
    labels[1]: {
        'images': [
            "https://i.ibb.co/gg7sptV/llm6teb-R7xwtny-Xz-Wqp-CFKFmti-EL8-Fh-Ol-Dd1-Nzl-Vbdjtjd27roi-JMtqorl-RMHm6b-KSS5s-Jdr-V-3xm-Ggdhi-O.webp",
            "https://i.ibb.co/gtxGTrn/1.jpg",
            "https://i.ibb.co/7zw22XB/images-3.jpg"
        ],
        'videos': [
            "https://youtu.be/VbFBJ8XzmTs?feature=shared",
            "https://youtu.be/KuKYxjVw3V4?feature=shared",
            "https://youtu.be/eDpdmjELIhw?feature=shared"
        ],
        'texts': [
            "시나모롤은 먼 하늘 구름 위에서 태어난 강아지예요. 하늘에서 날아오던 시나모롤을 '카페 시나몬'의 주인 누나가 발견해 함께 살게 되었어요. 꼬리가 마치 시나몬롤처럼 돌돌 말려있어서 '시나몬' 이라고 이름이 붙여졌어요. 특기는 큰 귀를 파닥파닥 해서 하늘을 나는 일! 얌전하지만 붙임성이 좋아 손님들의 무릎 위에서 자버리기도 한답니다. (산리오 코리아 페이지의 시나모롤 소개글)",
            "시나모롤의 공식 유튜브의 애니메이션",
            "시나모롤 뮤직비디오도 있다! 이 노래는 세븐틴 노래의 커버곡."
        ]
    },
    labels[2]: {
        'images': [
            "https://i.ibb.co/d2T4cfk/Lh-Ic0-Y0-FXbwll-WQp-Yykfg-PZ8-Me1qa-nx21-Bs-HSZPs-Dn-HHsf-OVww-HFw-w-X8ba-Vb37-RIOuk-OHnbc-DEva0a8y.webp",
            "https://via.placeholder.com/300?text=Label3_Image2",
            "https://via.placeholder.com/300?text=Label3_Image3"
        ],
        'videos': [
            "https://www.youtube.com/watch?v=3JZ_D3ELwOQ",
            "https://www.youtube.com/watch?v=2Vv-BfVoq4g",
            "https://www.youtube.com/watch?v=3JZ_D3ELwOQ"
        ],
        'texts': [
            "치이카와는 먼작귀(먼가 작고 기여운 녀석의 줄임말)의 주요 캐릭터 3인방 중 하나이며 그 중에서도 주인공 격 캐릭터이다. 햄스터 캐릭터로, 처음에는 농담곰 계정에 명확한 배경 설정없이 뭔가 작고 귀여운 녀석이 되고 싶다는 소망을 담은 내용의 낙서(https://x.com/ngntrtr/status/859037354920624128)로 첫 등장했다.",
            "Label 3 관련 두 번째 텍스트 내용입니다.",
            "Label 3 관련 세 번째 텍스트 내용입니다."
        ]
    }
}

# 레이아웃 설정
left_column, right_column = st.columns([1, 2])  # 왼쪽과 오른쪽의 비율 조정

# 파일 업로드 컴포넌트 (jpg, png, jpeg, webp, tiff 지원)
uploaded_file = st.file_uploader("이미지를 업로드하세요", type=["jpg", "png", "jpeg", "webp", "tiff"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    img = PILImage.create(uploaded_file)
    prediction, _, probs = learner.predict(img)

    with left_column:
        display_left_content(image, prediction, probs, labels)

    with right_column:
        # 분류 결과에 따른 콘텐츠 선택
        data = content_data.get(prediction, {
            'images': ["https://via.placeholder.com/300"] * 3,
            'videos': ["https://www.youtube.com/watch?v=3JZ_D3ELwOQ"] * 3,
            'texts': ["기본 텍스트"] * 3
        })
        display_right_content(prediction, data)

