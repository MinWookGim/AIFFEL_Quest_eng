"""
Day 4 - MNIST 추론 대시보드
FastAPI 백엔드와 연동하는 Streamlit 프론트엔드
"""
import streamlit as st
import requests
import base64
import io
from PIL import Image


# ===== 페이지 설정 =====
st.set_page_config(
    page_title="MNIST 숫자 인식",
    page_icon="🔢",
    layout="wide",
)


# ===== API 호출 함수 =====
API_BASE = "http://localhost:8000"

def call_api(url, json_data=None, method="post"):
    """API를 호출하고, 실패 시 에러 메시지를 표시합니다."""
    try:
        if method == "get":
            resp = requests.get(url, timeout=10)
        else:
            resp = requests.post(url, json=json_data, timeout=30)
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.ConnectionError:
        st.error("🔌 **서버에 연결할 수 없습니다.** FastAPI 서버가 실행 중인지 확인하세요.")
        return None
    except requests.exceptions.Timeout:
        st.warning("⏱️ **응답 시간 초과.** 잠시 후 다시 시도하세요.")
        return None
    except requests.exceptions.HTTPError as e:
        st.error(f"❌ **서버 에러** (HTTP {e.response.status_code})")
        return None
    except Exception as e:
        st.error(f"❌ **오류:** {type(e).__name__}")
        return None


# ===== 사이드바 =====
with st.sidebar:
    st.header("⚙️ 설정")

    # 서버 상태 표시
    health = call_api(f"{API_BASE}/health", method="get")
    if health and health.get("status") == "healthy":
        st.success("🟢 서버 연결됨")
        server_ok = True
    else:
        st.error("🔴 서버 연결 실패")
        server_ok = False

    st.divider()

    # 옵션
    show_probabilities = st.checkbox("전체 확률 표시", value=True)
    show_preprocessed = st.checkbox("전처리된 이미지 표시", value=True)

    st.divider()
    st.caption("MNIST Prediction Dashboard v1.0")


# ===== 메인 영역 =====
st.title("🔢 MNIST 숫자 인식")
st.write("손글씨 숫자 이미지를 업로드하면 0~9 중 어떤 숫자인지 예측합니다.")

col_input, col_result = st.columns(2)

# ----- 입력 영역 -----
with col_input:
    st.subheader("📤 이미지 입력")

    input_method = st.radio(
        "입력 방식:", ["파일 업로드", "샘플 이미지 사용"], horizontal=True,
    )

    image_bytes = None

    if input_method == "파일 업로드":
        # 샘플 모드에서 저장한 상태를 지웁니다 (업로드 이미지에 샘플 정답이 비교되는 것 방지)
        st.session_state.pop("sample_image_bytes", None)
        st.session_state.pop("sample_caption", None)
        st.session_state.pop("sample_label", None)
        uploaded = st.file_uploader(
            "이미지를 업로드하세요:",
            type=["png", "jpg", "jpeg"],
            help="28x28 그레이스케일 권장. 다른 크기도 자동 변환됩니다.",
        )
        if uploaded:
            image_bytes = uploaded.getvalue()
            st.image(uploaded, caption="업로드된 이미지", width=200)

    else:
        st.info("보고 싶은 숫자를 고르면, MNIST 테스트셋에서 그 숫자의 손글씨를 보여줍니다.")
        digit = st.number_input("보고 싶은 숫자 (0~9):", min_value=0, max_value=9, value=1)
        variant = st.number_input("같은 숫자의 몇 번째 샘플? (0~99):", min_value=0, max_value=99, value=0)

        if st.button("샘플 이미지 로드"):
            try:
                from torchvision import datasets
                test_dataset = datasets.MNIST(root="data", train=False, download=True)
                # digit인 샘플들의 위치를 찾아 variant번째를 뽑는다 — 같은 입력이면 항상 같은 이미지
                matches = (test_dataset.targets == digit).nonzero().flatten()
                idx = int(matches[variant])
                sample_image, sample_label = test_dataset[idx]

                buffer = io.BytesIO()
                sample_image.save(buffer, format="PNG")
                # 버튼 클릭은 "그 한 번의 재실행"에서만 True입니다.
                # 이미지를 session_state에 보관해야 '추론 실행' 클릭(또 다른 재실행) 후에도 남습니다.
                st.session_state["sample_image_bytes"] = buffer.getvalue()
                st.session_state["sample_caption"] = f"숫자 {digit} — {variant}번째 샘플 (테스트셋 #{idx})"
                st.session_state["sample_label"] = sample_label
                st.session_state.pop("last_result", None)   # 이전 추론 결과가 새 샘플과 섞여 보이지 않게
            except Exception as e:
                st.error(f"샘플 로드 실패: {e}")

        # 저장된 샘플 이미지를 매 재실행마다 다시 표시합니다.
        if "sample_image_bytes" in st.session_state:
            image_bytes = st.session_state["sample_image_bytes"]
            st.image(image_bytes, caption=st.session_state["sample_caption"], width=200)

    # 전처리된 이미지 미리보기
    if image_bytes and show_preprocessed:
        st.caption("전처리된 이미지 (28x28 그레이스케일):")
        img = Image.open(io.BytesIO(image_bytes)).convert("L").resize((28, 28))
        st.image(img, width=150)


# ----- 결과 영역 -----
with col_result:
    st.subheader("📊 추론 결과")

    if image_bytes is None:
        st.info("👈 왼쪽에서 이미지를 업로드하거나 샘플을 선택하세요.")

    elif not server_ok:
        st.error("서버에 연결할 수 없습니다. 사이드바의 서버 상태를 확인하세요.")

    else:
        if st.button("🚀 추론 실행", type="primary", use_container_width=True):
            with st.spinner("모델 추론 중..."):
                # Base64 인코딩 → API 호출
                image_base64 = base64.b64encode(image_bytes).decode("utf-8")
                result = call_api(
                    f"{API_BASE}/predict/image",
                    json_data={
                        "image_base64": image_base64,
                        "return_probabilities": show_probabilities,
                    },
                )

            if result:
                st.session_state["last_result"] = result

        # 결과 표시
        if "last_result" in st.session_state:
            result = st.session_state["last_result"]

            # 메트릭
            m1, m2 = st.columns(2)
            with m1:
                st.metric(label="예측 결과", value=result["predicted_class"])
            with m2:
                st.metric(label="확신도", value=f"{result['confidence']:.1%}")

            # 확률 분포
            if result.get("probabilities"):
                st.subheader("📊 클래스별 확률 분포")
                probs = result["probabilities"]
                for cls in sorted(probs.keys(), key=lambda x: int(x)):
                    prob = probs[cls]
                    c1, c2 = st.columns([1, 5])
                    with c1:
                        is_pred = cls == result["predicted_class"]
                        st.write(f"**{'👉 ' if is_pred else ''}{cls}**")
                    with c2:
                        st.progress(float(prob), text=f"{prob:.2%}")

            # 샘플 이미지인 경우 정답 비교
            if "sample_label" in st.session_state:
                label = st.session_state["sample_label"]
                if result["predicted_class"] == str(label):
                    st.success(f"✅ 정답! (정답: {label})")
                else:
                    st.error(f"❌ 오답 (정답: {label}, 예측: {result['predicted_class']})")
