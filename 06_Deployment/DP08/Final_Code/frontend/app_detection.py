"""
Day 8 - 객체 탐지 대시보드

이미지를 올리면 API 를 불러 찾은 물체를 상자로 그려 보여준다.
Day 4~5 의 대시보드와 구조는 같고, 결과가 숫자 하나가 아니라 목록이라
표와 그림 두 가지로 보여주는 점이 다르다.
"""
import io

import pandas as pd
import requests
import streamlit as st
from PIL import Image, ImageDraw

st.set_page_config(page_title="객체 탐지", page_icon="🔍", layout="wide")

API_BASE = "http://localhost:8509"   # 8000 은 Day5 집값 서버가 쓰는 중이라 포트를 옮겼다

# 상자 색을 라벨마다 다르게 주려고 미리 몇 가지 뽑아둔다.
COLORS = ["#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4",
          "#42d4f4", "#f032e6", "#bfef45", "#469990", "#9a6324"]


def call_detect_api(file_bytes, filename, mime, api_key, threshold):
    """탐지 API 를 부른다. 실패하면 화면에 이유를 띄우고 None 을 돌려준다."""
    try:
        resp = requests.post(
            f"{API_BASE}/predict",
            files={"file": (filename, file_bytes, mime)},
            params={"threshold": threshold},
            headers={"X-API-Key": api_key},
            timeout=120,
        )
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.ConnectionError:
        st.error("서버에 연결할 수 없습니다. 8509 포트에 서버가 떠 있는지 확인하세요.")
    except requests.exceptions.HTTPError as e:
        code = e.response.status_code
        if code == 401:
            st.error("인증 실패. API Key 를 확인하세요.")
        elif code == 400:
            st.error(f"입력이 거부됐습니다: {e.response.json().get('detail', '')}")
        else:
            st.error(f"서버 에러 (HTTP {code})")
    except Exception as e:
        st.error(f"오류: {type(e).__name__}")
    return None


def draw_boxes(image, detections):
    """찾은 물체를 원본 이미지 위에 상자로 그린다."""
    canvas = image.copy()
    draw = ImageDraw.Draw(canvas)
    # 이미지가 크든 작든 선이 보이게 두께를 크기에 맞춰 정한다.
    width = max(2, int(min(canvas.size) / 200))
    for i, d in enumerate(detections):
        b = d["box"]
        color = COLORS[i % len(COLORS)]
        draw.rectangle([b["x_min"], b["y_min"], b["x_max"], b["y_max"]], outline=color, width=width)
        # 이름과 확신도를 상자 왼쪽 위에 적는다. 글자가 이미지 밖으로 나가지 않게 눌러준다.
        text = f'{d["label"]} {d["score"]:.2f}'
        ty = max(0, b["y_min"] - 14)
        draw.rectangle([b["x_min"], ty, b["x_min"] + 8 * len(text), ty + 14], fill=color)
        draw.text((b["x_min"] + 2, ty + 1), text, fill="white")
    return canvas


with st.sidebar:
    st.header("설정")
    api_key = st.text_input("API Key", value="test-key-001", type="password")
    threshold = st.slider("확신도 기준값", 0.0, 1.0, 0.5, step=0.05,
                          help="이 값보다 확신도가 낮은 물체는 버린다")
    st.divider()
    try:
        h = requests.get(f"{API_BASE}/health", timeout=3).json()
        if h.get("status") == "healthy":
            st.success("서버 연결됨")
            st.caption(f"모델: {h.get('model')}")
            st.caption(f"장치: {h.get('device')} / 라벨 {h.get('num_labels')}종")
        else:
            st.warning("모델 로딩 중")
    except Exception:
        st.error("서버 연결 실패")
    st.divider()
    st.caption("Object Detection Service v1.0")

st.title("🔍 이미지에서 물체 찾기")
st.write("이미지를 올리면 무엇이 어디에 있는지 찾아 표시한다. PNG 또는 JPEG, 5MB 까지.")

uploaded = st.file_uploader("이미지 올리기", type=["png", "jpg", "jpeg"])

if uploaded is not None:
    file_bytes = uploaded.getvalue()
    image = Image.open(io.BytesIO(file_bytes)).convert("RGB")

    with st.spinner("찾는 중..."):
        result = call_detect_api(file_bytes, uploaded.name, uploaded.type, api_key, threshold)

    if result and result.get("success"):
        left, right = st.columns([3, 2])
        with left:
            st.subheader("결과")
            st.image(draw_boxes(image, result["detections"]), use_column_width=True)
        with right:
            st.subheader("찾은 것")
            c1, c2 = st.columns(2)
            c1.metric("물체 수", result["count"])
            c2.metric("걸린 시간", f'{result["elapsed_ms"]:.0f} ms')
            if result["detections"]:
                df = pd.DataFrame([
                    {"물체": d["label"], "확신도": round(d["score"], 3),
                     "위치(x1,y1,x2,y2)": f'{d["box"]["x_min"]:.0f}, {d["box"]["y_min"]:.0f}, '
                                          f'{d["box"]["x_max"]:.0f}, {d["box"]["y_max"]:.0f}'}
                    for d in result["detections"]
                ])
                st.dataframe(df, use_container_width=True, hide_index=True)
            else:
                st.info("기준값 이상으로 확신하는 물체가 없다. 사이드바에서 기준값을 낮춰보자.")
            st.caption(f'이미지 {result["image_size"][0]}x{result["image_size"][1]} · '
                       f'기준값 {result["threshold"]} · 사용자 {result["user"]}')
    else:
        st.image(image, caption="올린 이미지", width=400)
else:
    st.info("이미지를 올리면 결과가 여기에 나온다.")
