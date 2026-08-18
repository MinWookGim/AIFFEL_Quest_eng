"""
미션 2 - st.cache_resource 가 있을 때와 없을 때 차이를 재는 실험실

노트북에는 이 데코레이터가 마크다운 설명에만 나와서, 직접 만들어 재보려고 쓴 앱이다.
설계는 _scratch/06_Deployment/DP04/build/미션2_캐시실험_설계.md 에 적어뒀다.

실행:
  .venv_checkpoint/bin/python -m streamlit run frontend/app_cache_lab.py --server.port 8506
"""
import os
import sys
import time

import streamlit as st

# 이 파일은 frontend/ 안에 있는데 모델 코드는 app/ 에 있다.
# Streamlit 은 스크립트가 있는 폴더만 import 경로에 넣어주므로, 프로젝트 루트를 직접 넣어준다.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

MODEL_PATH = os.path.join(ROOT, "models", "mnist_state_dict.pth")
DATA_DIR = os.path.join(ROOT, "data")

st.set_page_config(page_title="캐시 실험실", page_icon="🧪", layout="wide")


# ============================================================
# 0. 카운터를 어디에 둘 것인가
# ============================================================
# Streamlit 은 스크립트를 통째로 다시 실행하니까, 그냥 전역 변수에 카운터를 두면
# 재실행할 때마다 0으로 돌아가서 셀 수가 없다.
# 그래서 카운터 자체를 cache_resource 에 담는다. 실험 도구가 실험 대상을 쓰는 모양이 된다.
@st.cache_resource
def get_counters():
    """앱 프로세스가 살아 있는 동안 계속 남는 상자. 모든 탭이 이걸 같이 쓴다."""
    return {"init_calls": {}, "app_started": time.time()}


counters = get_counters()


# ============================================================
# 1. 진짜 초기화를 하는 함수 (실험 대상)
# ============================================================
def _really_build(kind: str):
    """실제로 무거운 일을 하는 함수. 이게 몇 번 불렸는지가 이 실험의 핵심 숫자다."""
    counters["init_calls"][kind] = counters["init_calls"].get(kind, 0) + 1

    if kind == "가벼움":
        # 노트북이 캐시하라고 시킨 바로 그것. API 클라이언트 하나 만들기.
        import requests
        return {"무엇": "requests.Session()", "객체": requests.Session()}

    if kind == "중간":
        # 모델 가중치를 디스크에서 읽어 올린다.
        import torch
        from app.model_utils import SimpleClassifier
        model = SimpleClassifier(10)
        model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu", weights_only=True))
        model.eval()
        return {"무엇": "MNIST 모델 로드", "객체": model}

    # 무거움 - 모델을 올린 다음 테스트셋 2000장을 실제로 추론해서 정확도까지 낸다.
    # 대시보드가 켜질 때 "이 모델 성능 요약"을 보여준다면 딱 이만큼의 일을 하게 된다.
    import torch
    from torch.utils.data import DataLoader, Subset
    from torchvision import datasets, transforms
    from app.model_utils import SimpleClassifier

    model = SimpleClassifier(10)
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu", weights_only=True))
    model.eval()

    tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    ds = datasets.MNIST(root=DATA_DIR, train=False, download=False, transform=tf)
    loader = DataLoader(Subset(ds, range(2000)), batch_size=256)

    맞은수 = 전체 = 0
    with torch.no_grad():
        for x, y in loader:
            맞은수 += (model(x).argmax(1) == y).sum().item()
            전체 += y.numel()
    return {"무엇": "테스트셋 2000장 추론", "객체": model, "정확도": 맞은수 / 전체}


# 데코레이터는 함수를 정의할 때 붙는 것이라, 화면에서 켜고 끌 수가 없다.
# 그래서 같은 일을 하는 함수를 두 벌 만들어 두고, 화면에서 어느 쪽을 부를지 고른다.
@st.cache_resource
def build_cached(kind: str):
    """캐시가 걸린 쪽. 같은 kind 로는 앱이 살아 있는 동안 딱 한 번만 실제로 실행된다."""
    return _really_build(kind)


def build_uncached(kind: str):
    """캐시가 없는 쪽. 재실행할 때마다 매번 실제로 실행된다."""
    return _really_build(kind)


# ============================================================
# 2. 실험 C 용 - cache_data 와 cache_resource 는 뭐가 다른가
# ============================================================
@st.cache_resource
def 리스트_resource():
    """돌려주는 것: 만들어 둔 그 객체 자체"""
    return [1, 2, 3]


@st.cache_data
def 리스트_data():
    """돌려주는 것: 저장해 둔 값의 복사본"""
    return [1, 2, 3]


# ============================================================
# 3. 이 탭에서 스크립트가 몇 번 재실행됐는지 (탭마다 따로)
# ============================================================
if "reruns" not in st.session_state:
    st.session_state["reruns"] = 0
    st.session_state["기록"] = []
st.session_state["reruns"] += 1


# ============================================================
# 4. 화면
# ============================================================
with st.sidebar:
    st.header("실험 설정")

    kind = st.radio(
        "초기화를 얼마나 무겁게 할까",
        ["가벼움", "중간", "무거움"],
        captions=[
            "requests.Session() 만들기",
            "MNIST 모델 로드",
            "모델 로드 + 2000장 추론",
        ],
    )

    use_cache = st.toggle("st.cache_resource 켜기", value=False)

    st.divider()
    if st.button("캐시 비우기", use_container_width=True):
        # 캐시에 들어 있는 것도, 지금까지 센 횟수도 같이 지운다
        build_cached.clear()
        get_counters.clear()
        st.session_state["기록"] = []
        st.rerun()

    if st.button("이 탭 재실행 횟수만 초기화", use_container_width=True):
        st.session_state["reruns"] = 0
        st.session_state["기록"] = []
        st.rerun()

    st.divider()
    st.caption(f"앱이 뜬 지 {time.time() - counters['app_started']:.0f}초")


st.title("st.cache_resource 실험실")
st.write(
    "아래 '다시 실행' 버튼을 20번쯤 눌러보고, 사이드바에서 캐시를 켰다 껐다 하면서 "
    "**초기화가 실제로 몇 번 일어나는지**를 본다."
)

# ----- 실험 대상을 실제로 부르고 시간을 잰다 -----
시작 = time.perf_counter()
결과 = build_cached(kind) if use_cache else build_uncached(kind)
걸린시간ms = (time.perf_counter() - 시작) * 1000

# 최근 기록 20개만 남긴다
st.session_state["기록"].append(
    {"재실행": st.session_state["reruns"], "무게": kind,
     "캐시": "켬" if use_cache else "끔", "걸린시간(ms)": round(걸린시간ms, 3)}
)
st.session_state["기록"] = st.session_state["기록"][-20:]

st.button("다시 실행", type="primary")   # 누르면 스크립트가 통째로 다시 돈다

c1, c2, c3 = st.columns(3)
with c1:
    st.metric("이 탭에서 재실행된 횟수", st.session_state["reruns"],
              help="st.session_state 에 담아서 센다. 탭마다 따로다.")
with c2:
    st.metric(f"'{kind}' 초기화가 실제로 실행된 횟수",
              counters["init_calls"].get(kind, 0),
              help="cache_resource 에 담아서 센다. 모든 탭이 같이 쓴다.")
with c3:
    st.metric("이번 재실행에서 걸린 시간", f"{걸린시간ms:.3f} ms")

if 결과.get("정확도") is not None:
    st.caption(f"(참고) 방금 계산한 정확도: {결과['정확도']:.4f}")

st.warning(
    "**첫 번째 값은 빼고 봐야 한다.** 각 무게를 처음 고를 때는 import 하는 시간이 같이 잡힌다. "
    "실제로 재보니 '중간'의 첫 호출은 3527ms 였는데 그중 대부분이 torch 를 처음 불러오는 시간이었고, "
    "두 번째부터는 5ms 언저리다. 이건 캐시가 아니라 파이썬이 알아서 해주는 것이라, "
    "캐시 효과로 읽으면 결론이 뒤집힌다.",
    icon=None,
)

st.divider()

왼, 오 = st.columns(2)

with 왼:
    st.subheader("최근 재실행 기록")
    st.dataframe(list(reversed(st.session_state["기록"])), use_container_width=True, hide_index=True)
    st.caption(
        "캐시를 끄면 '초기화 실행 횟수'가 재실행 횟수를 그대로 따라 올라가고, "
        "켜면 1에서 멈춘다. 시간은 흔들리지만 이 횟수는 안 흔들린다."
    )

with 오:
    st.subheader("cache_resource 와 cache_data 는 뭐가 다른가")
    st.write("같은 리스트를 돌려주는 함수 두 개를 부르고, 돌려받은 객체의 id 를 찍어본다.")
    st.code(
        f"id( 리스트_resource() ) = {id(리스트_resource())}\n"
        f"id( 리스트_data()     ) = {id(리스트_data())}",
        language="text",
    )
    st.caption(
        "다시 실행을 눌러 보면, resource 쪽 id 는 그대로인데 data 쪽 id 는 계속 바뀐다. "
        "resource 는 만들어 둔 객체를 그대로 주고, data 는 복사본을 준다."
    )

st.divider()
st.subheader("탭을 하나 더 열어보기")
st.write(
    "브라우저에서 같은 주소를 새 탭으로 열면, 왼쪽 '이 탭에서 재실행된 횟수'는 1부터 다시 시작하는데 "
    "가운데 '초기화가 실제로 실행된 횟수'는 그대로다. "
    "session_state 는 탭마다 따로인데 cache_resource 는 앱 전체가 같이 쓴다는 뜻이다."
)
