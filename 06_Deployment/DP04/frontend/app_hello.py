"""
Day 4 - 첫 번째 Streamlit 앱
"""
import streamlit as st

# 페이지 설정
st.set_page_config(
    page_title="My First Streamlit App",
    page_icon="🤖",
    layout="centered",
)

# 제목
st.title("🤖 나의 첫 Streamlit 앱")
st.write("Python 코드만으로 이 화면이 만들어졌습니다.")

# 구분선
st.divider()

# 텍스트 입력
name = st.text_input("이름을 입력하세요:", placeholder="홍길동")

# 조건부 출력
if name:
    st.success(f"안녕하세요, {name}님! 환영합니다. 🎉")
else:
    st.info("위에 이름을 입력해 보세요.")

# 버튼
if st.button("날짜 확인"):
    from datetime import datetime
    now = datetime.now().strftime("%Y년 %m월 %d일 %H시 %M분")
    st.write(f"현재 시각: {now}")
