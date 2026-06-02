import streamlit as st
from ultralytics import YOLO
from PIL import Image
from openai import OpenAI
from dotenv import load_dotenv
import numpy as np
import os

# RAG 관련 임포트
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

# ----------------------------------------------------------------
# 1. 환경 설정 및 API 키 로드
# ----------------------------------------------------------------
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# OpenAI 클라이언트 초기화
client = OpenAI(api_key=OPENAI_API_KEY)

# ----------------------------------------------------------------
# 2. 페이지 설정 및 스타일
# ----------------------------------------------------------------
st.set_page_config(
    page_title="성남시 AI 쓰레기 분리배출 가이드 (RAG)",
    page_icon="♻️",
    layout="centered"
)

st.title("♻️ 성남시 AI 쓰레기 분리배출 가이드")
st.markdown("""
이미지를 업로드하면 **YOLOv8**이 쓰레기를 탐지하고, 
**성남시 자원순환 가이드(PDF)**를 기반으로 **ChatGPT**가 정확한 방법을 안내합니다.
""")

# ----------------------------------------------------------------
# 3. 모델 및 RAG 데이터 로드 (캐싱 적용)
# ----------------------------------------------------------------
@st.cache_resource
def load_yolo_model():
    model_path = "best.pt" 
    if not os.path.exists(model_path):
        st.error(f"모델 파일('{model_path}')을 찾을 수 없습니다.")
        return None
    return YOLO(model_path)

@st.cache_resource
def get_vector_store():
    pdf_path = "재활용품 분리배출 안내 _ 자원순환 정보 _ 성남시 자원순환 통합 플랫폼.pdf"
    if not os.path.exists(pdf_path):
        st.error(f"PDF 가이드 파일('{pdf_path}')을 찾을 수 없습니다.")
        return None
    
    try:
        # PDF 로드 및 텍스트 분할
        loader = PyPDFLoader(pdf_path)
        docs = loader.load()
        
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        split_docs = text_splitter.split_documents(docs)
        
        # 임베딩 및 벡터 저장소(FAISS) 생성
        embeddings = OpenAIEmbeddings()
        vectorstore = FAISS.from_documents(split_docs, embeddings)
        return vectorstore
    except Exception as e:
        st.error(f"RAG 데이터베이스 생성 중 오류 발생: {e}")
        return None

yolo_model = load_yolo_model()
vector_store = get_vector_store()

# ----------------------------------------------------------------
# 4. LLM 가이드 생성 함수 (RAG 적용)
# ----------------------------------------------------------------
def get_llm_disposal_guide(detected_items, vectorstore):
    try:
        items_str = ", ".join(detected_items)
        
        # RAG: 관련 컨텍스트 검색
        context = ""
        if vectorstore:
            # 탐지된 모든 품목에 대해 관련 정보 검색
            query = f"{items_str} 분리배출 방법"
            relevant_docs = vectorstore.similarity_search(query, k=3)
            context = "\n\n".join([doc.page_content for doc in relevant_docs])

        prompt = f"""
        당신은 대한민국 성남시의 환경 보호 및 쓰레기 분리수거 전문가입니다.
        AI 모델이 사진에서 다음 품목들을 탐지했습니다: {items_str}
        
        아래 제공된 [성남시 공식 가이드 내용]을 최우선으로 참고하여, 사용자가 어떻게 버려야 하는지 안내해주세요.
        
        [성남시 공식 가이드 내용]:
        {context if context else "가이드에서 관련 내용을 찾을 수 없습니다. 일반적인 원칙을 안내해주세요."}
        
        형식 지침:
        - 각 품목은 `### ♻️ 품목명` 형식의 헤더로 시작하세요.
        - 제공된 가이드 내용을 바탕으로 '성남시 기준' 배출 방법을 상세히 설명하세요.
        - 주의사항은 `⚠️ 주의:`, 폐기 방법은 `🗑️ 폐기:` 로 시작하세요.
        - 반드시 한국어로 답변해주세요.
        """
        
        response = client.chat.completions.create(
            model="gpt-4o-mini", # 더 효율적인 모델 사용
            messages=[
                {"role": "system", "content": "당신은 성남시 자원순환 가이드를 숙지한 전문가입니다."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"LLM 가이드 생성 중 오류가 발생했습니다: {str(e)}"

# ----------------------------------------------------------------
# 5. 메인 UI 및 로직
# ----------------------------------------------------------------
st.sidebar.title("🛠️ 시스템 정보")
st.sidebar.info("성남시 공식 PDF 가이드를 기반으로 한 RAG(Retrieval-Augmented Generation) 시스템이 적용되었습니다.")
if vector_store:
    st.sidebar.success("✅ 성남시 가이드 DB 로드 완료")
else:
    st.sidebar.error("❌ 가이드 DB 로드 실패")

uploaded_file = st.file_uploader("📸 쓰레기 사진을 업로드하세요 (JPG, PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    col1, col2 = st.columns(2)
    image = Image.open(uploaded_file)
    
    with col1:
        st.subheader("원본 이미지")
        st.image(image, use_container_width=True)
    
    if st.button("✨ 성남시 기준 분석 시작", use_container_width=True, type="primary"):
        if yolo_model is None:
            st.error("모델이 로드되지 않았습니다.")
        else:
            with st.spinner("성남시 가이드를 확인하며 분석 중입니다..."):
                results = yolo_model(image, conf=0.4) # 신뢰도 임계값 적용
                
                boxes = results[0].boxes
                if len(boxes) > 0:
                    found_classes = [yolo_model.names[int(box.cls[0])] for box in boxes]
                    unique_items = list(set(found_classes))
                    
                    with col2:
                        st.subheader("AI 탐지 결과")
                        res_plotted = results[0].plot()
                        res_image = Image.fromarray(res_plotted[:, :, ::-1])
                        st.image(res_image, use_container_width=True)
                    
                    st.divider()
                    st.success(f"✅ 탐지된 품목: **{', '.join(unique_items).upper()}**")
                    st.subheader("📝 성남시 공식 분리배출 가이드")
                    
                    if not OPENAI_API_KEY:
                        st.warning("API 키가 없어 상세 가이드를 생성할 수 없습니다.")
                    else:
                        guide_text = get_llm_disposal_guide(unique_items, vector_store)
                        st.markdown(guide_text)
                else:
                    with col2:
                        st.warning("쓰레기를 탐지하지 못했습니다.")

st.divider()
st.caption("🚀 기계학습 프로젝트 - 성남시 자원순환 RAG 시스템")
