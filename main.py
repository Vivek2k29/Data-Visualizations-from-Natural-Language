import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.io as pio
import re
from google import genai
from google.genai import types
from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="AI Data Analyst Pro",
    layout="wide",
    page_icon="🤖"
)

# ---------------- THEME STATE ----------------
if "theme" not in st.session_state:
    st.session_state.theme = "dark"

# ---------------- HEADER ----------------
col1, col2 = st.columns([9, 1])
with col1:
    st.title("📊 Data Visualizations from Natural Language")
with col2:
    if st.button("🌓"):
        st.session_state.theme = (
            "light" if st.session_state.theme == "dark" else "dark"
        )

# ---------------- CSS ----------------
if st.session_state.theme == "light":
    st.markdown("""
    <style>
    .stApp {
        background-color: #ffffff !important;
        color: #000000 !important;
    }

    html, body, p, span, label, div {
        color: #000000 !important;
    }

    h1,h2,h3,h4,h5,h6 {
        color: #000000 !important;
    }

    section[data-testid="stSidebar"] {
        background-color: #f4f6f9 !important;
    }

    section[data-testid="stSidebar"] * {
        color: #000000 !important;
    }

    div[data-testid="stFileUploader"] {
        background-color: #ffffff !important;
        border: 1px solid #cccccc !important;
        border-radius: 12px !important;
        padding: 12px !important;
    }

    div[data-testid="stFileUploader"] section {
        background-color: #ffffff !important;
    }

    div[data-testid="stFileUploader"] span {
        color: #000000 !important;
    }

    div[data-testid="stFileUploader"] button {
        background-color: #ffffff !important;
        color: #000000 !important;
        border: 1px solid #999 !important;
        border-radius: 8px !important;
    }

    input, textarea {
        background-color: #ffffff !important;
        color: #000000 !important;
        border: 1px solid #cccccc !important;
    }

    div[data-testid="metric-container"] {
        background-color: #ffffff !important;
        border: 1px solid #dddddd !important;
        border-radius: 10px !important;
        padding: 10px !important;
    }

    .stDownloadButton button {
        background-color: #000000 !important;
        color: #ffffff !important;
        border-radius: 8px !important;
    }
    </style>
    """, unsafe_allow_html=True)

    plot_template = "plotly_white"

else:
    st.markdown("""
    <style>
    .stApp {
        background-color: #0E1117;
        color: white;
    }

    section[data-testid="stSidebar"] {
        background-color: #111111 !important;
    }

    button {
        background-color: #1f2937 !important;
        color: white !important;
    }
    </style>
    """, unsafe_allow_html=True)

    plot_template = "plotly_dark"

# ---------------- API KEY ----------------
try:
    API_KEY = st.secrets["API_KEY"]
    client = genai.Client(api_key=API_KEY)
except Exception:
    st.error("API Key not found")
    st.stop()

# ---------------- DATA LOADER ----------------
def load_data(file):
    name = file.name.lower()
    try:
        if name.endswith(".csv"):
            return pd.read_csv(file)
        elif name.endswith((".xls", ".xlsx")):
            return pd.read_excel(file)
        elif name.endswith(".pdf"):
            reader = PdfReader(file)
            text = " ".join([p.extract_text() for p in reader.pages[:3]])
            return pd.DataFrame({"Content": [text]})
    except Exception as e:
        st.error(e)
    return None

# ---------------- VECTOR STORE ----------------
class VectorStore:
    def __init__(self):
        self.vectorizer = TfidfVectorizer()
        self.documents = []
        self.vectors = None

    def index_documents(self, texts):
        self.documents = texts
        if texts:
            self.vectors = self.vectorizer.fit_transform(texts)

    def search(self, query, top_k=1):
        if not self.documents or self.vectors is None:
            return []
        qv = self.vectorizer.transform([query])
        sim = (self.vectors * qv.T).toarray()
        idx = np.argsort(sim.flatten())[::-1][:top_k]
        return [self.documents[i] for i in idx]

vector_db = VectorStore()

# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.header("Upload Center")
    uploaded_file = st.file_uploader(
        "Upload CSV, Excel, or PDF",
        type=["csv", "xlsx", "xls", "pdf"]
    )

# ---------------- MAIN ----------------
if uploaded_file:
    df = load_data(uploaded_file)

    if df is not None:
        try:
            sample_text = df.astype(str).apply(" ".join, axis=1).tolist()[:50]
            vector_db.index_documents(sample_text)
        except:
            pass

        st.subheader("📋 Dataset Overview")
        c1, c2, c3 = st.columns(3)
        c1.metric("Rows", df.shape[0])
        c2.metric("Columns", df.shape[1])
        c3.info(f"File: {uploaded_file.name}")

        with st.expander("👀 Preview Dataset"):
            st.dataframe(df.head(50), width="stretch")

        st.divider()

        query = st.text_input(
            "💬 What should I visualize?",
            placeholder="e.g. scatter plot of magnitude"
        )

        if query:
            cols = ", ".join(df.columns)
            prompt = f"""
Write ONLY Python Plotly Express code for: {query}
Dataframe name: df
Columns: {cols}

Rules:
- import plotly.express as px
- assign to fig
- use template='{plot_template}'
- don't call fig.show()
"""

            try:
                with st.spinner("Generating visualization..."):
                    response = client.models.generate_content(
                        model="gemini-3-flash-preview",
                        contents=prompt,
                        config=types.GenerateContentConfig(temperature=0)
                    )

                    raw = response.text
                    match = re.search(r"```(?:python)?(.*?)```", raw, re.DOTALL)
                    code = match.group(1).strip() if match else raw.strip()

                    local_scope = {"df": df, "px": px, "pd": pd}
                    exec(code, globals(), local_scope)
                    fig = local_scope.get("fig")

                    if fig:

                        # -------- FORCE WHITE BACKGROUND + BLACK TEXT --------
                        fig.update_layout(
                            plot_bgcolor="white",
                            paper_bgcolor="white",
                            font_color="black",
                            title_font_color="black",
                            xaxis=dict(
                                title_font=dict(color="black"),
                                tickfont=dict(color="black"),
                                gridcolor="lightgray"
                            ),
                            yaxis=dict(
                                title_font=dict(color="black"),
                                tickfont=dict(color="black"),
                                gridcolor="lightgray"
                            )
                        )

                        st.plotly_chart(fig, width="stretch")

                        img = pio.to_image(fig, format="png")
                        st.download_button(
                            "📥 Download PNG",
                            img,
                            file_name="chart.png",
                            mime="image/png"
                        )

            except Exception as e:
                st.error(e)

else:
    st.info("👆 Upload a dataset to get started")