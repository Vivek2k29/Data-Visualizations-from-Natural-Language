import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.io as pio
import re
from google import genai
from google.genai import types
from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import KNNImputer
from sklearn.preprocessing import LabelEncoder
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
            "light" if st.session_state.theme == "dark"
            else "dark"
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

            text = " ".join(
                [p.extract_text() for p in reader.pages[:3]]
            )

            return pd.DataFrame({"Content": [text]})

    except Exception as e:

        st.error(e)

    return None

# ---------------- TIME SERIES DETECTION ----------------
def detect_time_series(df):

    datetime_columns = []

    for col in df.columns:

        try:

            converted = pd.to_datetime(
                df[col],
                errors='coerce'
            )

            if converted.notnull().sum() > len(df) * 0.5:

                df[col] = converted

                datetime_columns.append(col)

        except:
            pass

    return datetime_columns

# ---------------- DATA CLEANING ----------------
def clean_dataset(df):

    cleaning_report = []

    # remove duplicates
    duplicate_count = df.duplicated().sum()

    if duplicate_count > 0:

        df.drop_duplicates(inplace=True)

    # replace empty strings
    df.replace(r'^\s*$', np.nan, regex=True, inplace=True)

    numerical_cols = df.select_dtypes(
        include=['number']
    ).columns.tolist()

    categorical_cols = df.select_dtypes(
        include=['object']
    ).columns.tolist()

    encoded_df = df.copy()

    label_encoders = {}

    # encode categorical columns
    for col in categorical_cols:

        encoded_df[col] = encoded_df[col].astype(str)

        le = LabelEncoder()

        encoded_df[col] = le.fit_transform(
            encoded_df[col]
        )

        label_encoders[col] = le

    try:

        # KNN Imputer
        imputer = KNNImputer(n_neighbors=5)

        imputed_array = imputer.fit_transform(encoded_df)

        imputed_df = pd.DataFrame(
            imputed_array,
            columns=encoded_df.columns
        )

        # restore categorical values
        for col in categorical_cols:

            imputed_df[col] = np.round(
                imputed_df[col]
            ).astype(int)

            le = label_encoders[col]

            max_label = len(le.classes_) - 1

            imputed_df[col] = imputed_df[col].clip(
                0,
                max_label
            )

            imputed_df[col] = le.inverse_transform(
                imputed_df[col]
            )

        # restore numeric types
        for col in numerical_cols:

            imputed_df[col] = pd.to_numeric(
                imputed_df[col]
            )

        # cleaning report
        for col in df.columns:

            missing_before = df[col].isnull().sum()

            if missing_before > 0:

                cleaning_report.append({
                    "Column": col,
                    "Missing Fixed": missing_before,
                    "Method": "KNN Imputer"
                })

        return imputed_df, cleaning_report, duplicate_count

    except Exception as e:

        st.warning(f"Cleaning failed: {e}")

        return df, [], duplicate_count

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

    if uploaded_file:

        st.markdown("### ⚙️ Actions")

        show_summary = st.button("📊 Summary")

        show_suggestions = st.button("💡 Suggestions")

# ---------------- MAIN ----------------
if uploaded_file:

    df = load_data(uploaded_file)

    if df is not None:

        # -------- DETECT TIME SERIES --------
        datetime_columns = detect_time_series(df)

        dataset_type = (
            "Time Series Dataset"
            if len(datetime_columns) > 0
            else "Normal Dataset"
        )

        # -------- CLEAN DATASET --------
        with st.spinner(
            "Cleaning dataset and fixing missing values..."
        ):

            df, cleaning_report, duplicate_count = clean_dataset(df)

        # -------- VECTOR INDEX --------
        try:

            sample_text = df.astype(str).apply(
                " ".join,
                axis=1
            ).tolist()[:50]

            vector_db.index_documents(sample_text)

        except:
            pass

        # ---------------- OVERVIEW ----------------
        st.subheader("📋 Dataset Overview")

        c1, c2, c3, c4 = st.columns(4)

        c1.metric("Rows", df.shape[0])

        c2.metric("Columns", df.shape[1])

        c3.metric("Duplicates Removed", duplicate_count)

        c4.info(dataset_type)

        # -------- TIME SERIES INFO --------
        if dataset_type == "Time Series Dataset":

            st.info(f"""
⏳ Time Series Dataset Detected

Detected datetime columns:
{', '.join(datetime_columns)}

Recommended charts:
- Trend Analysis
- Line Charts
- Seasonal Graphs
            """)

        # -------- CLEANING REPORT --------
        if len(cleaning_report) > 0:

            st.subheader("🛠️ Data Cleaning Report")

            st.dataframe(
                pd.DataFrame(cleaning_report),
                width="stretch"
            )

        with st.expander("👀 Preview Dataset"):

            st.dataframe(
                df.head(50),
                width="stretch"
            )

        # -------- SUMMARY --------
        if 'show_summary' in locals() and show_summary:

            st.subheader("📊 Dataset Summary")

            st.write("Shape:", df.shape)

            st.write("Columns:", list(df.columns))

            st.write("### Data Types")

            st.write(df.dtypes)

            st.write("### Statistical Summary")

            st.write(df.describe(include='all'))

        # -------- SUGGESTIONS --------
        if 'show_suggestions' in locals() and show_suggestions:

            st.subheader("💡 Visualization Suggestions")

            numeric_cols = df.select_dtypes(
                include=np.number
            ).columns.tolist()

            cat_cols = df.select_dtypes(
                exclude=np.number
            ).columns.tolist()

            suggestions = []

            if len(numeric_cols) >= 2:

                suggestions.append(
                    f"Scatter plot of {numeric_cols[0]} vs {numeric_cols[1]}"
                )

                suggestions.append(
                    f"Line chart of {numeric_cols[0]} over index"
                )

            if len(cat_cols) >= 1 and len(numeric_cols) >= 1:

                suggestions.append(
                    f"Bar chart of {cat_cols[0]} vs {numeric_cols[0]}"
                )

            if len(cat_cols) >= 1:

                suggestions.append(
                    f"Count plot of {cat_cols[0]}"
                )

            if len(numeric_cols) >= 1:

                suggestions.append(
                    f"Histogram of {numeric_cols[0]}"
                )

            suggestions = suggestions[:5]

            for i, s in enumerate(suggestions, 1):

                st.write(f"{i}. {s}")

        st.divider()

        # ---------------- QUERY ----------------
        query = st.text_input(
            "💬 What should I visualize?",
            placeholder="e.g. scatter plot of magnitude"
        )

        # ---------------- VISUALIZATION ----------------
        if query:

            cols = ", ".join(df.columns)

            prompt = f"""
Write ONLY Python Plotly Express code for: {query}

Dataset Type: {dataset_type}

Dataframe name: df

Columns: {cols}

Datetime Columns:
{datetime_columns}

Rules:
- import plotly.express as px
- assign chart to variable fig
- use template='{plot_template}'
- don't call fig.show()

If dataset is time-series:
- prefer line charts
- use datetime columns properly
- create trend visualizations

Generate proper professional visualization.
"""

            try:

                with st.spinner(
                    "Generating visualization..."
                ):

                    response = client.models.generate_content(
                        model="gemini-3-flash-preview",
                        contents=prompt,
                        config=types.GenerateContentConfig(
                            temperature=0
                        )
                    )

                    raw = response.text

                    match = re.search(
                        r"```(?:python)?(.*?)```",
                        raw,
                        re.DOTALL
                    )

                    code = (
                        match.group(1).strip()
                        if match
                        else raw.strip()
                    )

                    local_scope = {
                        "df": df,
                        "px": px,
                        "pd": pd
                    }

                    exec(
                        code,
                        globals(),
                        local_scope
                    )

                    fig = local_scope.get("fig")

                    if fig:

                        # -------- FIX TEXT COLORS --------
                        fig.update_layout(

                            plot_bgcolor="white",
                            paper_bgcolor="white",

                            font=dict(
                                color="black"
                            ),

                            title_font=dict(
                                color="black",
                                size=20
                            ),

                            xaxis=dict(
                                title_font=dict(color="black"),
                                tickfont=dict(color="black"),
                                gridcolor="lightgray"
                            ),

                            yaxis=dict(
                                title_font=dict(color="black"),
                                tickfont=dict(color="black"),
                                gridcolor="lightgray"
                            ),

                            legend=dict(
                                font=dict(color="black")
                            )
                        )

                        # -------- TOOLBAR CONFIG --------
                        config = {
                            "displaylogo": False,
                            "responsive": True
                        }

                        st.plotly_chart(
                            fig,
                            width="stretch",
                            config=config
                        )

                        # -------- DOWNLOAD --------
                        img = pio.to_image(
                            fig,
                            format="png"
                        )

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