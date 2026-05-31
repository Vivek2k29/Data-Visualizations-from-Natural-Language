# 📊 Data Visualizations from Natural Language

An AI-powered data analytics application that transforms natural language queries into interactive visualizations using Google Gemini and Plotly.

## 🚀 Live Demo

https://data-visualizations-from-natural-languagegit.streamlit.app/

---

## ✨ Features

### 📂 Multi-Format Data Upload

* CSV files
* Excel files (.xlsx, .xls)
* PDF files

### 🧹 Automatic Data Cleaning

* Duplicate removal
* Missing value handling using KNN Imputer
* Categorical data encoding
* Dataset quality reporting

### ⏳ Time-Series Detection

* Automatic datetime column detection
* Time-series dataset recognition
* Trend analysis support
* Line chart recommendations

### 📊 AI-Powered Visualization Generation

Ask questions in natural language such as:

* Show sales trend over time
* Create a histogram of age
* Scatter plot of income vs expenses
* Bar chart of category and revenue

The application automatically generates Plotly visualizations using Google Gemini AI.

### 💡 Visualization Suggestions

The system automatically recommends:

* Scatter plots
* Line charts
* Bar charts
* Histograms
* Count plots

### 📈 Interactive Charts

* Zoom and pan
* Hover information
* Responsive design
* Download charts as PNG

### 🌙 Modern UI

* Dark mode
* Light mode
* Responsive dashboard

---

## 🛠️ Tech Stack

### Frontend

* Streamlit

### Data Processing

* Pandas
* NumPy
* Scikit-learn

### Visualization

* Plotly Express
* Kaleido

### AI Integration

* Google Gemini API

### File Processing

* PyPDF
* OpenPyXL

---

## 📂 Project Structure

```text
GenVis/
│
├── datasets/
├── .streamlit/
│   └── secrets.toml
├── main.py
├── requirements.txt
├── README.md
└── .gitignore
```

---

## ⚙️ Installation

```bash
git clone https://github.com/Vivek2k29/Data-Visualizations-from-Natural-Language.git

cd Data-Visualizations-from-Natural-Language

pip install -r requirements.txt
```

---

## 🔑 Environment Setup

Create:

```text
.streamlit/secrets.toml
```

Add:

```toml
API_KEY="YOUR_GEMINI_API_KEY"
```

---

## ▶️ Run Locally

```bash
streamlit run main.py
```

---

## 🚀 Deployment

This project is deployed using Streamlit Community Cloud.

Live Application:

https://data-visualizations-from-natural-languagegit.streamlit.app/

---

## 👨‍💻 Author

Vivek Arasam

B.Tech | AI & Data Analytics Enthusiast

GitHub:
https://github.com/Vivek2k29
