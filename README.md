# 🛡️ SourceShield - AI News Analysis Tool

**AI-powered platform for detecting bias, analyzing sentiment, and classifying facts vs opinions in news articles**

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg) ![Streamlit](https://img.shields.io/badge/Streamlit-Web%20App-red.svg) ![OpenAI](https://img.shields.io/badge/OpenAI-GPT--3.5-green.svg) ![MongoDB](https://img.shields.io/badge/MongoDB-Database-darkgreen.svg)

<div align="center">
  <img src="https://github.com/user-attachments/assets/6dbb4a94-eb28-4980-bd78-7cee2924a61b" alt="SourceShield Interface" width="700"/>
</div>

---

## 🎯 What it Does

- **✅ Fact vs Opinion Classification** - Distinguishes factual statements from opinions
- **⚖️ Bias Detection** - Identifies political and emotional bias
- **📊 Sentiment Analysis** - Determines positive/negative tone
- **🤖 AI Insights** - Advanced analysis using OpenAI GPT
- **📈 Multi-source Comparison** - Compare different news outlets

---

## 📱 Screenshots

### Analysis Dashboard
<img src="https://github.com/user-attachments/assets/e0258625-2d39-4938-a4a5-bf29548beff9" alt="Overview Dashboard" width="600"/>

### Fact vs Opinion Results
<img src="https://github.com/user-attachments/assets/ae763db4-b931-4a8e-8aed-f360e0a9a0d2" alt="Fact Opinion Analysis" width="600"/>
<img width="1919" height="912" alt="Screenshot 2025-08-22 172506" src="https://github.com/user-attachments/assets/3e796e54-4726-4c92-9f48-7e11853c32b7" />

<img width="1919" height="902" alt="Screenshot 2025-08-22 172520" src="https://github.com/user-attachments/assets/e79c9dcd-1366-47d9-b8bb-9027b2793b0b" />




### Bias Detection
<img src="https://github.com/user-attachments/assets/052b8408-ac5c-4725-99b9-fa3e083a8de6" alt="Bias Analysis" width="600"/>

### AI Analysis
<img src="https://github.com/user-attachments/assets/3c8bf5f7-ccda-49d0-b810-eb95c96a7849" alt="AI Insights" width="600"/>
<img width="1915" height="891" alt="Screenshot 2025-08-22 172542" src="https://github.com/user-attachments/assets/d6f40dce-0407-40b7-84b7-15cbe08a494c" />

---

## 🚀 Quick Demo

**Input:** *"The GDP grew by 7.2% according to RBI data. This is amazing news for investors!"*

**Results:**
- **Fact vs Opinion:** 50% Factual, 50% Opinion
- **Sentiment:** Positive (78% confidence)
- **Political Bias:** Neutral
- **AI Analysis:** *"Factual data with positive editorial framing"*

---

## ⚡ Installation

### Prerequisites
- Python 3.8+
- MongoDB
- OpenAI API key (optional)

### Setup
```bash
# Clone repository
git clone https://github.com/ratnesh90859/SourceShield.git


# Create virtual environment
python -m venv venv
.\venv\Scripts\Activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt


```

### Configure Environment
Create `.env` file:
```env
OPENAI_API_KEY=sk-proj-your-api-key-here
MONGODB_URI=mongodb://localhost:27017/
```

### Run Application
```bash
streamlit run app/main.py
```
**Open:** http://localhost:8501

---

## 🔧 Tech Stack

| Component | Technology |
|-----------|------------|
| **Frontend** | Streamlit |
| **AI Analysis** | OpenAI GPT-3.5 |
| **NLP Models** | HuggingFace Transformers |
| **Database** | MongoDB |
| **Web Scraping** | Newspaper3k |
| **Similarity** | Sentence Transformers |

---

## 🎮 Usage

### Option 1: URL Analysis
1. Paste news article URL
2. Click "Analyze Article"
3. View results in 4 tabs

### Option 2: Direct Text
1. Copy-paste article content
2. Click "Analyze Text"
3. Get comprehensive analysis

### Option 3: Multi-Source Comparison
1. Add multiple news sources
2. Compare bias and perspectives
3. Identify factual consistency

---

## 📊 Features

- **🔍 Content Extraction** - Automatic article parsing
- **📈 Real-time Analysis** - Results in under 20 seconds
- **💾 Historical Storage** - MongoDB for trend tracking
- **📱 Interactive Dashboard** - Beautiful Streamlit interface
- **🤖 AI-Powered** - Advanced GPT-based insights
- **⚡ Parallel Processing** - Multiple analysis methods simultaneously

---





## 🎯 Use Cases

- **📰 Journalists** - Source credibility verification
- **🎓 Students** - Media literacy education
- **🔬 Researchers** - Content analysis studies
- **👥 General Public** - News fact-checking





---




