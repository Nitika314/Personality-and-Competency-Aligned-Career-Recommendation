# Personality and Competency-Aligned Career Recommendation

A career recommendation engine using Machine Learning to match individuals with suitable career clusters based on comprehensive personality, skill, and interest assessments — enhanced with a **Retrieval-Augmented Generation (RAG) AI advisor** for personalized explanations and interactive career guidance.

## What Decision Does This System Support?

This system supports career path selection and planning decisions by helping individuals:

1. Identify suitable career clusters aligned with their personality and competencies
2. Explore career options they may not have considered based on their profile
3. Make informed career transitions by understanding which fields match their strengths
4. Get personalized, AI-generated answers to follow-up career questions

## Target Users

This system is designed for students, graduates, and career counselors or advisors to support data-driven career guidance based on personality, skills, and interests.

## Dataset

* **Size:** 7000+ records
* **Features:** 23+ attributes including personality traits, skills, interests, work preferences
* **Target:** 6 career clusters
* **Source:** Synthetically generated to reflect realistic distributions of personality traits, skills, and career outcomes

## Tech Stack

* **Frontend:** Streamlit
* **ML Model:** Random Forest Classifier (Scikit-learn)
* **AI Advisor / RAG:** Manually built retrieval pipeline (cosine similarity over embeddings) + locally hosted LLM via Ollama (`gemma3:1b` for generation, `nomic-embed-text` for embeddings) — no LangChain dependency
* **Data Processing:** Pandas, NumPy
* **Visualization:** Matplotlib, Seaborn
* **Model Serialization:** Joblib

## Features

* **Personalized Recommendations:** Based on Big Five personality traits, skills, and interests
* **Top 3 Career Matches:** Ranked recommendations with confidence scores
* **Ambiguous Profile Detection:** Flags low-confidence or flat-interest profiles and surfaces alternate matches
* **Background-Alignment Check:** Cross-checks predictions against expected alignments for a user's field of study and suggests overlooked paths
* **Interactive UI:** Built with Streamlit for easy user interaction
* **Role Suggestions:** Specific job roles based on education level and field of study
* **AI-Generated Decision Rationale:** RAG-powered, 4-point explanation grounded in the user's actual scores and a custom knowledge base
* **Conversational Career Advisor (RAG Chat):** Ask free-form or quick-select questions (skills to build, target companies, further studies, certifications, career switching, salary expectations) about any of the top 3 clusters, with context-aware answers retrieved from the knowledge base
* **Downloadable Reports:** Save recommendations, profile inputs, and AI explanations as text files

## Model Details

* **Algorithm:** Random Forest Classifier
* **Accuracy:** 83.24% (Test Set)
* **Hyperparameters:** ccp_alpha = 0.0005
* **Features:** 40+ engineered features
* **Classes:** 6 career clusters

## RAG Pipeline

* Custom-built (not LangChain) retrieval layer using cosine similarity over locally generated embeddings (`nomic-embed-text` via Ollama)
* Knowledge base pre-embedded and cached (`knowledge.pkl`), loaded once per session via `st.cache_resource`
* Two RAG-backed functions:
  * `explain_prediction()` — generates the personalized Decision Rationale bullets
  * `rag_chat()` — powers the interactive advisor chat, scoped to the user's profile and selected career cluster
* Graceful fallback responses if Ollama is unavailable

## Project Structure

```
├── app.py                    # Streamlit application
├── config.py                 # Career roles, info, skills, interests, mappings
├── career.ipynb              # Model training notebook
├── career.csv                # Training dataset
├── knowledge.pkl             # Pre-embedded knowledge base (chunks + embeddings)
├── *.joblib                  # Model artifacts (model, feature columns, risk mapping)
├── requirements.txt          # Dependencies
└── README.md                 # Documentation
```
