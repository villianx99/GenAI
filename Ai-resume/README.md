# 🧠 AI Resume Screening System with LangChain + LangSmith

## 📌 Project Overview
This project is an AI-powered Resume Screening System that evaluates candidates based on a job description. It extracts skills, compares them with job requirements, assigns a score (0–100), and provides a detailed explanation using LLMs.

---

## 🚀 Features
- Resume Skill Extraction
- Experience & Tools Identification
- Job Matching Logic
- AI-based Scoring System (0–100)
- Explainable AI Output
- LangSmith Tracing for Debugging and Monitoring

---

## 🛠️ Tech Stack
- Python
- LangChain
- Groq LLM (Llama 3)
- LangSmith
- dotenv

---

## 🔄 Pipeline Flow
Resume → Skill Extraction → Matching → Scoring → Explanation → Output

---

## 📂 Project Structure
prompts/
- skill_extraction.py
- scoring.py
- explanation.py

utils/
- loader.py

data/
- strong.txt
- average.txt
- weak.txt
- job.txt

main.py  
requirements.txt  

---

## ⚙️ How to Run

```bash
pip install -r requirements.txt
python main.py