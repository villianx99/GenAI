from langchain_core.prompts import ChatPromptTemplate

scoring_prompt = ChatPromptTemplate.from_template("""
You are an AI recruiter.

Compare resume with job description.

Return JSON:
- score (0-100)
- reasoning

Job:
{job}

Resume:
{resume}

IMPORTANT:
Do not assume any skill not present in resume.
""")