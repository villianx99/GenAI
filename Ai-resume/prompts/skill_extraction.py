from langchain_core.prompts import ChatPromptTemplate

skill_extraction_prompt = ChatPromptTemplate.from_template("""
Extract structured info from resume.

Return ONLY JSON:
skills, experience, tools

Resume:
{resume}

Do NOT assume anything.
""")