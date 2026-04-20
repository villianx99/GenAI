from langchain_core.prompts import ChatPromptTemplate

explanation_prompt = ChatPromptTemplate.from_template("""
Explain score clearly.

Score: {score}
Resume: {resume}
Job: {job}

Give 4-6 lines explanation.
""")