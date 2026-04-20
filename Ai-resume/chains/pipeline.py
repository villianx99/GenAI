from langchain_groq import ChatGroq
import os

from prompts.skill_extraction import skill_extraction_prompt
from prompts.scoring import scoring_prompt
from prompts.explanation import explanation_prompt

llm = ChatGroq(
    api_key=os.getenv("GROQ_API_KEY"),
    model="llama-3.1-70b-versatile"
)

skill_chain = skill_extraction_prompt | llm
score_chain = scoring_prompt | llm
explain_chain = explanation_prompt | llm