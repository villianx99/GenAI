import os
import json
from dotenv import load_dotenv

from utils.loader import load_text
from langchain_groq import ChatGroq

from prompts.skill_extraction import skill_extraction_prompt
from prompts.scoring import scoring_prompt
from prompts.explanation import explanation_prompt

# Load environment variables
load_dotenv()

# -----------------------------
# Groq LLM setup
# -----------------------------
llm = ChatGroq(
    api_key=os.getenv("GROQ_API_KEY"),
    model="llama-3.3-70b-versatile"
)

# -----------------------------
# Helper function (clean output)
# -----------------------------
def get_response(chain, inputs):
    response = chain.invoke(inputs)
    return response.content


# -----------------------------
# Main Pipeline
# -----------------------------
def run(resume, job):

    # STEP 1: Skill Extraction
    skill_chain = skill_extraction_prompt | llm
    skills = get_response(skill_chain, {"resume": resume})

    # STEP 2: Scoring
    score_chain = scoring_prompt | llm
    score = get_response(score_chain, {"resume": resume, "job": job})

    # STEP 3: Explanation
    explain_chain = explanation_prompt | llm
    explanation = get_response(explain_chain, {
        "score": score,
        "resume": resume,
        "job": job
    })

    # Clean JSON parsing for score
    try:
        score_json = json.loads(
            score.replace("```json", "").replace("```", "").strip()
        )
    except:
        score_json = {"raw_output": score}

    return {
        "skills": skills,
        "score": score_json,
        "explanation": explanation
    }


# -----------------------------
# Runner
# -----------------------------
if __name__ == "__main__":

    job = load_text("data/job.txt")

    resumes = ["strong.txt", "average.txt", "weak.txt"]

    for file in resumes:
        resume = load_text(f"data/{file}")

        print("\n====================")
        print(f"Candidate: {file}")
        print("====================")

        result = run(resume, job)

        # ---------------- CLEAN OUTPUT ----------------
        print("\n SKILLS:")
        print(result["skills"])

        print("\n SCORE:")
        print(json.dumps(result["score"], indent=2))

        print("\n EXPLANATION:")
        print(result["explanation"])