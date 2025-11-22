import os
import json
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables from .env
load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
print("API KEY LOADED:", api_key[:6] + "********" if api_key else "NOT FOUND")

# Create OpenAI client
client = OpenAI(api_key=api_key)


def call_llm(prompt):
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3
    )
    return response.choices[0].message.content


def generate_questions(screen_text, explanation):
    prompt = f"""
You are an AI interviewer.

Screen Content:
{screen_text}

Student Explanation:
{explanation}

Generate:
- 3 technical interview questions
- 2 follow-up questions

Return ONLY this JSON format:
{{
  "questions": ["Q1", "Q2", "Q3"],
  "followups": ["F1", "F2"]
}}
"""

    result = call_llm(prompt)

    try:
        return json.loads(result)
    except Exception:
        print("❌ JSON Parsing Failed. Model Output:")
        print(result)
        return None


def evaluate_answer(screen_text, explanation, answer):
    prompt = f"""
Evaluate this student project presentation.

Screen Content:
{screen_text}

Explanation:
{explanation}

Student Answer:
{answer}

Give scores (1–10) for:
- technical_depth
- clarity
- originality
- implementation_understanding

Return ONLY this JSON:
{{
  "technical_depth": 0,
  "clarity": 0,
  "originality": 0,
  "implementation_understanding": 0,
  "feedback": "short feedback"
}}
"""

    result = call_llm(prompt)

    try:
        return json.loads(result)
    except Exception:
        print("❌ JSON Parsing Failed. Model Output:")
        print(result)
        return None


# --------- TEST RUN ---------
if __name__ == "__main__":
    screen_text = "This project uses CNN for image classification with ResNet architecture."
    explanation = "I fine-tuned ResNet50 using transfer learning on a medical image dataset."

    print("\n=== Generating Questions ===")
    questions = generate_questions(screen_text, explanation)
    print(json.dumps(questions, indent=2))

    print("\n=== Evaluating Answer ===")
    evaluation = evaluate_answer(
        screen_text,
        explanation,
        "CNN extracts features while ResNet improves gradient flow using skip connections."
    )

    print(json.dumps(evaluation, indent=2))
