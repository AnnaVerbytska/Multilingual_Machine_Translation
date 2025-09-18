"""
simplify_prompts.py
Utility functions for Gemini text simplification
"""

from google import genai

# --- Prompt Templates ---

def one_shot_prompt(input_text: str, audience: str) -> str:
    """Build a one-shot prompt for text simplification."""
    audience_guidelines = {
        "low-literacy": "Use very short sentences. Use simple words.",
        "children": "Use very simple words and short sentences. Make it fun and easy to understand. Give examples children can relate to, like games, school, or friends.",
        "ADHD": "Use bullet points. Keep sentences brief. Highlight the main idea."
    }

    return f"""
You are an assistant that simplifies texts about inclusive democracy.
Target audience: {audience}.
{audience_guidelines[audience]}

Example:
Original: Democracy requires the participation of citizens in decision-making processes to ensure equal representation.
Simplified: Democracy means people help make decisions so everyone is included.

Now simplify this text: {input_text}
"""


def few_shot_prompt(input_text: str, audience: str) -> str:
    """Build a few-shot prompt for text simplification."""
    audience_guidelines = {
        "low-literacy": "Use very short sentences. Use simple words.",
        "children": "Use very simple words and short sentences. Make it fun and easy to understand. Give examples children can relate to, like games, school, or friends.",
        "ADHD": "Use bullet points. Keep sentences brief. Highlight the main idea."
    }

    return f"""
You simplify texts about inclusive democracy.
Target audience: {audience}.
{audience_guidelines[audience]}

Example 1:
Original: Every citizen has the right to vote and be heard.
Simplified: Everyone can vote. Everyone's voice matters.

Example 2:
Original: Democracy depends on equal access to information.
Simplified: People need the same information to take part in democracy.

Now simplify this text: {input_text}
"""


def cot_prompt(input_text: str, audience: str) -> str:
    """Build a chain-of-thought (CoT) prompt for text simplification."""
    audience_guidelines = {
        "low-literacy": "Use very short sentences. Use simple words.",
        "children": "Use very simple words and short sentences. Make it fun and easy to understand. Give examples children can relate to, like games, school, or friends.",
        "ADHD": "Use bullet points. Keep sentences brief. Highlight the main idea."
    }

    return f"""
Simplify the following text about inclusive democracy for {audience}.
{audience_guidelines[audience]}

First, think step by step:
1. Identify the main idea.
2. Break the idea into small, clear parts.
3. Use short sentences.
4. Present key points clearly.

Text: {input_text}

Now give the final simplified version:
"""


# --- Wrapper Function ---

def simplify_with_gemini(client, text: str, style: str, audience: str) -> str:
    """Send a text simplification request to Gemini based on style and audience."""
    if style == "one-shot":
        prompt = one_shot_prompt(text, audience)
    elif style == "few-shot":
        prompt = few_shot_prompt(text, audience)
    elif style == "cot":
        prompt = cot_prompt(text, audience)
    else:
        raise ValueError(f"Unknown style: {style}")

    response = client.models.generate_content(
        model="gemini-2.0-flash-001",
        contents=prompt
    )
    return response.text

