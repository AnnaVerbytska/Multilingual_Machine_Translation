# mt_prompts.py
"""
Audience-aware machine translation (MT) with Gemini for specific immigrant/low-literacy groups.
"""

from google import genai

# --- Prompt Templates ---

def zero_shot_prompt(text: str, source_lang: str, target_lang: str, audience: str, level: str, notes: str="") -> str:
    """Zero-shot: translate text directly, audience-aware."""
    return f"""
Translate the following text from {source_lang} into {target_lang} for {audience} (Spanish proficiency: {level}).
Use simple sentences appropriate for the audience. Avoid idioms. Make the meaning clear and easy to understand.
{notes}

Text: {text}
"""

def one_shot_prompt(text: str, source_lang: str, target_lang: str, audience: str, level: str, notes: str="") -> str:
    """One-shot: provide one example translation for guidance."""
    example_src = "Todos los ciudadanos pueden votar y expresar su opinión."
    example_trg_dict = {
        "Ukrainian": "Всі громадяни можуть голосувати та висловлювати свою думку.",
        "German": "Alle Bürger können wählen und ihre Meinung äußern.",
        "English": "All citizens can vote and express their opinion.",
        "Polish": "Wszyscy obywatele mogą głosować i wyrażać swoją opinię.",
        "Simplified Spanish": "Todos pueden votar y decir lo que piensan."
    }
    example_trg = example_trg_dict.get(target_lang, example_src)

    return f"""
Translate the following text from {source_lang} into {target_lang} for {audience} (Spanish proficiency: {level}).
Use simple sentences appropriate for the audience. Avoid idioms. Make the meaning clear.

Example:
Original (Spanish): {example_src}
Translation ({target_lang}): {example_trg}

Now translate this text:
{text}
"""

def cot_prompt(text: str, source_lang: str, target_lang: str, audience: str, level: str, notes: str="") -> str:
    """CoT: chain-of-thought reasoning for careful audience-adapted translation."""
    return f"""
Translate the following text from {source_lang} into {target_lang} for {audience} (Spanish proficiency: {level}).
Use simple sentences appropriate for the audience. Avoid idioms. Make the meaning clear.

First, think step by step:
1. Identify the key points and meaning in Spanish.
2. Simplify sentences if necessary.
3. Adapt vocabulary to the target audience and language level.
4. Ensure the translation is accurate and clear.

Text: {text}

Now provide the final translation:
"""

# --- Wrapper function to call Gemini ---

def translate_with_gemini(client, text: str, style: str, source_lang: str, target_lang: str, audience: str, level: str, notes: str="") -> str:
    """Send translation request to Gemini based on style and audience."""
    style = style.lower()
    if style == "zero-shot":
        prompt = zero_shot_prompt(text, source_lang, target_lang, audience, level, notes)
    elif style == "one-shot":
        prompt = one_shot_prompt(text, source_lang, target_lang, audience, level, notes)
    elif style in ["cot", "chain-of-thought"]:
        prompt = cot_prompt(text, source_lang, target_lang, audience, level, notes)
    else:
        raise ValueError(f"Unknown style: {style}")

    response = client.models.generate_content(
        model="gemini-2.0-flash-001",
        contents=prompt
    )
    return response.text
