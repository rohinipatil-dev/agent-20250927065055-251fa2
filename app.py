import streamlit as st
from typing import List, Dict
from openai import OpenAI

# Initialize OpenAI client (expects OPENAI_API_KEY in environment)
client = OpenAI()


def build_instruction_prompt(level: str, topics: List[str], n_tips: int, code_examples: bool, style: str) -> str:
    topics_text = ", ".join(topics) if topics else "General best practices"
    return (
        f"Produce high-quality, actionable Python tips tailored to the audience and constraints:\n"
        f"- Audience: {level}\n"
        f"- Number of tips: {n_tips}\n"
        f"- Focus topics: {topics_text}\n"
        f"- Include code examples: {'Yes' if code_examples else 'No'}\n"
        f"- Style: {style}\n"
        f"Output requirements:\n"
        f"- Use Python 3.11+ features and standard library when appropriate.\n"
        f"- Present as a numbered list. Each tip should include a short rationale.\n"
        f"- Keep tips concise and practical; avoid unnecessary verbosity.\n"
        f"- Prefer built-in tools and patterns; call out common pitfalls where relevant.\n"
    )


def build_messages(user_text: str, level: str, topics: List[str], n_tips: int, code_examples: bool, style: str) -> List[Dict[str, str]]:
    system_base = "You are a helpful assistant."
    system_instructions = build_instruction_prompt(level, topics, n_tips, code_examples, style)
    user_content = user_text.strip() or "Tell me Python tips."
    return [
        {"role": "system", "content": system_base},
        {"role": "system", "content": system_instructions},
        {"role": "user", "content": user_content},
    ]


def generate_tips(model: str, messages: List[Dict[str, str]]) -> str:
    response = client.chat.completions.create(
        model=model,  # "gpt-4" or "gpt-3.5-turbo"
        messages=messages,
        temperature=0.6,
    )
    return response.choices[0].message.content


def ui_sidebar() -> Dict[str, any]:
    with st.sidebar:
        st.header("Settings")
        model = st.selectbox("Model", options=["gpt-4", "gpt-3.5-turbo"], index=0)
        level = st.selectbox("Experience level", options=["Beginner", "Intermediate", "Advanced"], index=1)
        topics = st.multiselect(
            "Focus topics (optional)",
            options=[
                "Best Practices",
                "Clean Code",
                "Typing",
                "Performance",
                "Debugging",
                "Testing",
                "Packaging",
                "Concurrency/Async",
                "OOP/Design",
                "Data Classes",
                "CLI Tools",
                "Security",
                "Error Handling",
                "I/O and Files",
                "Pandas",
                "NumPy",
                "Web (FastAPI/Flask)",
            ],
            default=["Best Practices", "Debugging"],
        )
        n_tips = st.slider("Number of tips", min_value=3, max_value=25, value=10, step=1)
        code_examples = st.checkbox("Include code examples", value=True)
        style = st.radio("Style", options=["Concise", "Detailed"], index=0, horizontal=True)
        return {
            "model": model,
            "level": level,
            "topics": topics,
            "n_tips": n_tips,
            "code_examples": code_examples,
            "style": style,
        }


def main():
    st.set_page_config(page_title="Python Tips Generator", page_icon="🐍")
    st.title("🐍 Python Tips Generator")
    st.caption("Generate tailored Python tips using OpenAI chat models.")

    cfg = ui_sidebar()

    default_prompt = "Tell me Python tips for writing clean, efficient, and maintainable code."
    user_text = st.text_area(
        "What would you like help with?",
        value=default_prompt,
        height=120,
        placeholder="e.g., Tips for writing robust async code and testing strategies",
    )

    col1, col2 = st.columns([1, 1])
    with col1:
        generate = st.button("Generate Tips")
    with col2:
        clear = st.button("Clear")

    if clear:
        st.experimental_rerun()

    if generate:
        with st.spinner("Generating tips..."):
            try:
                messages = build_messages(
                    user_text=user_text,
                    level=cfg["level"],
                    topics=cfg["topics"],
                    n_tips=cfg["n_tips"],
                    code_examples=cfg["code_examples"],
                    style=cfg["style"],
                )
                output = generate_tips(model=cfg["model"], messages=messages)
                st.subheader("Results")
                st.write(output)
            except Exception as e:
                st.error(
                    "Failed to generate tips. Ensure OPENAI_API_KEY is set in your environment. "
                    f"Details: {e}"
                )

    with st.expander("How to use"):
        st.markdown(
            "- Adjust the audience level, topics, and number of tips in the sidebar.\n"
            "- Toggle code examples and choose a concise or detailed style.\n"
            "- Provide additional context in the prompt for more specific guidance.\n"
        )


if __name__ == "__main__":
    main()