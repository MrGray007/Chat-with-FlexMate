# FlexMate 🏋️‍♂️

**FlexMate** is an AI-powered motivational gym companion that helps users with fitness guidance, answers their questions, and provides supportive encouragement during workouts. Built with **Streamlit**, **LangChain**, and **Hugging Face embeddings**, it offers an interactive chat experience with persistent session-based history.

---

## Features

- 🤖 AI-powered gym buddy for Q&A and motivation.
- 🏋️ Personalized coaching with mood-based responses (`supportive`, `angry`).
- 💬 Maintains session-based chat history with persistence.
- 🧠 Uses **Hugging Face embeddings** for semantic retrieval of context.
- ⚡ Quick and interactive UI built with **Streamlit**.
- 💾 Optional database storage for chat history.

---

## Installation

1. Clone the repository:

```bash
git clone https://github.com/MrGray007/Chat-with-FlexMate.git
cd Chat-with-FlexMate

pip install -r requirements.txt

F_TOKEN=your_huggingface_api_key
LANGSMITH_API_KEY=your_langsmith_api_key
GROQ_API_KEY=your_groq_api_key
