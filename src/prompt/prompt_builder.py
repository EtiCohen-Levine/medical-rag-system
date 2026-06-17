class PromptBuilder:
    @staticmethod
    def build_prompt(question, documents):
        context = "\n\n".join(document.page_content for document in documents)

        return f"""Answer the question using only the context below.

Context:
{context}

Question:
{question}

Answer:"""
