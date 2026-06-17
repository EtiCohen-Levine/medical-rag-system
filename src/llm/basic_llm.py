import re


class BasicLLM:
    @staticmethod
    def generate_answer(question, documents):
        if not documents:
            return "I could not find relevant information in the documents."

        question_words = BasicLLM._extract_words(question)
        sentences = BasicLLM._extract_sentences(documents)
        scored_sentences = []

        for sentence in sentences:
            sentence_words = BasicLLM._extract_words(sentence)
            score = len(question_words.intersection(sentence_words))
            scored_sentences.append((score, sentence))

        scored_sentences.sort(reverse=True, key=lambda item: item[0])
        best_sentences = [
            sentence
            for score, sentence in scored_sentences[:3]
            if score > 0
        ]

        if not best_sentences:
            best_sentences = [documents[0].page_content.strip()]

        return " ".join(best_sentences)

    @staticmethod
    def _extract_sentences(documents):
        text = "\n".join(document.page_content for document in documents)
        sentences = re.split(r"(?<=[.!?])\s+", text)

        return [
            sentence.strip()
            for sentence in sentences
            if sentence.strip()
        ]

    @staticmethod
    def _extract_words(text):
        return set(re.findall(r"\w+", text.lower()))
