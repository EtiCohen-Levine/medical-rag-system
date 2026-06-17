import hashlib
import math
import re


class SimpleEmbeddingModel:
    def __init__(self, size=384):
        self.size = size

    def embed_documents(self, texts):
        return [self._embed_text(text) for text in texts]

    def embed_query(self, text):
        return self._embed_text(text)

    def _embed_text(self, text):
        vector = [0.0] * self.size
        words = re.findall(r"\w+", text.lower())

        for word in words:
            word_hash = hashlib.sha256(word.encode("utf-8")).hexdigest()
            index = int(word_hash, 16) % self.size
            vector[index] += 1.0

        vector_size = math.sqrt(sum(value * value for value in vector))

        if vector_size == 0:
            return vector

        return [value / vector_size for value in vector]


class IngestionEmbedding:
    @staticmethod
    def create_embedding_model():
        return SimpleEmbeddingModel()

    @staticmethod
    def create_embeddings(chunks, limit=None):
        embedding_model = IngestionEmbedding.create_embedding_model()
        selected_chunks = chunks[:limit] if limit else chunks
        texts = [chunk.page_content for chunk in selected_chunks]

        return embedding_model.embed_documents(texts)
