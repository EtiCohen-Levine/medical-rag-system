import shutil
import math
import re
import sqlite3
from pathlib import Path
from uuid import uuid4

import chromadb
from langchain_core.documents import Document


class ChromaRetriever:
    def __init__(self, persist_directory, embedding_model, search_results=3, min_keyword_overlap=1):
        self.persist_directory = Path(persist_directory)
        self.embedding_model = embedding_model
        self.search_results = search_results
        self.min_keyword_overlap = min_keyword_overlap

    def invoke(self, question):
        query_embedding = self.embedding_model.embed_query(question)
        question_words = self._extract_meaningful_words(question)
        stored_documents = self._load_documents_from_chroma_sqlite()
        scored_items = []

        for page_content, metadata in stored_documents:
            document_words = self._extract_meaningful_words(page_content)
            keyword_overlap = len(question_words.intersection(document_words))

            if keyword_overlap < self.min_keyword_overlap:
                continue

            embedding = self.embedding_model.embed_query(page_content)
            score = self._cosine_similarity(query_embedding, embedding)
            scored_items.append((score, page_content, metadata))

        scored_items.sort(reverse=True, key=lambda item: item[0])

        relevant_documents = []

        for _, page_content, metadata in scored_items[: self.search_results]:
            relevant_documents.append(
                Document(
                    page_content=page_content,
                    metadata=metadata or {},
                )
            )

        return relevant_documents

    def _load_documents_from_chroma_sqlite(self):
        db_path = self.persist_directory / "chroma.sqlite3"

        if not db_path.exists():
            raise FileNotFoundError(f"Chroma DB was not found: {db_path}")

        connection = sqlite3.connect(db_path)
        cursor = connection.cursor()
        rows = cursor.execute(
            """
            SELECT rowid, string_value
            FROM embedding_fulltext_search
            """
        ).fetchall()

        documents = []

        for row_id, page_content in rows:
            metadata = self._load_metadata(cursor, row_id)
            documents.append((page_content, metadata))

        connection.close()

        return documents

    @staticmethod
    def _load_metadata(cursor, row_id):
        metadata_rows = cursor.execute(
            """
            SELECT key, string_value, int_value, float_value, bool_value
            FROM embedding_metadata
            WHERE id = ?
            """,
            (row_id,),
        ).fetchall()

        metadata = {}

        for key, string_value, int_value, float_value, bool_value in metadata_rows:
            if key == "chroma:document":
                continue

            if string_value is not None:
                metadata[key] = string_value
            elif int_value is not None:
                metadata[key] = int_value
            elif float_value is not None:
                metadata[key] = float_value
            elif bool_value is not None:
                metadata[key] = bool(bool_value)

        return metadata

    @staticmethod
    def _cosine_similarity(first_vector, second_vector):
        dot_product = sum(
            first_value * second_value
            for first_value, second_value in zip(first_vector, second_vector)
        )
        first_size = math.sqrt(sum(value * value for value in first_vector))
        second_size = math.sqrt(sum(value * value for value in second_vector))

        if first_size == 0 or second_size == 0:
            return 0.0

        return dot_product / (first_size * second_size)

    @staticmethod
    def _extract_meaningful_words(text):
        stop_words = {
            "a",
            "an",
            "and",
            "are",
            "for",
            "in",
            "is",
            "of",
            "on",
            "the",
            "to",
            "what",
        }
        words = re.findall(r"\w+", text.lower())

        return {
            word
            for word in words
            if len(word) > 2 and word not in stop_words
        }


class ChromaStore:
    COLLECTION_NAME = "medical_documents"

    @staticmethod
    def create_from_documents(documents, embedding_model, persist_directory, reset=True):
        persist_path = Path(persist_directory)

        if reset and persist_path.exists():
            shutil.rmtree(persist_path)

        client = chromadb.PersistentClient(path=str(persist_path))
        collection = client.get_or_create_collection(name=ChromaStore.COLLECTION_NAME)

        texts = [document.page_content for document in documents]
        metadatas = [document.metadata for document in documents]
        ids = [str(uuid4()) for _ in documents]
        embeddings = embedding_model.embed_documents(texts)

        collection.add(
            documents=texts,
            metadatas=metadatas,
            ids=ids,
            embeddings=embeddings,
        )

        return collection

    @staticmethod
    def load_existing(embedding_model, persist_directory):
        return Path(persist_directory)

    @staticmethod
    def create_retriever(persist_directory, embedding_model, search_results=3):
        return ChromaRetriever(
            persist_directory=persist_directory,
            embedding_model=embedding_model,
            search_results=search_results,
        )
