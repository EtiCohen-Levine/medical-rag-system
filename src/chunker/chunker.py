from xml.dom.minidom import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

class Chunker:
    @staticmethod
    def divide_to_chunks(document: Document):
        return RecursiveCharacterTextSplitter(
            chunk_size=100,
            chunk_overlap=20,
            length_function=len,
        ).create_documents(document)