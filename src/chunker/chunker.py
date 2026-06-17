from langchain_text_splitters import RecursiveCharacterTextSplitter


class Chunker:
    @staticmethod
    def divide_to_chunks(documents):
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            length_function=len,
        )

        return text_splitter.split_documents(documents)
