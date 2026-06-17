from pathlib import Path

from langchain_community.document_loaders import Docx2txtLoader, PyPDFLoader, TextLoader


class LoadFiles:
    @staticmethod
    def load_file_from_folder(folder_path: str, file_name: str):
        file_path = Path(folder_path) / file_name
        type_document = file_path.suffix.lower()

        if type_document == ".txt":
            return TextLoader(str(file_path), encoding="utf-8").load()
        if type_document == ".pdf":
            return PyPDFLoader(str(file_path)).load()
        if type_document == ".docx":
            return Docx2txtLoader(str(file_path)).load()

        raise ValueError(f"Unsupported file type: {type_document}")

    @staticmethod
    def load_pdfs_from_folder(folder_path: str):
        folder = Path(folder_path)
        documents = []

        if not folder.exists():
            raise FileNotFoundError(f"Folder not found: {folder}")

        for file_path in folder.glob("*.pdf"):
            documents.extend(PyPDFLoader(str(file_path)).load())

        return documents

    @staticmethod
    def save_file(path: str, file_name: str, content: str):
        file_path = Path(path) / file_name

        with open(file_path, "w", encoding="utf-8") as wf:
            wf.write(content)
