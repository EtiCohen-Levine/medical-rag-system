import string
from langchain_community.document_loaders import TextLoader, AmazonTextractPDFLoader, Docx2txtLoader


class LoadFiles:
    @staticmethod
    def load_file_from_folder(folder_path: string, file_name: string):
        type_document = file_name.split(".")[1]
        file_path = folder_path + "\\" + file_name
        if type_document == "txt":
            return TextLoader(file_path).load()
        if type_document == "pdf":
            return AmazonTextractPDFLoader(file_path).load()
        if type_document == "docx":
            return Docx2txtLoader(file_path).load()
        return None

    @staticmethod
    def save_file(path:string, file_name:string, content: string):
        with open(path + "\\" + file_name, 'w') as wf:
            wf.write(content)