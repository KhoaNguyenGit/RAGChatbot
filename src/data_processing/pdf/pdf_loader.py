from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import DirectoryLoader

class PDFLoader:
    def __init__(self, dir: str):
        self.dir = dir

    def load_pdf_file(self):
        loader = DirectoryLoader(self.dir,
                                glob="*.pdf",
                                loader_cls=PyPDFLoader)

        documents = loader.load()

        return documents