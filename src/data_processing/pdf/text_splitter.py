from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from typing import List
from uuid import uuid4
import re

class PDFTextSplitter:
    def __init__(self, chunk_size: int, chunk_overlap: int):
        """
        Splits cleaned PDF text into semantically meaningful chunks.
        """
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    @staticmethod
    def filter_to_minimal_docs(docs: List[Document]) -> List[Document]:
        """
        Reduce each Document to a minimal structure:
        - Unique ID
        - Cleaned text
        - Source metadata
        """
        minimal_docs: List[Document] = []
        for doc in docs:
            src = doc.metadata.get("source")
            minimal_docs.append(
                Document(
                    page_content=doc.page_content,
                    metadata={
                        "id": str(uuid4()),
                        "source": src,
                    }
                )
            )
        return minimal_docs

    def split(self, documents: List[Document]) -> List[Document]:
        texts_chunk = self.text_splitter.split_documents(documents)
        return texts_chunk