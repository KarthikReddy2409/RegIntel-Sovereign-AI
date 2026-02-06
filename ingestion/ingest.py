"""
Script to ingest PDFs into Qdrant for regulatory docs search.
Put PDFs in data/regs/, run with: python ingestion/ingest.py
Needs Qdrant on localhost:6333.
"""

import os
import glob
import torch
from typing import List
from langchain_core.documents import Document
from unstructured.partition.pdf import partition_pdf
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_qdrant import Qdrant
from qdrant_client import QdrantClient, models

DATA_DIR = "data/regs"
QDRANT_HOST = "http://localhost:6333"
COLLECTION_NAME = "regulatory_docs"
EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"

def get_device():
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"

def process_pdf(file_path: str) -> List[Document]:
    print(f"Processing PDF: {file_path}")

    elements = partition_pdf(
        filename=file_path,
        strategy="hi_res",
        infer_table_structure=True,
        chunking_strategy="by_title",
        max_characters=1000,
        new_after_n_chars=800,
        combine_text_under_n_chars=200,
    )

    chunks = []
    for element in elements:
        metadata = element.metadata.to_dict()

        doc_metadata = {
            "source": file_path,
            "page": metadata.get("page_number"),
            "filename": metadata.get("filename"),
            "category": element.category,
        }

        if element.category == "Table":
            doc_metadata["is_table"] = True
            doc_metadata["html"] = metadata.get("text_as_html")

        chunks.append(Document(
            page_content=element.text,
            metadata=doc_metadata
        ))

    return chunks

def run():
    device = get_device()
    print(f"Using device: {device.upper()}")

    pdf_files = glob.glob(os.path.join(DATA_DIR, "*.pdf"))
    if not pdf_files:
        print(f"No PDFs found in {DATA_DIR}")
        return

    all_chunks = []
    for pdf_file in pdf_files:
        try:
            chunks = process_pdf(pdf_file)
            all_chunks.extend(chunks)
        except Exception as e:
            print(f"Error processing {pdf_file}: {e}")

    print(f"Extracted {len(all_chunks)} chunks")

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": device},
        encode_kwargs={"normalize_embeddings": True}
    )

    qdrant_client = QdrantClient(url=QDRANT_HOST)
    qdrant_client.recreate_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=models.VectorParams(
            size=768,
            distance=models.Distance.COSINE
        )
    )

    vector_store = Qdrant(
        client=qdrant_client,
        collection_name=COLLECTION_NAME,
        embeddings=embeddings
    )

    vector_store.add_documents(all_chunks)
    print("Done ingesting to Qdrant")

if __name__ == "__main__":
    run()
