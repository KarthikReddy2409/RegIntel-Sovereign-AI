import os
import glob
import torch
from typing import List
from langchain_core.documents import Document
from unstructured.partition.pdf import partition_pdf
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_qdrant import Qdrant

# Settings
DATA_DIR = "../data/regs"
QDRANT_HOST = "http://localhost:6333"
COLLECTION = "regulatory_docs"
MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"

def get_device():
    if torch.backends.mps.is_available(): return "mps"
    if torch.cuda.is_available(): return "cuda"
    return "cpu"

def process_pdf(path: str) -> List[Document]:
    print(f"Processing: {path}")
    
    # hi_res strategy for table extraction
    elements = partition_pdf(
        filename=path,
        strategy="hi_res",
        infer_table_structure=True,
        chunking_strategy="by_title",
        max_characters=1000,
        new_after_n_chars=800,
        combine_text_under_n_chars=200,
    )

    chunks = []
    for el in elements:
        meta = el.metadata.to_dict()
        
        doc_meta = {
            "source": path,
            "page": meta.get("page_number"),
            "file": meta.get("filename"),
            "category": el.category
        }

        if el.category == "Table":
            doc_meta["is_table"] = True
            doc_meta["html"] = meta.get("text_as_html")

        chunks.append(Document(page_content=el.text, metadata=doc_meta))
        
    return chunks

def run():
    device = get_device()
    print(f"Device: {device.upper()}")

    files = glob.glob(os.path.join(DATA_DIR, "*.pdf"))
    if not files:
        print(f"No PDFs in {DATA_DIR}")
        return

    corpus = []
    for f in files:
        try:
            corpus.extend(process_pdf(f))
        except Exception as e:
            print(f"Failed {f}: {e}")

    print(f"Indexing {len(corpus)} chunks...")
    
    emb = HuggingFaceEmbeddings(
        model_name=MODEL_NAME,
        model_kwargs={'device': device},
        encode_kwargs={'normalize_embeddings': True} 
    )

    print("Pushing to Qdrant...")
    Qdrant.from_documents(
        documents=corpus,
        embedding=emb,
        url=QDRANT_HOST,
        prefer_grpc=True,
        collection_name=COLLECTION,
        force_recreate=True, 
        batch_size=64 
    )
    print("Done.")

if __name__ == "__main__":
    run()
