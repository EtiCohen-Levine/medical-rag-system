from pathlib import Path
import sys


project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

from src.chunker.chunker import Chunker
from src.embedding.injestionEmbedding import IngestionEmbedding
from src.Loader.load_files import LoadFiles
from src.vector_db.chroma_store import ChromaStore


data_folder = project_root / "data" / "original"
chroma_folder = project_root / "data" / "chroma_db"

documents = LoadFiles.load_pdfs_from_folder(str(data_folder))
chunks = Chunker.divide_to_chunks(documents)
embedding_model = IngestionEmbedding.create_embedding_model()
embeddings = IngestionEmbedding.create_embeddings(chunks, limit=3)
collection = ChromaStore.create_from_documents(
    chunks,
    embedding_model,
    chroma_folder,
)

print(f"Loaded {len(documents)} document pages")
print(f"Created {len(chunks)} chunks")
print(f"Created {len(embeddings)} sample embeddings")
print(f"Embedding size: {len(embeddings[0]) if embeddings else 0}")
print(f"Saved chunks to Chroma DB: {chroma_folder}")
