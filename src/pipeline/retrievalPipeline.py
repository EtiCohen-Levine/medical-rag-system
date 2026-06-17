from pathlib import Path
import sys


project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

from src.embedding.injestionEmbedding import IngestionEmbedding
from src.llm.basic_llm import BasicLLM
from src.prompt.prompt_builder import PromptBuilder
from src.vector_db.chroma_store import ChromaStore


chroma_folder = project_root / "data" / "chroma_db"
question = "What do you do for clogged ears??"

embedding_model = IngestionEmbedding.create_embedding_model()
collection = ChromaStore.load_existing(embedding_model, chroma_folder)
retriever = ChromaStore.create_retriever(collection, embedding_model, search_results=3)
results = retriever.invoke(question)
prompt = PromptBuilder.build_prompt(question, results)
answer = BasicLLM.generate_answer(question, results)

print(f"Question: {question}")
print(f"Found {len(results)} relevant chunks")
print("\nPrompt:")
print(prompt[:1000])
print("\nAnswer:")
print(answer)

for index, document in enumerate(results, start=1):
    print(f"\n--- Result {index} ---")
    print(document.page_content[:700])
    print(f"Metadata: {document.metadata}")
