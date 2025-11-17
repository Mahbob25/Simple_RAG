# this will load the files emmbedds them and sotre in the vectore store one time.

from src.ragcopy import process_all_files, chucnk_data, EmbeddingsManager, VectorStore

all_docs = process_all_files("data/")
chunks = chucnk_data(all_docs)

emb = EmbeddingsManager()
texts = [doc.page_content for doc in chunks]
embeddings = emb.generate_embeddings(texts)

vs = VectorStore()
vs.add_documents(chunks, embeddings)

print("Done! Vector store saved.")
