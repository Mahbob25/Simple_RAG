import os
from langchain_community.document_loaders import TextLoader 
from langchain_text_splitters import RecursiveCharacterTextSplitter
import numpy as np
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
import uuid
from typing import List, Dict, Any, Tuple
from sklearn.metrics.pairwise import cosine_similarity
import os
from dotenv import load_dotenv
import google.generativeai as genai
from pathlib import Path


# Read all docs 
def process_all_files(directory):
    all_documents = []
    text_dir = Path(directory).resolve()

    #find all text files recursively
    text_files = list(text_dir.glob("**/*.txt"))
    print(f"Found {len(text_files)} text files to process")

    for text_file in text_files:
        print(f"\nProcessing:{text_file.name}")
        try:
            loader = TextLoader(str(text_file))
            documents = loader.load()

            #add source info to metadata
            for doc in documents:
                doc.metadata['source_file'] = text_file.name
                doc.metadata['file_type'] = "text"

            all_documents.extend(documents)
        except Exception as e:
            print(f" Error: {e}")

    print(f"\nTotal documents loaded: {len(all_documents)}")
    return all_documents

all_text_documents = process_all_files("D:/RAG/MY_RAG/data/")


def chucnk_data(documents, chunk_size=1000, chunk_overlap=200):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )

    split_docs = text_splitter.split_documents(documents)
    print(f"split {len(documents)} documents into {len(split_docs)} chuncks")

    #show example of a chunk
    if split_docs:
        print(f"\n Example chunks:")
        print(f"content:{split_docs[0].page_content[:200]} ...")
        print(f"metadata: {split_docs[0].metadata}")
    return split_docs
chuncks = chucnk_data(all_text_documents)




class EmbeddingsManager:
    """
    Manages loading of embedding models and generating embeddings.
    Args:
        model_name (str): Name of the embedding model to load.   
    """
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        self.model_name = model_name
        self.model = None
        self._load_model()

    def _load_model(self):
        try:
            print(f"Loading embedding model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
            print(f"Model loaded successfully. Embedding dimension: {self.model.get_sentence_embedding_dimension()} ")
        except Exception as e:
            print(f"Error loading model {self.model_name}: {e}")
            raise e
        
    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings for a list of texts.

        Args:
            texts (List[str]): List of input texts.

        Returns:   
            np.ndarray: Array of embeddings.
        """

        if not self.model:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        print(f"Generating embeddings for {len(texts)} texts...")
        embeddings = self.model.encode(texts, show_progress_bar=True)
        print(f"Generated embeddings with shape: {embeddings.shape}")
        return embeddings
    

emb_manager = EmbeddingsManager()
emb_manager

class VectorStore:
    """
    Manages a vector store using ChromaDB for storing and retrieving document embeddings.
    
    Args:
        collection_name (str): Name of the ChromaDB collection.
        persist_directory (str): Directory to persist the ChromaDB database.
    """
    def __init__(self, collection_name: str = "documents", persist_directory: str = r"D:\\RAG\\my_rag\\data\\vector_store"):
        """
        Initialize the VectorStore with ChromaDB client and collection.
        Args:
            collection_name (str): Name of the ChromaDB collection.
            persist_directory (str): Directory to persist the ChromaDB database.
        """
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.client = None
        self.collection = None
        self._initialize_store()

    def _initialize_store(self):
        """
        Initialize the ChromaDB client and collection.
        """
        try:
            #create persist chromaDB 
            os.makedirs(self.persist_directory, exist_ok=True)
            self.client = chromadb.PersistentClient(path=self.persist_directory)

            # Get or create collection
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"description": "text documents embeddings for RAG"}
                )
            print(f"ChromaDB collection '{self.collection_name}' initialized at '{self.persist_directory}'")
            print(f"Current number of documents in collection: {self.collection.count()}")
        except Exception as e:
            print(f"Error initializing ChromaDB: {e}")
            raise e

    def add_documents(self, documents: List[Any], embeddings: np.ndarray):
        """
        add documents and their embeddings to the vector store.

        args:
            documents (List[Any]): List of document objects with metadata.
            embeddings (np.ndarray): Array of embeddings corresponding to the documents.
        """

        if len(documents) != embeddings.shape[0]:
            raise ValueError("Number of documents and embeddings must match.")
        
        print(f"Adding {len(documents)} documents to the vector store...")

        # Prepare data for insertion for ChromaDB
        ids = []
        metadatas = []
        documents_texts = []
        embeddings_list = []

        for i, (doc, embeddings) in enumerate(zip(documents, embeddings)):
            # Generate unique ID for each document
            doc_id = f"doc_{uuid.uuid4().hex[:8]}_{i}"
            ids.append(doc_id)

            # Prepare metadata and text
            metadata = dict(doc.metadata)  # Ensure metadata is a dictionary
            metadata['doc_index'] = i
            metadata['content_length'] = len(doc.page_content)
            metadatas.append(metadata)
            
            # Document content
            documents_texts.append(doc.page_content)

            # Embedding
            embeddings_list.append(embeddings.tolist())

        # print(f"metadata count: {len(metadatas)}")
        # Add to collection chromaDB
        try:
            self.collection.add(
                ids=ids,
                metadatas=metadatas,
                documents=documents_texts,
                embeddings=embeddings_list
            )
            print(f"Successfully added {len(documents)} documents to the vector store.")
            print(f"Total documents in collection: {self.collection.count()}")

        except Exception as e:
            print(f"Error adding documents to vector store: {e}")
            raise e

vectorstore=VectorStore()
vectorstore

### convert chuncks to embeddings

## Extract texts from chuncks
texts =[doc.page_content for doc in chuncks]

## Generate embeddings
embeddings = emb_manager.generate_embeddings(texts)

## Add documents and embeddings to vector store
vectorstore.add_documents(chuncks, embeddings) 


class RAGRetriever:
    """
    Retriever class to fetch relevant documents from the vector store based on query embeddings.
    
    Args:
        vector_store (VectorStore): Instance of the VectorStore class.
        emb_manager (EmbeddingsManager): Instance of the EmbeddingsManager class.
        top_k (int): Number of top similar documents to retrieve.
    """
    def __init__(self, vector_store: VectorStore, emb_manager: EmbeddingsManager):
        self.vector_store = vector_store
        self.emb_manager = emb_manager


    def retrieve(self, query: str, top_k: int = 5, score_threshold: float = 0.0) -> List[Dict[str, Any]]:
        """
        Retrieve top_k similar documents from the vector store based on the query.

        Args:
            query (str): Input query string.
            top_k (int): Number of top similar documents to retrieve.
            score_threshold (float): Minimum similarity score threshold for retrieval.
        Returns:   
            List[Dict[str, Any]]: List of retrieved documents with metadata.
        """

        print(f"Retrieving documents for query: '{query}'")
        print(f"Using top_k={top_k} and score_threshold={score_threshold}")


        print(f"Generating embedding for the query: '{query}'")
        query_embedding = self.emb_manager.generate_embeddings([query])[0]

        # Search in vector store
        try:
            print(f"Searching for top {top_k} similar documents in the vector store...")
            results = self.vector_store.collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=top_k
            )

            # Process results
            retrieved_docs = []
            if results['documents'] and results['documents'][0]:
                print(f"Found {len(results['documents'][0])} documents.")
                document = results['documents'][0]
                metadata = results['metadatas'][0]
                distances = results['distances'][0]
                ids = results['ids'][0]

                for i, (doc_id, document, metadata, distance) in enumerate(zip(ids, document, metadata, distances)):
                    similarity_score = 1 - distance  # Convert distance to similarity score
                    if similarity_score >= score_threshold:
                        retrieved_docs.append({
                            "id": doc_id,
                            "content": document,
                            "metadata": metadata,
                            "similarity_score": similarity_score,
                            'distance': distance,
                            'rank': i + 1
                        })

            print(f"Retrieved {len(retrieved_docs)} documents.")
            return retrieved_docs
        except Exception as e:
            print(f"Error during retrieval: {e}")
            raise e

rag_retriever = RAGRetriever(vector_store=vectorstore, emb_manager=emb_manager)


retrieved_chunks = rag_retriever.retrieve(query="What is machine learning and what is python?", top_k=5, score_threshold=0.1)



def configure_gemini():
    load_dotenv()
    api_key = os.getenv("GOOGLE_API_KEY")

    if not api_key:
        raise ValueError("GOOGLE_API_KEY is missing in your .env file.")

    genai.configure(api_key=api_key)
    print("Gemini API initialized successfully.")

configure_gemini()

class RAGGenerator:
    """
    Generates answers based on retrieved document chunks using a generative model.
    Args:
        model_name (str): Name of the generative model to use.
    returns:
        str: Generated answer.
    """
    def __init__(self, model_name: str = "gemini-2.5-flash"):
        self.model_name = model_name
        
        # Initialize the generative AI client
        # self.client = generativeai.Client()  --- IGNORE ---

    def generate_answer(self, question: str, retrieved_chunks: List[Dict[str, Any]]) -> str:
        """
        Generate an answer to the question based on the retrieved document chunks.

        Args:
            question (str): The user's question.
            retrieved_chunks (List[Dict[str, Any]]): List of retrieved document chunks.

        Returns:
            str: Generated answer.
        """
        # Convert chunks into a single context block
        context_text = "\n\n".join(
            [f"Source {i+1}:\n{chunk['content']}" for i, chunk in enumerate(retrieved_chunks)]
        )

        prompt = f"""
You are an expert AI assistant.

Use ONLY the following retrieved context to answer the user's question.
if you do not know the answer, say "The information is not available in the provided documents."

Context:
{context_text}

User Question: {question}

Give a clear, helpful answer. Do NOT hallucinate. If the answer is not in the context, say "The information is not available in the provided documents."
"""
        model = genai.GenerativeModel(self.model_name)
        response = model.generate_content(prompt)
        return response.text
    

rag_generator = RAGGenerator()
        

def user_call(question: str):
    retrieved_chunks = rag_retriever.retrieve(query=question, top_k=5, score_threshold=0.1)
    
    answer = rag_generator.generate_answer(
        question=question,
        retrieved_chunks=retrieved_chunks
    )
    if not answer or answer.strip() == "":
        raise ValueError("Unable to generate an answer.")
    return answer


print("=== RAG System ===")


while True:
    question = input("\nEnter your question (or type 'exit' to quit): ")

    if question.lower() in ["exit", "quit", "q"]:
        print("Goodbye!")
        break

    print("\nGenerating answer...\n")
    answer = user_call(question)

    print("=== Answer ===")
    print(answer)



