"""
RAG Chatbot - Complete Implementation
Single file contains entire RAG pipeline
Author: AI Assistant | Date: 2025-10-02
"""

import os
import json
import time
import uuid
import logging
from datetime import datetime
from typing import List, Dict, Optional
from dataclasses import dataclass
import re

# Core dependencies
import numpy as np
import requests
from pathlib import Path

# Vector database
try:
    import chromadb
    from chromadb.config import Settings
except ImportError:
    print("Installing chromadb...")
    os.system("pip install chromadb")
    import chromadb
    from chromadb.config import Settings

# OpenAI for embeddings and generation
try:
    from openai import OpenAI
except ImportError:
    print("Installing openai...")
    os.system("pip install openai")
    from openai import OpenAI

# Document processing
try:
    import PyPDF2
except ImportError:
    print("Installing PyPDF2...")
    os.system("pip install PyPDF2")
    import PyPDF2

try:
    from docx import Document as DocxDocument
except ImportError:
    print("Installing python-docx...")
    os.system("pip install python-docx")
    from docx import Document as DocxDocument

# Tokenizer
try:
    import tiktoken
except ImportError:
    print("Installing tiktoken...")
    os.system("pip install tiktoken")
    import tiktoken

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class Config:
    """Configuration class"""
    OPENAI_API_KEY: str = os.getenv("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", "")
    CHROMA_DB_PATH: str = "./chroma_db"
    CHUNK_SIZE: int = 500
    CHUNK_OVERLAP: int = 50
    TOP_K: int = 5
    MODEL_NAME: str = "gpt-3.5-turbo"
    EMBEDDING_MODEL: str = "text-embedding-3-small"

class DocumentLoader:
    """Load documents from various formats"""
    
    def __init__(self):
        self.supported_formats = ['.pdf', '.docx', '.txt', '.md']
    
    def load_pdf(self, file_path: str) -> List[Dict]:
        """Load PDF file"""
        documents = []
        try:
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page_num, page in enumerate(pdf_reader.pages):
                    text = page.extract_text().strip()
                    if text:
                        documents.append({
                            'text': text,
                            'metadata': {
                                'filename': os.path.basename(file_path),
                                'page': page_num + 1,
                                'type': 'pdf'
                            }
                        })
            logger.info(f"✅ Loaded {len(documents)} pages from {file_path}")
        except Exception as e:
            logger.error(f"❌ Error loading PDF {file_path}: {e}")
        return documents
    
    def load_docx(self, file_path: str) -> List[Dict]:
        """Load DOCX file"""
        documents = []
        try:
            doc = DocxDocument(file_path)
            full_text = []
            for para in doc.paragraphs:
                if para.text.strip():
                    full_text.append(para.text)
            
            if full_text:
                documents.append({
                    'text': '\n'.join(full_text),
                    'metadata': {
                        'filename': os.path.basename(file_path),
                        'type': 'docx'
                    }
                })
            logger.info(f"✅ Loaded DOCX: {file_path}")
        except Exception as e:
            logger.error(f"❌ Error loading DOCX {file_path}: {e}")
        return documents
    
    def load_text(self, file_path: str) -> List[Dict]:
        """Load text file"""
        documents = []
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                text = file.read().strip()
                if text:
                    documents.append({
                        'text': text,
                        'metadata': {
                            'filename': os.path.basename(file_path),
                            'type': 'text'
                        }
                    })
            logger.info(f"✅ Loaded text file: {file_path}")
        except Exception as e:
            logger.error(f"❌ Error loading text {file_path}: {e}")
        return documents
    
    def load_all_documents(self, directory: str) -> List[Dict]:
        """Load all supported documents from directory"""
        all_docs = []
        doc_dir = Path(directory)
        
        if not doc_dir.exists():
            logger.warning(f"📁 Directory {directory} not found, creating...")
            doc_dir.mkdir(parents=True, exist_ok=True)
            return all_docs
        
        for file_path in doc_dir.glob('*'):
            if file_path.is_file():
                ext = file_path.suffix.lower()
                if ext == '.pdf':
                    all_docs.extend(self.load_pdf(str(file_path)))
                elif ext == '.docx':
                    all_docs.extend(self.load_docx(str(file_path)))
                elif ext in ['.txt', '.md']:
                    all_docs.extend(self.load_text(str(file_path)))
        
        logger.info(f"📚 Total documents loaded: {len(all_docs)}")
        return all_docs

class DocumentProcessor:
    """Process and chunk documents"""
    
    def __init__(self, config: Config):
        self.config = config
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
    
    def clean_text(self, text: str) -> str:
        """Clean text content"""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s\.\,\!\?\;\:\-\(\)\[\]]', ' ', text)
        return text.strip()
    
    def chunk_text(self, text: str) -> List[str]:
        """Chunk text by tokens with overlap"""
        tokens = self.tokenizer.encode(text)
        chunks = []
        
        start = 0
        while start < len(tokens):
            end = min(start + self.config.CHUNK_SIZE, len(tokens))
            chunk_tokens = tokens[start:end]
            chunk_text = self.tokenizer.decode(chunk_tokens)
            
            if chunk_text.strip():
                chunks.append(chunk_text.strip())
            
            start = end - self.config.CHUNK_OVERLAP
            if start >= len(tokens):
                break
        
        return chunks
    
    def process_documents(self, documents: List[Dict]) -> List[Dict]:
        """Process all documents into chunks"""
        processed_chunks = []
        
        for doc_idx, doc in enumerate(documents):
            clean_text = self.clean_text(doc['text'])
            chunks = self.chunk_text(clean_text)
            
            for chunk_idx, chunk in enumerate(chunks):
                chunk_id = f"doc_{doc_idx}_chunk_{chunk_idx}"
                processed_chunks.append({
                    'id': chunk_id,
                    'text': chunk,
                    'metadata': {
                        **doc['metadata'],
                        'chunk_id': chunk_id,
                        'doc_index': doc_idx,
                        'chunk_index': chunk_idx,
                        'processed_at': datetime.now().isoformat()
                    }
                })
        
        logger.info(f"✂️ Created {len(processed_chunks)} chunks from {len(documents)} documents")
        return processed_chunks

class VectorStore:
    """ChromaDB vector store management"""
    
    def __init__(self, config: Config):
        self.config = config
        self.client = chromadb.PersistentClient(path=config.CHROMA_DB_PATH)
        self.collection = None
        self.openai_client = OpenAI(api_key=config.OPENAI_API_KEY) if config.OPENAI_API_KEY else None
    
    def setup_collection(self, collection_name: str = "rag_documents"):
        """Setup ChromaDB collection"""
        try:
            # Try to get existing collection
            self.collection = self.client.get_collection(collection_name)
            logger.info(f"📥 Loaded existing collection: {collection_name}")
        except:
            # Create new collection
            self.collection = self.client.create_collection(
                name=collection_name,
                metadata={"description": "RAG document collection"}
            )
            logger.info(f"🆕 Created new collection: {collection_name}")
    
    def create_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Create embeddings using OpenAI"""
        if not self.openai_client:
            logger.warning("⚠️ No OpenAI API key, using mock embeddings")
            return self._mock_embeddings(texts)
        
        try:
            response = self.openai_client.embeddings.create(
                model=self.config.EMBEDDING_MODEL,
                input=texts
            )
            return [item.embedding for item in response.data]
        except Exception as e:
            logger.error(f"❌ OpenAI API error: {e}")
            return self._mock_embeddings(texts)
    
    def _mock_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Create mock embeddings for testing"""
        embeddings = []
        for text in texts:
            # Simple hash-based mock embedding
            import hashlib
            text_hash = hashlib.md5(text.encode()).hexdigest()
            np.random.seed(int(text_hash[:8], 16))
            embedding = np.random.rand(1536).tolist()  # OpenAI embedding dimension
            embeddings.append(embedding)
        return embeddings
    
    def add_documents(self, chunks: List[Dict]):
        """Add document chunks to vector store"""
        if not chunks:
            logger.warning("⚠️ No chunks to add")
            return
        
        texts = [chunk['text'] for chunk in chunks]
        metadatas = [chunk['metadata'] for chunk in chunks]
        ids = [chunk['id'] for chunk in chunks]
        
        logger.info("🔄 Creating embeddings...")
        embeddings = self.create_embeddings(texts)
        
        # Add to collection in batches
        batch_size = 100
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            batch_metadatas = metadatas[i:i + batch_size]
            batch_ids = ids[i:i + batch_size]
            batch_embeddings = embeddings[i:i + batch_size]
            
            self.collection.add(
                documents=batch_texts,
                metadatas=batch_metadatas,
                ids=batch_ids,
                embeddings=batch_embeddings
            )
        
        logger.info(f"💾 Added {len(texts)} chunks to vector store")
    
    def search(self, query: str, top_k: int = None) -> List[Dict]:
        """Search for relevant documents"""
        if top_k is None:
            top_k = self.config.TOP_K
        
        results = self.collection.query(
            query_texts=[query],
            n_results=top_k,
            include=["documents", "metadatas", "distances"]
        )
        
        # Format results
        search_results = []
        for i in range(len(results['documents'][0])):
            search_results.append({
                'text': results['documents'][0][i],
                'metadata': results['metadatas'][0][i],
                'distance': results['distances'][0][i],
                'similarity': 1 - results['distances'][0][i]
            })
        
        return search_results

class RAGGenerator:
    """Generate responses using OpenAI"""
    
    def __init__(self, config: Config):
        self.config = config
        self.openai_client = OpenAI(api_key=config.OPENAI_API_KEY) if config.OPENAI_API_KEY else None
        
        self.system_prompt = """Bạn là trợ lý AI thông minh, trả lời câu hỏi dựa trên tài liệu được cung cấp.

QUY TẮC QUAN TRỌNG:
1. CHỈ trả lời dựa trên nội dung trong các tài liệu được cung cấp
2. Nếu câu hỏi không liên quan đến tài liệu → trả lời "Tôi không biết thông tin này dựa trên tài liệu hiện có"
3. Trích dẫn nguồn tài liệu trong câu trả lời
4. Trả lời ngắn gọn, chính xác và hữu ích

FORMAT TRẢ LỜI:
- Câu trả lời chi tiết
- Nguồn: [Tên file, trang (nếu có)]"""
    
    def generate_response(self, query: str, context_docs: List[Dict]) -> Dict:
        """Generate response from query and context"""
        if not context_docs:
            return {
                "answer": "Tôi không biết thông tin này dựa trên tài liệu hiện có.",
                "confidence": "low",
                "sources": [],
                "context_used": 0
            }
        
        # Prepare context
        context_parts = []
        sources = []
        
        for i, doc in enumerate(context_docs[:3]):  # Use top 3 documents
            context_parts.append(f"[Tài liệu {i+1}]: {doc['text']}")
            
            source_info = {
                "filename": doc['metadata'].get('filename', 'unknown'),
                "page": doc['metadata'].get('page'),
                "similarity": doc['similarity']
            }
            sources.append(source_info)
        
        context = "\n\n".join(context_parts)
        
        # Create prompt
        prompt = f"""Dựa trên các tài liệu sau:

{context}

Câu hỏi: {query}

Hãy trả lời câu hỏi dựa trên thông tin trong tài liệu."""
        
        # Generate response
        if self.openai_client:
            try:
                response = self.openai_client.chat.completions.create(
                    model=self.config.MODEL_NAME,
                    messages=[
                        {"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=500,
                    temperature=0.1
                )
                
                answer = response.choices[0].message.content
                confidence = self._assess_confidence(context_docs)
                
            except Exception as e:
                logger.error(f"❌ OpenAI API error: {e}")
                answer = self._generate_mock_response(query, context_docs)
                confidence = "medium"
        else:
            answer = self._generate_mock_response(query, context_docs)
            confidence = "medium"
        
        return {
            "answer": answer,
            "confidence": confidence,
            "sources": sources,
            "context_used": len(context_docs)
        }
    
    def _generate_mock_response(self, query: str, docs: List[Dict]) -> str:
        """Generate mock response for testing"""
        if not docs:
            return "Tôi không biết thông tin này."
        
        # Simple extractive response
        best_doc = docs[0]
        text_snippet = best_doc['text'][:300] + "..."
        filename = best_doc['metadata'].get('filename', 'document')
        
        return f"Dựa trên tài liệu {filename}: {text_snippet}"
    
    def _assess_confidence(self, docs: List[Dict]) -> str:
        """Assess confidence based on similarity scores"""
        if not docs:
            return "low"
        
        avg_similarity = np.mean([doc['similarity'] for doc in docs])
        
        if avg_similarity > 0.8:
            return "high"
        elif avg_similarity > 0.6:
            return "medium"
        else:
            return "low"

class RAGChatbot:
    """Main RAG Chatbot class"""
    
    def __init__(self, config: Config = None):
        self.config = config or Config()
        self.loader = DocumentLoader()
        self.processor = DocumentProcessor(self.config)
        self.vector_store = VectorStore(self.config)
        self.generator = RAGGenerator(self.config)
        
        self.conversation_history = []
        self.is_initialized = False
        
        logger.info("🤖 RAG Chatbot initialized")
    
    def setup_knowledge_base(self, documents_directory: str = "./documents"):
        """Setup knowledge base from documents directory"""
        logger.info(f"📚 Setting up knowledge base from {documents_directory}")
        
        # Load documents
        documents = self.loader.load_all_documents(documents_directory)
        if not documents:
            # Create sample document if none exist
            self._create_sample_document(documents_directory)
            documents = self.loader.load_all_documents(documents_directory)
        
        # Process documents
        chunks = self.processor.process_documents(documents)
        
        # Setup vector store
        self.vector_store.setup_collection()
        self.vector_store.add_documents(chunks)
        
        self.is_initialized = True
        
        setup_info = {
            "documents_loaded": len(documents),
            "chunks_created": len(chunks),
            "status": "success"
        }
        
        logger.info(f"✅ Knowledge base setup complete: {setup_info}")
        return setup_info
    
    def _create_sample_document(self, directory: str):
        """Create sample document for testing"""
        sample_content = """# Hướng dẫn sử dụng RAG Chatbot

## Giới thiệu
RAG Chatbot là hệ thống trả lời câu hỏi thông minh sử dụng Retrieval-Augmented Generation.
Hệ thống có thể trả lời câu hỏi dựa trên nội dung tài liệu được tải lên.

## Tính năng chính
- Tìm kiếm thông tin từ tài liệu PDF, DOCX, TXT
- Trả lời câu hỏi bằng ngôn ngữ tự nhiên
- Trích dẫn nguồn thông tin rõ ràng
- Đánh giá độ tin cậy của câu trả lời

## Cách sử dụng
1. Tải tài liệu vào thư mục documents/
2. Chạy lệnh setup để xây dựng knowledge base
3. Đặt câu hỏi và nhận câu trả lời với nguồn trích dẫn

## Lưu ý quan trọng
- Hệ thống chỉ trả lời dựa trên tài liệu đã tải
- Kiểm tra nguồn trích dẫn để xác thực thông tin
- Câu trả lời có đánh giá độ tin cậy (high/medium/low)

## Liên hệ hỗ trợ
- Email: support@example.com
- Hotline: 1900-1234
- Website: https://example.com"""
        
        os.makedirs(directory, exist_ok=True)
        sample_file = os.path.join(directory, "sample_knowledge.txt")
        
        with open(sample_file, 'w', encoding='utf-8') as f:
            f.write(sample_content)
        
        logger.info(f"📝 Created sample document: {sample_file}")
    
    def chat(self, question: str) -> Dict:
        """Main chat interface"""
        if not self.is_initialized:
            return {
                "answer": "❌ Chatbot chưa được khởi tạo. Vui lòng chạy setup_knowledge_base() trước.",
                "confidence": "low",
                "sources": [],
                "context_used": 0
            }
        
        if not question.strip():
            return {
                "answer": "Vui lòng đặt câu hỏi cụ thể.",
                "confidence": "low",
                "sources": [],
                "context_used": 0
            }
        
        # Search for relevant documents
        relevant_docs = self.vector_store.search(question, self.config.TOP_K)
        
        # Generate response
        response = self.generator.generate_response(question, relevant_docs)
        
        # Log conversation
        conversation_entry = {
            "timestamp": datetime.now().isoformat(),
            "question": question,
            "answer": response["answer"],
            "confidence": response["confidence"],
            "sources_count": len(response["sources"]),
            "context_used": response["context_used"]
        }
        self.conversation_history.append(conversation_entry)
        
        return response
    
    def get_stats(self) -> Dict:
        """Get chatbot statistics"""
        if not self.conversation_history:
            return {"message": "Chưa có cuộc hội thoại nào"}
        
        total = len(self.conversation_history)
        confidence_dist = {}
        
        for conv in self.conversation_history:
            conf = conv["confidence"]
            confidence_dist[conf] = confidence_dist.get(conf, 0) + 1
        
        return {
            "total_conversations": total,
            "confidence_distribution": confidence_dist,
            "avg_sources_per_answer": sum(c["sources_count"] for c in self.conversation_history) / total,
            "avg_context_used": sum(c["context_used"] for c in self.conversation_history) / total,
            "last_conversation": self.conversation_history[-1]["timestamp"]
        }
    
    def export_history(self, filename: str = None) -> str:
        """Export conversation history"""
        if filename is None:
            filename = f"conversation_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        export_data = {
            "export_info": {
                "system": "RAG Chatbot",
                "export_time": datetime.now().isoformat(),
                "total_conversations": len(self.conversation_history)
            },
            "configuration": {
                "chunk_size": self.config.CHUNK_SIZE,
                "top_k": self.config.TOP_K,
                "model": self.config.MODEL_NAME
            },
            "conversations": self.conversation_history,
            "statistics": self.get_stats()
        }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"📄 Exported conversation history to {filename}")
        return filename

def main():
    """Main function for interactive usage"""
    print("🤖 RAG CHATBOT - COMPLETE SYSTEM")
    print("=" * 50)
    
    # Check environment
    config = Config()
    if not config.OPENAI_API_KEY:
        print("⚠️  WARNING: No OpenAI API key found. Using mock responses.")
        print("💡 Set OPENAI_API_KEY environment variable for full functionality.")
    
    # Initialize chatbot
    chatbot = RAGChatbot(config)
    
    # Setup knowledge base
    try:
        setup_result = chatbot.setup_knowledge_base()
        print(f"✅ Setup completed: {setup_result}")
    except Exception as e:
        print(f"❌ Setup failed: {e}")
        return
    
    # Interactive chat
    print("\n🎯 INTERACTIVE CHAT")
    print("Commands: 'quit' to exit, 'stats' for statistics, 'export' to save history")
    print("-" * 50)
    
    while True:
        try:
            question = input("\n❓ Your question: ").strip()
            
            if question.lower() == 'quit':
                print("👋 Goodbye!")
                break
            elif question.lower() == 'stats':
                stats = chatbot.get_stats()
                print(f"\n📊 STATISTICS:")
                for key, value in stats.items():
                    print(f"   {key}: {value}")
                continue
            elif question.lower() == 'export':
                filename = chatbot.export_history()
                print(f"✅ History exported to: {filename}")
                continue
            elif not question:
                continue
            
            # Get response
            print("🔄 Processing...")
            start_time = time.time()
            response = chatbot.chat(question)
            end_time = time.time()
            
            # Display response
            print(f"\n🤖 **Answer:** {response['answer']}")
            print(f"📊 **Confidence:** {response['confidence']}")
            print(f"⏱️  **Response time:** {end_time - start_time:.2f}s")
            
            if response['sources']:
                print(f"📚 **Sources ({len(response['sources'])}):**")
                for i, source in enumerate(response['sources'][:2], 1):
                    page_info = f" (page {source['page']})" if source['page'] else ""
                    similarity = f" [{source['similarity']:.3f}]"
                    print(f"   {i}. {source['filename']}{page_info}{similarity}")
            
        except KeyboardInterrupt:
            print("\n👋 Interrupted. Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")
    
    # Final export
    if chatbot.conversation_history:
        filename = chatbot.export_history()
        print(f"\n✅ Final conversation history saved to: {filename}")

if __name__ == "__main__":
    main()