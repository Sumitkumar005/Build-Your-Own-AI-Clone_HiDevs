#!/usr/bin/env python3
"""
Professional Knowledge Base Initialization for AI Clone Chatbot
Processes real data from PDFs, texts, and unstructured sources
"""

import os
import sys
from pathlib import Path

def initialize_knowledge_base():
    """Initialize the vector database with real data from data folder"""
    print("🚀 Initializing Professional Knowledge Base...")
    
    try:
        from langchain_community.vectorstores import Chroma
        from langchain_community.embeddings import SentenceTransformerEmbeddings
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        from langchain.schema import Document
        import json
    except ImportError as e:
        print(f"❌ Missing dependencies: {e}")
        print("Run: pip install -r requirements.txt")
        return False
    
    # Initialize embeddings
    print("📊 Loading embedding model...")
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # Collect documents from all data sources
    documents = []
    data_dir = Path("data")
    
    # 1. Process text files
    print("📄 Processing text files...")
    text_dir = data_dir / "texts"
    if text_dir.exists():
        for text_file in text_dir.glob("*.txt"):
            try:
                with open(text_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    if content.strip():
                        doc = Document(
                            page_content=content,
                            metadata={
                                "source": str(text_file),
                                "type": "text_file",
                                "category": "knowledge_base"
                            }
                        )
                        documents.append(doc)
                        print(f"  ✅ Loaded: {text_file.name}")
            except Exception as e:
                print(f"  ⚠️ Could not load {text_file}: {e}")
    
    # 2. Process unstructured data
    print("📊 Processing unstructured data...")
    unstructured_dir = data_dir / "unstructured"
    if unstructured_dir.exists():
        # Process product reviews
        reviews_file = unstructured_dir / "product_reviews.txt"
        if reviews_file.exists():
            try:
                with open(reviews_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    doc = Document(
                        page_content=content,
                        metadata={
                            "source": str(reviews_file),
                            "type": "product_reviews",
                            "category": "customer_feedback"
                        }
                    )
                    documents.append(doc)
                    print(f"  ✅ Loaded: {reviews_file.name}")
            except Exception as e:
                print(f"  ⚠️ Could not load {reviews_file}: {e}")
        
        # Process JSON files
        for json_file in unstructured_dir.glob("*.json"):
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    # Convert JSON to text representation
                    content = json.dumps(data, indent=2)
                    doc = Document(
                        page_content=content,
                        metadata={
                            "source": str(json_file),
                            "type": "json_data",
                            "category": "structured_data"
                        }
                    )
                    documents.append(doc)
                    print(f"  ✅ Loaded: {json_file.name}")
            except Exception as e:
                print(f"  ⚠️ Could not load {json_file}: {e}")
    
    # 3. Process PDFs
    print("📚 Processing PDF files...")
    pdf_dir = data_dir / "pdfs"
    if pdf_dir.exists():
        try:
            from pypdf import PdfReader
            for pdf_file in pdf_dir.glob("*.pdf"):
                try:
                    reader = PdfReader(pdf_file)
                    text = ""
                    for page in reader.pages:
                        text += page.extract_text() + "\n"
                    
                    if text.strip():
                        doc = Document(
                            page_content=text,
                            metadata={
                                "source": str(pdf_file),
                                "type": "pdf_document",
                                "category": "document",
                                "pages": len(reader.pages)
                            }
                        )
                        documents.append(doc)
                        print(f"  ✅ Loaded: {pdf_file.name} ({len(reader.pages)} pages)")
                except Exception as e:
                    print(f"  ⚠️ Could not load {pdf_file}: {e}")
        except ImportError:
            print("  ⚠️ PyPDF not available, skipping PDF processing")
    
    if not documents:
        print("❌ No documents found! Please add data to the data/ folder")
        return False
    
    print(f"📄 Total documents loaded: {len(documents)}")
    
    # Split documents into chunks
    print("✂️ Splitting documents into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )
    
    split_docs = text_splitter.split_documents(documents)
    print(f"📄 Created {len(split_docs)} document chunks")
    
    # Create vector store
    persist_directory = "./chroma_db"
    
    # Remove existing database if it exists
    if os.path.exists(persist_directory):
        import shutil
        shutil.rmtree(persist_directory)
        print("🗑️ Removed existing database")
    
    # Create new vector store
    print("💾 Creating vector database...")
    vectorstore = Chroma.from_documents(
        documents=split_docs,
        embedding=embeddings,
        persist_directory=persist_directory
    )
    
    # Persist the database
    vectorstore.persist()
    
    print(f"✅ Knowledge base initialized successfully!")
    print(f"📁 Vector database saved to: {persist_directory}")
    print(f"📊 Total chunks: {len(split_docs)}")
    
    # Test the database
    print("\n🧪 Testing knowledge base...")
    test_queries = [
        "What products are reviewed?",
        "Tell me about headphones",
        "What is artificial intelligence?",
        "Customer feedback"
    ]
    
    for query in test_queries:
        try:
            results = vectorstore.similarity_search(query, k=2)
            if results:
                print(f"✅ Query '{query}': Found {len(results)} results")
                print(f"   Sample: {results[0].page_content[:100]}...")
            else:
                print(f"⚠️ Query '{query}': No results found")
        except Exception as e:
            print(f"❌ Query '{query}' failed: {e}")
    
    return True

if __name__ == "__main__":
    try:
        success = initialize_knowledge_base()
        if success:
            print("\n🎉 Knowledge base setup complete!")
            print("🚀 Now restart your Streamlit app: streamlit run app.py")
            print("💡 Your chatbot can now answer questions about:")
            print("   - Product reviews and customer feedback")
            print("   - AI and technology concepts")
            print("   - Any content from your data files")
        else:
            print("\n❌ Knowledge base setup failed!")
            sys.exit(1)
    except Exception as e:
        print(f"❌ Error initializing knowledge base: {e}")
        sys.exit(1)