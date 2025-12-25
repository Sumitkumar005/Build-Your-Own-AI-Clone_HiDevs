import os
import logging
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
import json
from datetime import datetime

# Try different langchain import patterns for compatibility
try:
    from langchain_groq import ChatGroq
except ImportError:
    try:
        from langchain.llms import ChatGroq
    except ImportError:
        ChatGroq = None

try:
    from langchain_community.vectorstores import Chroma
    from langchain_community.embeddings import SentenceTransformerEmbeddings
except ImportError:
    try:
        from langchain.vectorstores import Chroma
        from langchain.embeddings import SentenceTransformerEmbeddings
    except ImportError:
        Chroma = None
        SentenceTransformerEmbeddings = None

try:
    from langchain_community.chat_message_histories import MongoDBChatMessageHistory
except ImportError:
    try:
        from langchain.memory import MongoDBChatMessageHistory
    except ImportError:
        MongoDBChatMessageHistory = None

try:
    from langchain.chains import ConversationalRetrievalChain
    from langchain.prompts import PromptTemplate
    from langchain.schema import Document
except ImportError:
    ConversationalRetrievalChain = None
    PromptTemplate = None
    Document = None

try:
    from langchain.retrievers import ContextualCompressionRetriever
    from langchain.retrievers.document_compressors import LLMChainExtractor
    from langchain.retrievers.multi_query import MultiQueryRetriever
except ImportError:
    ContextualCompressionRetriever = None
    LLMChainExtractor = None
    MultiQueryRetriever = None

try:
    from src.prompt import prompt_manager
except ImportError:
    prompt_manager = None

load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AIChatbot:
    """Advanced AI Chatbot with RAG, dynamic prompts, and evaluation capabilities"""

    def __init__(self, model_name: str = "llama3-8b-8192", temperature: float = 0.1):
        """Initialize the AI Chatbot with advanced RAG features"""
        self.model_name = model_name
        self.temperature = temperature

        # Initialize Groq LLM with specified model
        groq_api_key = os.getenv("GROQ_API_KEY")
        if groq_api_key and groq_api_key != "your_groq_api_key_here":
            try:
                self.llm = ChatGroq(
                    groq_api_key=groq_api_key,
                    model_name=self.model_name,
                    temperature=self.temperature,
                    max_tokens=2048
                )
                logger.info(f"Groq LLM initialized with model: {self.model_name}")
            except Exception as e:
                logger.warning(f"Failed to initialize Groq LLM: {e}")
                self.llm = None
        else:
            logger.warning("Groq API key not found or not set properly")
            self.llm = None

        # Initialize embeddings
        try:
            self.embeddings = SentenceTransformerEmbeddings(
                model_name="all-MiniLM-L6-v2"
            )
            logger.info("Embeddings initialized successfully")
        except Exception as e:
            logger.warning(f"Failed to initialize embeddings: {e}")
            self.embeddings = None

        # Initialize vector store
        self.vectorstore = None
        self.memory = None
        self.chain = None

        # Advanced RAG components
        self.multi_query_retriever = None
        self.compression_retriever = None

        # Conversation analytics
        self.conversation_stats = {
            'total_queries': 0,
            'avg_response_time': 0,
            'successful_responses': 0,
            'failed_responses': 0
        }

        logger.info(f"AIChatbot initialized with model: {self.model_name}")

    def initialize_vectorstore(self, persist_directory: str = "./chroma_db"):
        """Initialize or load ChromaDB vector store"""
        try:
            self.vectorstore = Chroma(
                persist_directory=persist_directory,
                embedding_function=self.embeddings
            )
            logger.info("Vector store initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize vector store: {e}")
            raise

    def initialize_memory(self, session_id: str, connection_string: str = None):
        """Initialize MongoDB memory for conversation history"""
        try:
            if not connection_string:
                connection_string = os.getenv("MONGODB_URI", "mongodb://localhost:27017")

            self.memory = MongoDBChatMessageHistory(
                connection_string=connection_string,
                session_id=session_id,
                database_name="ai_chatbot",
                collection_name="conversations"
            )
            logger.info(f"Memory initialized for session: {session_id}")
        except Exception as e:
            logger.warning(f"Failed to initialize MongoDB memory: {e}. Using in-memory storage.")
            # Fallback to in-memory if MongoDB fails
            from langchain.memory import ConversationBufferMemory
            self.memory = ConversationBufferMemory(
                memory_key="chat_history",
                return_messages=True
            )

    def setup_advanced_rag(self):
        """Set up advanced RAG components: multi-query and compression"""
        if not self.vectorstore:
            raise ValueError("Vector store not initialized")

        # Multi-query retriever for query expansion
        self.multi_query_retriever = MultiQueryRetriever.from_llm(
            retriever=self.vectorstore.as_retriever(search_kwargs={"k": 5}),
            llm=self.llm
        )

        # Contextual compression retriever
        compressor = LLMChainExtractor.from_llm(self.llm)
        self.compression_retriever = ContextualCompressionRetriever(
            base_compressor=compressor,
            base_retriever=self.multi_query_retriever
        )

        logger.info("Advanced RAG components initialized")

    def create_chain(self):
        """Create the conversational retrieval chain with dynamic prompt selection"""
        if not self.vectorstore or not self.memory:
            raise ValueError("Vector store and memory must be initialized")

        # Use advanced retriever if available, otherwise use basic
        retriever = self.compression_retriever or self.vectorstore.as_retriever(search_kwargs={"k": 3})

        # Base prompt template - will be customized per query
        prompt_template = """
        You are an AI assistant with expertise in multiple domains. Use the following context and structured reasoning to provide accurate, helpful responses.

        Context: {context}
        Chat History: {chat_history}
        Question: {question}

        Answer:
        """

        PROMPT = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "chat_history", "question"]
        )

        chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=retriever,
            memory=self.memory,
            combine_docs_chain_kwargs={"prompt": PROMPT},
            return_source_documents=True,
            verbose=True
        )

        self.chain = chain
        logger.info("Conversational chain created successfully")
        return chain

    def get_dynamic_prompt(self, question: str, context: str = None, chat_history: str = None) -> str:
        """Get dynamically selected prompt based on query characteristics"""
        return prompt_manager.get_prompt_for_query(
            query=question,
            context=context,
            chat_history=chat_history
        )

    def chat(self, question: str, session_id: str = "default") -> Dict[str, Any]:
        """Process a chat query with comprehensive response and analytics"""
        start_time = datetime.now()

        try:
            # Initialize components if not done
            if not self.vectorstore:
                self.initialize_vectorstore()
            if not self.memory:
                self.initialize_memory(session_id)
            if not self.chain:
                self.create_chain()

            # Get dynamic prompt for this query
            dynamic_prompt = self.get_dynamic_prompt(question)

            # Update chain with dynamic prompt
            self._update_chain_prompt(dynamic_prompt)

            # Get response from chain
            response = self.chain({"question": question})

            # Extract answer and sources
            answer = response.get("answer", "")
            sources = response.get("source_documents", [])

            # Calculate response time
            response_time = (datetime.now() - start_time).total_seconds()

            # Update conversation statistics
            self._update_conversation_stats(success=True, response_time=response_time)

            # Prepare comprehensive response
            result = {
                "answer": answer,
                "sources": sources,
                "confidence_score": self._calculate_confidence_score(answer, sources),
                "response_time": response_time,
                "model_used": self.model_name,
                "session_id": session_id,
                "timestamp": datetime.now().isoformat(),
                "analytics": self._get_query_analytics(question, answer, sources)
            }

            logger.info(f"Query processed successfully in {response_time:.2f}s")
            return result

        except Exception as e:
            # Update failure statistics
            self._update_conversation_stats(success=False)

            logger.error(f"Error processing query: {e}")
            return {
                "answer": "I apologize, but I encountered an error while processing your question. Please try again.",
                "error": str(e),
                "session_id": session_id,
                "timestamp": datetime.now().isoformat()
            }

    def _update_chain_prompt(self, prompt_template: str):
        """Update the chain's prompt template dynamically"""
        if self.chain:
            PROMPT = PromptTemplate(
                template=prompt_template,
                input_variables=["context", "chat_history", "question"]
            )
            self.chain.combine_docs_chain.llm_chain.prompt = PROMPT

    def _calculate_confidence_score(self, answer: str, sources: List[Document]) -> float:
        """Calculate confidence score based on answer quality and source relevance"""
        if not sources:
            return 0.3  # Low confidence without sources

        # Basic confidence calculation
        source_count = len(sources)
        answer_length = len(answer.split())

        # Higher confidence for more sources and substantial answers
        confidence = min(0.9, (source_count * 0.1) + (min(answer_length, 100) * 0.005))

        return round(confidence, 2)

    def _update_conversation_stats(self, success: bool, response_time: float = None):
        """Update conversation statistics"""
        self.conversation_stats['total_queries'] += 1

        if success:
            self.conversation_stats['successful_responses'] += 1
            if response_time:
                # Update rolling average
                current_avg = self.conversation_stats['avg_response_time']
                total_queries = self.conversation_stats['total_queries']
                self.conversation_stats['avg_response_time'] = (
                    (current_avg * (total_queries - 1)) + response_time
                ) / total_queries
        else:
            self.conversation_stats['failed_responses'] += 1

    def _get_query_analytics(self, question: str, answer: str, sources: List[Document]) -> Dict[str, Any]:
        """Generate analytics for the current query"""
        return {
            "question_length": len(question.split()),
            "answer_length": len(answer.split()),
            "sources_used": len(sources),
            "conversation_stats": self.conversation_stats.copy()
        }

    def get_conversation_history(self, session_id: str) -> List[Dict[str, Any]]:
        """Retrieve conversation history for a session"""
        if hasattr(self.memory, 'messages'):
            return [{"role": "user" if i % 2 == 0 else "assistant",
                    "content": msg.content,
                    "timestamp": getattr(msg, 'timestamp', None)}
                   for i, msg in enumerate(self.memory.messages)]
        return []

    def clear_memory(self, session_id: str):
        """Clear conversation memory for a session"""
        if self.memory:
            self.memory.clear()
            logger.info(f"Memory cleared for session: {session_id}")

    def export_conversation(self, session_id: str, format: str = "json") -> str:
        """Export conversation history in specified format"""
        history = self.get_conversation_history(session_id)

        if format == "json":
            return json.dumps(history, indent=2, default=str)
        elif format == "txt":
            return "\n".join([f"{msg['role'].title()}: {msg['content']}" for msg in history])
        else:
            raise ValueError(f"Unsupported format: {format}")

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get comprehensive performance metrics"""
        return {
            "model_info": {
                "name": self.model_name,
                "temperature": self.temperature
            },
            "conversation_stats": self.conversation_stats,
            "system_status": {
                "vectorstore_ready": self.vectorstore is not None,
                "memory_ready": self.memory is not None,
                "chain_ready": self.chain is not None,
                "advanced_rag_ready": self.compression_retriever is not None
            }
        }

# Global chatbot instance
_chatbot_instance = None

def get_chatbot_response(question: str, session_id: str = "streamlit_session") -> str:
    """Simple interface function for Streamlit app"""
    global _chatbot_instance
    
    try:
        # Check if vector database exists first
        import os
        if not os.path.exists("./chroma_db"):
            return "❌ Vector database not found! Please run: python initialize_kb.py"
        
        # Initialize chatbot if not exists
        if _chatbot_instance is None:
            _chatbot_instance = AIChatbot()
            
        # Initialize vector store if not done
        if _chatbot_instance.vectorstore is None:
            _chatbot_instance.initialize_vectorstore()
            
        # If no Groq API key, use vector search directly
        if _chatbot_instance.llm is None:
            # Direct vector search without LLM
            try:
                results = _chatbot_instance.vectorstore.similarity_search(question, k=3)
                if results:
                    # Combine the top results
                    context = "\n\n".join([doc.page_content for doc in results])
                    sources = [doc.metadata.get('source', 'Unknown') for doc in results]
                    
                    response = f"""**Based on your knowledge base:**

{context[:2000]}...

**Sources:** {', '.join(set(sources))}

*Note: For enhanced AI responses, add your Groq API key in the sidebar.*"""
                    return response
                else:
                    return f"I couldn't find information about '{question}' in the knowledge base."
            except Exception as e:
                return f"Error searching knowledge base: {str(e)}"
        
        # Use full RAG system with LLM
        result = _chatbot_instance.chat(question, session_id)
        return result.get("answer", "I'm sorry, I couldn't process your question.")
        
    except Exception as e:
        logger.error(f"Error in get_chatbot_response: {e}")
        return f"Error: {str(e)}"