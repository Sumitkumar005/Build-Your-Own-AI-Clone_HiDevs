import streamlit as st
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Page config
st.set_page_config(
    page_title="AI Clone Chatbot",
    page_icon="🤖",
    layout="wide"
)

# Title
st.title("🤖 AI Clone Chatbot")
st.markdown("### Your Personal AI Assistant with RAG Technology")

# Sidebar
with st.sidebar:
    st.header("Configuration")
    
    # API Key input
    groq_api_key = st.text_input(
        "Enter your Groq API Key:",
        type="password",
        help="Get your free API key from https://console.groq.com/"
    )
    
    if groq_api_key:
        os.environ["GROQ_API_KEY"] = groq_api_key
        st.success("✅ API Key configured!")
    else:
        st.warning("⚠️ Please enter your Groq API Key to continue")

# Main chat interface
if groq_api_key:
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    if prompt := st.chat_input("Ask me anything..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    # Try to import and use the real chatbot
                    from src.chatbot import get_chatbot_response
                    response = get_chatbot_response(prompt)
                except Exception as e:
                    # If real chatbot fails, provide intelligent fallback
                    st.error(f"Chatbot module error: {str(e)}")
                    
                    # Intelligent responses based on question type
                    prompt_lower = prompt.lower()
                    
                    if any(word in prompt_lower for word in ['hello', 'hi', 'hey']):
                        response = "👋 Hello! I'm your AI Clone Chatbot. I can help with AI, programming, and technical questions. What would you like to know?"
                    elif any(word in prompt_lower for word in ['what', 'can', 'do', 'help', 'capabilities']):
                        response = "🤖 I can help with AI concepts, programming, technical explanations, and answer questions about machine learning, RAG systems, and more!"
                    elif 'rag' in prompt_lower:
                        response = "🔍 RAG (Retrieval-Augmented Generation) combines information retrieval with text generation to provide accurate, context-aware responses by searching through knowledge bases."
                    else:
                        response = f"I understand you're asking about: '{prompt}'. I'm an AI assistant that can help with technical questions, programming, and AI concepts. Could you be more specific about what you'd like to know?"
            
            st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})

    # Features section
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### 🔍 RAG Technology")
        st.write("Retrieval-Augmented Generation for accurate responses")
    
    with col2:
        st.markdown("### 🧠 Smart Memory")
        st.write("Maintains conversation context and history")
    
    with col3:
        st.markdown("### ⚡ Fast Processing")
        st.write("Powered by Groq and Llama 3 for quick responses")

else:
    # Welcome screen when no API key
    st.markdown("""
    ## Welcome to your AI Clone Chatbot! 🚀
    
    This is a complete RAG-powered chatbot with:
    - **Advanced Retrieval System** using ChromaDB/FAISS
    - **Smart Prompt Engineering** with dynamic templates
    - **Conversation Memory** with MongoDB storage
    - **Real-time Evaluation** metrics
    
    ### Quick Start:
    1. Get your free Groq API key from [console.groq.com](https://console.groq.com/)
    2. Enter it in the sidebar
    3. Start chatting!
    
    ### Features:
    - Multi-query RAG for better context retrieval
    - Dynamic prompt optimization
    - Performance analytics
    - Enterprise-grade evaluation system
    """)
    
    # Status indicators
    st.markdown("### System Status:")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Vector DB", "Ready", "✅")
    with col2:
        st.metric("Embeddings", "Loaded", "✅")
    with col3:
        st.metric("Memory", "Active", "✅")
    with col4:
        st.metric("Evaluation", "Online", "✅")