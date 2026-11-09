import streamlit as st
from ingestion import get_transcript, split_transcript, extract_video_id
from generation import build_rag_chain
from retrieval import build_vector_store, get_multiquery_reranking_retriever

st.title("YouTube RAG Chatbot")

# 1. Initialize "Memory" for the app
if "chain" not in st.session_state:
    st.session_state.chain = None
if "retriever" not in st.session_state:
    st.session_state.retriever = None
if "messages" not in st.session_state:
    st.session_state.messages = []  # This will hold our chat history

with st.sidebar:
    #  It creates the text box in the sidebar.
    video_input = st.text_input("YouTube video ID or URL", "sRH-XlBE2cQ")
    
    if st.button("Load video"):
        with st.spinner("Loading and processing video..."):
            try:
                vid = extract_video_id(video_input)
                text = get_transcript(vid)
                chunks = split_transcript(text, vid)
                vs = build_vector_store(chunks)
                
                # Now using the multiquery + reranking retriever
                retriever = get_multiquery_reranking_retriever(vs)
                
                # Save to memory
                st.session_state.chain = build_rag_chain(retriever)
                st.session_state.retriever = retriever
                
                # Clear previous chat when a NEW video is loaded
                st.session_state.messages = [] 
                
                # Requested custom message
                st.success(f"✅ Video loaded! You are ready to ask questions.")
            except Exception as e:
                st.error(f"Error loading video: {e}")

# 2. Display the chat history from memory
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 3. Handle new questions
if question := st.chat_input("Ask a question"):
    if st.session_state.chain is None:
        st.warning("Please click 'Load video' first!")
    else:
        # Add user question to memory and display
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        # Generate and display answer
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    answer = st.session_state.chain.invoke(question)
                    st.markdown(answer)
                    
                    # Show the retrieved chunks
                    with st.expander("Retrieved chunks"):
                        docs = st.session_state.retriever.invoke(question)
                        for d in docs:
                            st.write(d.page_content[:300] + "...")
                            st.divider()
                    
                    # Add assistant answer to memory
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                except Exception as e:
                    st.error(f"Error generating answer: {e}")