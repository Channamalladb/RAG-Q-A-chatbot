import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import HuggingFaceHub
from langchain.chains import ConversationalRetrievalChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.memory import ConversationBufferMemory
from langchain.schema import Document
import os
from huggingface_hub import login

# --- App Config ---
st.set_page_config(page_title="Customer Segmentation + RAG", layout="wide")
st.title("🧠 Customer Segmentation + RAG Chatbot")

# --- Configuration ---
# IMPORTANT: Configure your Hugging Face API key and Model ID here.
# Option 1: Set them as environment variables (recommended for security)
#   - HUGGINGFACEHUB_API_TOKEN
#   - MODEL_ID
# Option 2: Hardcode the values directly in the script (less secure)
API_KEY = os.getenv("HUGGINGFACEHUB_API_TOKEN", "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")
MODEL_ID = os.getenv("MODEL_ID", "HuggingFaceTB/SmolLM3-3B-Base")


# --- Sidebar ---
st.sidebar.header("Settings")
top_k = st.sidebar.slider("Top K Docs", 1, 10, 4)
num_clusters = st.sidebar.slider("Number of Segments (k)", 2, 7, 5)

# --- Load Dataset ---
@st.cache_data
def load_data():
    """Loads and preprocesses the customer data."""
    df = pd.read_csv("Mall_Customers.csv")
    df.rename(columns={"Annual Income (k$)": "Income", "Spending Score (1-100)": "Score"}, inplace=True)
    return df

df = load_data()

# --- KMeans Clustering ---
@st.cache_resource
def run_kmeans(data, n_clusters):
    """Fits the KMeans model to the data."""
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    kmeans.fit(data[['Income', 'Score']])
    return kmeans

kmeans = run_kmeans(df, num_clusters)
df['Cluster'] = kmeans.predict(df[['Income', 'Score']])


# --- Visualizations ---
st.subheader("📊 Customer Segmentation")
fig, ax = plt.subplots(figsize=(10, 6))
sns.scatterplot(data=df, x="Income", y="Score", hue="Cluster", palette="viridis", s=100, alpha=0.7, ax=ax)
ax.set_title("K-Means Clustering of Mall Customers")
st.pyplot(fig)

with st.expander("📁 View Full Dataset with Cluster Assignments"):
    st.dataframe(df)

# --- RAG Q&A Chatbot ---
st.markdown("---")
st.header("💬 Ask the Chatbot About the Customers")

@st.cache_resource
def create_retriever(data, k):
    """Creates a FAISS vector store and retriever from the customer data."""
    documents = [
        Document(page_content=f"Customer {row['CustomerID']} is a {row['Age']} year old {row['Gender']} with an annual income of ${row['Income']}k and a spending score of {row['Score']}. They belong to cluster {row['Cluster']}.")
        for _, row in data.iterrows()
    ]
    
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    split_docs = splitter.split_documents(documents)
    
    embeddings = HuggingFaceEmbeddings()
    vectorstore = FAISS.from_documents(split_docs, embeddings)
    
    return vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": k})

# --- Chat Interface ---
if not API_KEY:
    st.warning("🔑 Hugging Face API key is not configured. Please set the HUGGINGFACEHUB_API_TOKEN environment variable to activate the chatbot.")
else:
    try:
        # Programmatically login to Hugging Face Hub to authenticate all downloads
        login(token=API_KEY)

        # Create the retriever
        retriever = create_retriever(df, top_k)
        
        # Initialize LLM and Conversation Chain
        llm = HuggingFaceHub(repo_id=MODEL_ID, huggingfacehub_api_token=API_KEY)
        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        qa_chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=retriever, memory=memory)

        # Initialize chat history
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []

        # Get user input
        user_q = st.text_input("Ask a question like: 'Tell me about the customers in cluster 2.'")
        if user_q:
            with st.spinner("Thinking..."):
                response = qa_chain.run(user_q)
                st.session_state.chat_history.append((user_q, response))
        
        # Display chat history
        if st.session_state.chat_history:
            for q, a in reversed(st.session_state.chat_history):
                st.markdown(f"**You:** {q}")
                st.markdown(f"**Bot:** {a}")
    except Exception as e:
        st.error(f"An error occurred with the chatbot. Please ensure your Hugging Face API key is correct and has the necessary permissions. Error: {e}")


# --- Footer ---
st.markdown("---")
st.markdown("Made with ❤️ by [Your Name]")
