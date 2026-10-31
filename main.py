import streamlit as st
import pandas as pd
import pandasai as pai
from pandasai_litellm.litellm import LiteLLM

# Page Title
st.set_page_config(page_title="PandasAI Data Chatbot", layout="wide")
st.title("🐼 Chat with Your Data using PandasAI")

# Sidebar Configuration for optional Groq API Key
st.sidebar.header("Configuration")
user_provided_key = st.sidebar.text_input(
    "Enter your Groq API Key (Optional)", 
    type="password", 
    placeholder="gsk_..."
)

# Dynamic File Uploader
uploaded_file = st.file_uploader("Upload your CSV dataset", type=["csv"])

if uploaded_file:
    df_raw = pd.read_csv(uploaded_file)
    
    st.subheader("Data Preview")
    st.dataframe(df_raw.head(), use_container_width=True)
    
    st.subheader("Ask a Question or Request a Chart")
    user_query = st.text_input("Example: 'Show me the average revenue by category'")
    
    if st.button("Run Query"):
        if not user_query.strip():
            st.warning("Please enter a question.")
        else:
            with st.spinner("AI is analyzing your data..."):
                try:
                    # Smart Fallback Logic:
                    # 1. Use user's key if provided
                    # 2. Otherwise, look for your secret key in st.secrets
                    # 3. If neither exists, raise an error
                    
                    active_api_key = None
                    
                    if user_provided_key.strip():
                        active_api_key = user_provided_key
                    elif "GROQ_API_KEY" in st.secrets:
                        active_api_key = st.secrets["GROQ_API_KEY"]
                    
                    if not active_api_key:
                        st.error("No API key found. Please provide a Groq API key in the sidebar.")
                    else:
                        llm = LiteLLM(model="groq/qwen/qwen3.8-27b", api_key=active_api_key)

                        
                        # Apply configuration globally
                        pai.config.set({
                            "llm": llm,
                            "save_charts": True,
                            "save_charts_path": "exports/charts"
                        })
                        
                        # Convert to PandasAI DataFrame and chat
                        df = pai.DataFrame(df_raw)
                        response = df.chat(user_query)
                        
                        st.subheader("Result")
                        st.write(response)
                        
                except Exception as e:
                    st.error(f"An error occurred: {e}")
else:
    st.info("💡 Please upload a CSV file to get started.")
