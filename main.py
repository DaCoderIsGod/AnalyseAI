import streamlit as st
import pandas as pd
import pandasai as pai
from pandasai_litellm.litellm import LiteLLM

st.set_page_config(page_title="Data Chatbot", layout="wide")
st.title("Chat with Your Data")

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
                    active_api_key = None
                    if "GROQ_API_KEY" in st.secrets:
                        active_api_key = st.secrets["GROQ_API_KEY"]
                    
                    if not active_api_key:
                        st.error("No API key found. Please provide a Groq API key in the sidebar.")
                    else:
                        llm = LiteLLM(model="groq/qwen/qwen3.8-27b", api_key=active_api_key)

                        pai.config.set({
                            "llm": llm,
                            "save_charts": True,
                            "save_charts_path": "exports/charts"
                        })
                        df = pai.DataFrame(df_raw)
                        response = df.chat(user_query)
                        
                        import os

                    st.subheader("Result")
                    if isinstance(response, str) and (response.endswith('.png') or response.endswith('.jpg') or 'charts' in response):
                        if os.path.exists(response):
                            st.image(response, caption="Generated Chart", use_container_width=True)
                        else:
                            st.write(response)
                    else:
                        st.write(response)
                except Exception as e:
                    st.error(f"An error occurred: {e}")
else:
    st.info("💡 Please upload a CSV file to get started.")
