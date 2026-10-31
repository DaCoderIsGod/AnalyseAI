import streamlit as st
import pandas as pd
import pandasai as pai
from pandasai_litellm.litellm import LiteLLM
import os

st.set_page_config(page_title="Data Chatbot", layout="wide")
st.title("Chat with Your Data")

# FIX 1: Updated types list to accept excel formats
uploaded_file = st.file_uploader("Upload your CSV or Excel dataset", type=["csv", "xlsx", "xls"])

if uploaded_file:
    # FIX 2: Choose the right pandas reader based on file extension
    file_extension = os.path.splitext(uploaded_file.name)[-1].lower()
    
    try:
        if file_extension in [".xlsx", ".xls"]:
            df_raw = pd.read_excel(uploaded_file)
        else:
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
                            st.error("No API key found. Please provide a Groq API key in the secrets dashboard.")
                        else:
                            llm = LiteLLM(
                                model="groq/qwen/qwen3.8-27b", 
                                api_key=active_api_key,
                                max_tokens=500
                            )

                            pai.config.set({
                                "llm": llm,
                                "save_charts": True,
                                "save_charts_path": "exports/charts"
                            })
                            
                            df = pai.DataFrame(df_raw)
                            response = df.chat(user_query)
                            
                            st.subheader("Result")
                            
                            response_str = str(response).strip().strip("'\"{}")
                            
                            if (".png" in response_str or ".jpg" in response_str or "charts" in response_str):
                                if "value':" in response_str:
                                    response_str = response_str.split("value':")[-1].strip("'\" ")
                                
                                if os.path.exists(response_str):
                                    st.image(response_str, caption="Generated Chart", use_container_width=True)
                                else:
                                    st.write(response)
                            else:
                                st.write(response)
                                
                    except Exception as e:
                        st.error(f"An error occurred: {e}")
                        
    except Exception as e:
        st.error(f"Error parsing file: {e}. Please ensure it is a valid format.")
else:
    st.info("💡 Please upload a CSV or Excel file to get started.")
