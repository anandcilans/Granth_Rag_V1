from langchain.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS,DistanceStrategy
import streamlit as st
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI

from langchain.vectorstores import FAISS
import pandas as pd
import os
import shutil  # For cleaning up merged databases


OPENAI_API_KEY = st.secrets['OPENAI_API_KEY']

# Get API key with fallback
if not OPENAI_API_KEY:
    st.error("OpenAI API key not found in environment variables")
    OPENAI_API_KEY = st.text_input("Enter your OpenAI API key:", type="password")
    if OPENAI_API_KEY:
        os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
    else:
        st.stop()

def vectordb_store(selected_db):

    embedding_function = HuggingFaceEmbeddings(
        model_name="Alibaba-NLP/gte-base-en-v1.5",
        model_kwargs={"trust_remote_code": True}
    )
    
    # Check if it's a merged database (by path instead of key)
    if selected_db == "Merged_db":
        faiss_index_path = selected_db
    else:
        # Look up path using the human-readable key from db_options
        faiss_index_path = db_options[selected_db]

    vectordb = FAISS.load_local(
        folder_path=faiss_index_path,
        embeddings=embedding_function,
        allow_dangerous_deserialization=True
    )
    return vectordb

def get_similarchunks_details(query, selected_db):
    
    vectordb = vectordb_store(selected_db)
    results = vectordb.similarity_search(query, k=10)
    data = []
    for res in results:
        data.append({
            "Page Number": res.metadata.get("page_number", "Unknown"),
            "Book Name": res.metadata.get("book_name", "Unknown"),
            "Chunk": res.page_content
        })

    df = pd.DataFrame(data)
    df.index = range(1, len(df) + 1)
    return df


def get_answer(query,selected_db):
    vectordb = vectordb_store(selected_db)
    results = vectordb.similarity_search(query,k=10)
   
    prompt_template = PromptTemplate(
        input_variables=['query', 'context'],
        template="""
        You are an expert assistant specializing in question-answering using religious texts, including holy books, stories, history books and research papers, also u are a good summarizer.

        Your task is to thoroughly understand the provided context and answer the user's question as accurately and clearly as possible. Keep your response concise, human-friendly, and to the point. If user asks casual question than you can communicate , keep conversation more human like.

        Important:

        - Only answer based on the provided context. Do not fabricate or make assumptions.
        - Sometime user will make spelling mistake in question , you have to understand the question and provide answer.
        - If no relevant answer can be found, respond with "Answer is not available in this book."

        - Avoid mentioning the source of the answer in your response, such as "according to the context."

        Context: {context}

        Question: {question}

        Answer:
        """
    )
    
    llm = ChatOpenAI(model='gpt-4o-mini',temperature=0,openai_api_key=OPENAI_API_KEY)#,max_tokens=100
    chain = LLMChain(llm=llm, prompt=prompt_template)

    return chain.run(question=query, context=results)

def vectordb_store2(selected_db):
    embedding_model = OpenAIEmbeddings(model="text-embedding-3-large")
    
    faiss_index_path = db_options[selected_db]
    vectordb = FAISS.load_local(
        folder_path=faiss_index_path,
        embeddings=embedding_model,
        distance_strategy=DistanceStrategy.COSINE,
        allow_dangerous_deserialization=True
    )

    return vectordb

def get_similarchunks_details2(query, selected_db):

    vectordb = vectordb_store2(selected_db)
    results = vectordb.similarity_search(query, k=10)
    data = []
    for res in results:
        data.append({
            "Page Number": res.metadata.get("page_number", "Unknown"),
            "Chunk": res.page_content
        })

    df = pd.DataFrame(data)
    df.index = range(1, len(df) + 1)
    return df

def get_answer2(query,selected_db):
    vectordb = vectordb_store2(selected_db)
    results = vectordb.similarity_search(query,k=10)
   
    prompt_template = PromptTemplate(
        input_variables=['query', 'context'],
        template="""
        You are an expert assistant specializing in question-answering on holy books, stories, history books and research papers, also u are a good summarizer.

        Your task is to thoroughly understand the provided context and answer the user's question as accurately and clearly as possible. Keep your response concise, human-friendly, and to the point. If user asks casual question than you can communicate , keep conversation more human like.

        Important:

        - Only answer based on the provided context. Do not fabricate or make assumptions.
        - Sometime user will make spelling mistake in question , you have to understand the question and provide answer.
        - If no relevant answer can be found, respond with "Answer is not available in this book."

        - Avoid mentioning the source of the answer in your response, such as "according to the context."

        Context: {context}

        Question: {question}

        Answer:
        """
    )
    
    
    llm = ChatOpenAI(model='gpt-4o-mini',temperature=0,openai_api_key=OPENAI_API_KEY)#,max_tokens=100
    chain = LLMChain(llm=llm, prompt=prompt_template)

    return chain.run(question=query, context=results)

def merge_faiss_databases(db_paths, merged_db_path="Merged_db"):
    
    if not db_paths:
        raise ValueError("No databases to merge")
    
    embeddings = HuggingFaceEmbeddings(
        model_name="Alibaba-NLP/gte-base-en-v1.5",
        model_kwargs={"trust_remote_code": True}
    )
    
    merged_db = FAISS.load_local(
        db_paths[0], 
        embeddings,
        allow_dangerous_deserialization=True
    )
    
    for db_path in db_paths[1:]:
        current_db = FAISS.load_local(
            db_path,
            embeddings,
            allow_dangerous_deserialization=True
        )
        merged_db.merge_from(current_db)
    
    os.makedirs(merged_db_path, exist_ok=True)
    merged_db.save_local(merged_db_path)
    print(f"Merged {len(db_paths)} databases into {merged_db_path}")
    return merged_db_path

# Streamlit code

# Add custom CSS to control the input box width
st.markdown(
    """
    <style>
        /* Targeting the input box under 'Ask a question' */
        div[data-testid="stTextInput"] > div > div > input {
            width: 100% !important; /* Adjust the width as needed */
            
        }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown("<h6 style='text-align: center;color:black;'>Bringing Sacred Knowledge to Life with AI.</h6>", unsafe_allow_html=True)

col1, col2 = st.columns([15, 8])

with col1:
    question = st.text_area("Ask a question", height=120)

with col2:
    languages = ["English", "Gujarati"]
    selected_language = st.radio("Language", languages)

    if selected_language == "English":
        db_options = {
            "Books": "faiss_index_english_new_files",
            "Articles": "articles_faissdb28jan",
            "Home Front to Battlefront An Ohio Teenager in World War II":"Home_Front___faissdb28jan"
        }
    elif selected_language == "Gujarati":
        db_options = {
            "શ્રીમદ ભાગવદ ગીતા": "faiss_guj_smd_bhagvatam"
        }
    else:
        db_options = {}

    # Multiselect for database selection
    if db_options:
        selected_dbs = st.multiselect(
            "Select reference books (You can select multiple books here)",
            options=list(db_options.keys()),
            default=list(db_options.keys())[0]
        )

        selected_db = [db_options[db] for db in selected_dbs]

    else:
        st.warning("No database options available for the selected language.")


if st.button("➔"):
    if question:

        st.markdown("<div style='text-align: center;'>", unsafe_allow_html=True)
        with st.spinner("Searching... Please wait for a few seconds."):
            if selected_language == "English":
                if len(selected_dbs) == 0:
                    st.error("Please select at least one database")
                    st.stop()
                
                # Get actual paths from selected database names
                db_paths = [db_options[db] for db in selected_dbs]
                
                if len(db_paths) == 1:
                    # Single database case
                    db_key = selected_dbs[0]
                    answer = get_answer(question, db_key)
                    source = get_similarchunks_details(question, db_key)
                else:
                    #write a code to remoce thw folder named Merged_db
                    if os.path.exists("Merged_db"):
                        shutil.rmtree("Merged_db")

                    # Multiple databases case
                    merged_db_path = "Merged_db"
                    try:
                        # Clear previous merged database
                        if os.path.exists(merged_db_path):
                            shutil.rmtree(merged_db_path)
                            
                        # Merge all selected databases
                        merged_path = merge_faiss_databases(db_paths, merged_db_path)
                        answer = get_answer(question, merged_path)
                        source = get_similarchunks_details(question, merged_path)
                    except Exception as e:
                        st.error(f"Error merging databases: {str(e)}")
                        st.stop()
                        
            elif selected_language == "Gujarati":

                answer = get_answer2(question, selected_dbs[0])
                source = get_similarchunks_details2(question, selected_dbs[0])

        st.markdown("</div>", unsafe_allow_html=True)
        st.markdown(f"<div style='font-size:18px';>{answer}</div>", unsafe_allow_html=True)
        st.subheader("Source of Information : ")
        st.dataframe(source)
    else:
        st.write("Please enter a question.") 
