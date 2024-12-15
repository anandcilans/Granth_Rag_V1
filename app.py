import streamlit as st
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
import base64

# system.sh
!bash system.sh


OPENAI_API_KEY = st.secrets['OPENAI_API_KEY']

# Load the database
def vectordb_store(selected_db):
    embedding_function = HuggingFaceEmbeddings(
        model_name="Alibaba-NLP/gte-base-en-v1.5",
        model_kwargs={"trust_remote_code": True}  # This allows loading custom model code
    )

    persist_directory = db_options[selected_db]
    print(f"Loading database from {persist_directory}")
    vectordb = Chroma(persist_directory=persist_directory, embedding_function=embedding_function)
    return vectordb

def get_answer(query,selected_db):
    vectordb = vectordb_store(selected_db)
    results = vectordb.similarity_search(query,k=10)
    for res in results:
        print(res)
        print("______________________________________")
   
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

    return chain.run(question=query, context=results)#len(results),

#background
# Function to encode the image to base64
def add_background_image(image_file, opacity):
    with open(image_file, "rb") as file:
        encoded_image = base64.b64encode(file.read()).decode()

    st.markdown(
        f"""
        <style>
        .stApp {{
            background: url(data:image/jpeg;base64,{encoded_image});
            background-size: cover;
            opacity: {opacity};
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# Call the function with your image path and desired opacity
add_background_image("Rectangle 180.png", opacity=0.9)


# Streamlit code
# Title
st.markdown(
    "<h1 style='text-align: center; color:black;'>GRANTH-RAG</h1>", 
    unsafe_allow_html=True
)

st.markdown("<h6 style='text-align: center;color:black;'>Bringing Sacred Knowledge to Life with AI.</h6>", unsafe_allow_html=True)

# Layout for question input and database selection
col1, col2 = st.columns([3, 1])
with col1:
    question = st.text_input("Ask a question")
with col2:
    db_options = {
        "Kamandakiya Niti Sara": "chroma_db_o.7",
        "Shreemad BhagvadGeeta": "srmdbhgvdgeeta_chroma_db_o.7_1"
    }
    selected_db = st.selectbox("Choose Your Reference Book", list(db_options.keys()))

if st.button("➔"):  # Unicode for a right arrow
    if question:
        # Center the spinner
        st.markdown("<div style='text-align: center;'>", unsafe_allow_html=True)
        with st.spinner("Searching... Please wait for a few seconds."):
            answer = get_answer(question, selected_db)
        st.markdown("</div>", unsafe_allow_html=True)
        #st.write(answer)
        st.markdown(f"<div style='font-size:18px';>{answer}</div>",unsafe_allow_html=True)
    else:
        st.write("Please Select a PDF and enter a question.")