import streamlit as st
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
import base64
from langchain.vectorstores import FAISS
from PIL import Image
import pandas as pd
# Convert the image to a base64 string (for embedding in HTML)
import base64
from io import BytesIO

OPENAI_API_KEY = st.secrets['OPENAI_API_KEY']

def vectordb_store(selected_db):
    embedding_function = HuggingFaceEmbeddings(
        model_name="Alibaba-NLP/gte-base-en-v1.5",
        model_kwargs={"trust_remote_code": True}  # This allows loading custom model code
    )
    
    faiss_index_path = db_options[selected_db]

    # Load the FAISS index with the dangerous deserialization flag enabled
    vectordb = FAISS.load_local(
        folder_path=faiss_index_path,
        embeddings=embedding_function,
        allow_dangerous_deserialization=True
    )

    return vectordb

def get_similarchunks_details(query, selected_db):
    # Initialize the vector store from the selected database
    vectordb = vectordb_store(selected_db)
    
    # Perform similarity search
    results = vectordb.similarity_search(query, k=10)
    
    # Collect page numbers and content into a list
    data = []
    for res in results:
        data.append({
            "Page Number": res.metadata.get("page_number", "Unknown"),
            "Chunk": res.page_content
        })
    
    # Create a DataFrame from the list
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

    return chain.run(question=query, context=results)#len(results),


# Streamlit code
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
add_background_image(r"Rectangle 180.png", opacity=0.9)



# Open and resize the image
image = Image.open("webdevelopment (3).png")
resized_image = image.resize((200, 200))  # Specify new width and height

# Add CSS for centering
st.markdown(
    """
    <style>
    .center-image {
        display: flex;
        justify-content: center;
        align-items: center;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


buffer = BytesIO()
resized_image.save(buffer, format="PNG")
image_base64 = base64.b64encode(buffer.getvalue()).decode()

# Center the image
st.markdown(
    f"""
    <div class="center-image">
        <img src="data:image/png;base64,{image_base64}" alt="Centered Image" width="100">
    </div>
    """,
    unsafe_allow_html=True,
)

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
# Title
#st.markdown(
#    "<h1 style='text-align: center; color:black;'>GRANTH-RAG</h1>", 
#    unsafe_allow_html=True
#)

st.markdown("<h6 style='text-align: center;color:black;'>Bringing Sacred Knowledge to Life with AI.</h6>", unsafe_allow_html=True)

# Layout for question input and database selection
col1, col2 = st.columns([15,8])

with col1:
    question = st.text_area("Ask a question",height=120)

with col2:
    db_options = {
        "Kamandakiya Niti Sara": "faiss_index_kamandakiya_nitisara",
        "Shreemad BhagvadGeeta": "faiss_index_bhagvad_geeta"
    }
    selected_db = st.selectbox("Choose Your Reference Book", list(db_options.keys()))

    languages = ["English", "Gujarati"]
    selected_language = st.selectbox("Language", languages)


if st.button("➔"):  # Unicode for a right arrow
    if question:
        # Center the spinner
        st.markdown("<div style='text-align: center;'>", unsafe_allow_html=True)
        with st.spinner("Searching... Please wait for a few seconds."):
            answer = get_answer(question, selected_db)
            source = get_similarchunks_details(question,selected_db)
        st.markdown("</div>", unsafe_allow_html=True)
        #st.write(answer)
        st.markdown(f"<div style='font-size:18px';>{answer}</div>",unsafe_allow_html=True)
        # Display the DataFrame in Streamlit
        st.subheader("Source of Information : ")
        st.dataframe(source)
        
    else:
        st.write("Please Select a PDF and enter a question.")
 
 