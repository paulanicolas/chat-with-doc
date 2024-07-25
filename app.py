import os
import time
import json
import boto3
import streamlit as st
from langchain.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA, ConversationChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.memory import ConversationBufferMemory
from langchain.llms.bedrock import Bedrock
from langchain_community.chat_models import BedrockChat
from langchain_community.embeddings import BedrockEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.embeddings.openai import OpenAIEmbeddings
#from langchain.llms.openai import OpenAI
#import openai
from openai import OpenAI

# Initialize Bedrock client
bedrock = boto3.client(service_name="bedrock-runtime")

client = OpenAI(
    # This is the default and can be omitted
    api_key=os.environ.get("OPENAI_API_KEY"),
)

def data_ingestion(file_path):
    loader = PyPDFLoader(file_path)
    documents = loader.load()

    # Using Character splitter for better results with this PDF data set
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    docs = text_splitter.split_documents(documents)
    return docs

def get_vector_store(docs):
    openai_embeddings = OpenAIEmbeddings()
    vectorstore_faiss = FAISS.from_documents(docs, openai_embeddings)
    if not os.path.exists("faiss_index"):
        os.makedirs("faiss_index")
    vectorstore_faiss.save_local("faiss_index")

def get_claude_llm():
    llm = BedrockChat(model_id="anthropic.claude-3-haiku-20240307-v1:0", client=bedrock)
    return llm

def get_titan_text_lite_llm():
    llm = Bedrock(model_id="amazon.titan-text-lite-v1", client=bedrock)
    return llm

def get_ai21_llm():
    return "ai21.jamba-instruct-v1:0"

def get_openai_llm():
    llm = OpenAI(model_name="gpt-4o-mini")
    return llm

def invoke_bedrock_model(prompt, model_id, max_tokens = 250000):
    body = json.dumps({
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "top_p": 0.3,
        "temperature": 0.3,
    })

    response = bedrock.invoke_model(
        body=body,
        modelId=model_id,
        accept="application/json",
        contentType="application/json"
    )
    result = json.loads(response.get('body').read())
    return result['choices'][0]['message']['content']
    

def invoke_openai_model(prompt):
    response = client.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=250000,
        api_key=os.getenv("OPENAI_API_KEY"),
        n=1,
        stop=None,
        temperature=0.3,
        top_p=0.8
    )
    return response.choices[0].message.content.strip()

prompt_template = """
Human: Use the following pieces of context to provide a concise answer to the question at the end. Do not summarize the results. If you don't know the answer, just say that you don't know, don't try to make up an answer.
<context>
{context}
</context>

Question: {question}

Assistant:
"""

PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

def get_response_llm(llm, vectorstore_faiss, query):
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore_faiss.as_retriever(search_type="similarity", search_kwargs={"k": 3}),
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT}
    )
    answer = qa({"query": query})
    return answer['result']

def main():
    st.set_page_config(page_title="Nicole AI", layout="wide")

    st.title("📄 Nicole AI: AI-powered Assistant 💼")

    st.sidebar.title("Select AI Model")
    if "model_choice" not in st.session_state:
        st.session_state.model_choice = "OpenAI GPT-4o-mini"

    model_choice = st.sidebar.radio("Choose a model to generate responses:", ("Anthropic Claude 3 Haiku", "Amazon Titan Text Lite", "AI21 Jamba-Instruct v1", "OpenAI GPT-4o-mini"))

    if model_choice != st.session_state.model_choice:
        st.session_state.model_choice = model_choice
        st.session_state.ai_messages = []
        st.session_state.memory = ConversationBufferMemory()

    tab1, tab2 = st.tabs(["Chat with AI", "Chat with Documents"])

    with tab1:
        st.header("Chat with AI")
        if "ai_messages" not in st.session_state:
            st.session_state.ai_messages = []

        for message in st.session_state.ai_messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        if ai_prompt := st.chat_input("You:"):
            st.session_state.ai_messages.append({"role": "user", "content": ai_prompt})
            with st.chat_message("user"):
                st.markdown(ai_prompt)

            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                full_response = ""

                if model_choice == "Anthropic Claude 3 Haiku":
                    llm = get_claude_llm()
                    model = ConversationChain(llm=llm, verbose=True, memory=st.session_state.memory)
                    result = model.predict(input=ai_prompt)
                elif model_choice == "Amazon Titan Text Lite":
                    llm = get_titan_text_lite_llm()
                    model = ConversationChain(llm=llm, verbose=True, memory=st.session_state.memory)
                    result = model.predict(input=ai_prompt)
                elif model_choice == "OpenAI GPT-4o-mini":
                    llm = get_openai_llm()
                    model = ConversationChain(llm=llm, verbose=True, memory=st.session_state.memory)
                    result = model.predict(input=ai_prompt)
                else:
                    model_id = get_ai21_llm()
                    result = invoke_bedrock_model(ai_prompt, model_id, max_tokens=4096)

                for chunk in result.split(' '):  # Simulate stream of response with milliseconds delay
                    full_response += chunk + ' '
                    time.sleep(0.05)
                    message_placeholder.markdown(full_response + "▌")

                message_placeholder.markdown(full_response)
                st.session_state.ai_messages.append({"role": "assistant", "content": full_response})

    with tab2:
        st.header("Chat with Documents")
        user_question = st.text_input("Enter your question related to the PDF files", "")

        st.sidebar.title("Manage Vector Store")
        uploaded_files = st.sidebar.file_uploader("Upload PDF files", type="pdf", accept_multiple_files=True)
        if st.sidebar.button("Update Vectors"):
            if uploaded_files:
                with st.spinner("Updating vectors..."):
                    all_docs = []
                    for uploaded_file in uploaded_files:
                        file_path = os.path.join("data", uploaded_file.name)
                        with open(file_path, "wb") as f:
                            f.write(uploaded_file.getbuffer())
                        docs = data_ingestion(file_path)
                        all_docs.extend(docs)
                    get_vector_store(all_docs)
                    st.sidebar.success("Vectors updated successfully!")
            else:
                st.sidebar.error("Please upload PDF files before updating vectors.")

        # List and delete uploaded files
        st.sidebar.title("Uploaded Files")
        if not os.path.exists("data"):
            os.makedirs("data")

        files = os.listdir("data")
        for file in files:
            file_path = os.path.join("data", file)
            if st.sidebar.button(f"Delete {file}", key=file):
                os.remove(file_path)
                st.sidebar.success(f"Deleted {file}")

        if st.button("Get Response"):
            if not os.path.exists("faiss_index/index.faiss"):
                st.error("FAISS index file not found. Please update vectors first.")
                return

            with st.spinner(f"Generating response with {model_choice}..."):
                faiss_index = FAISS.load_local("faiss_index", OpenAIEmbeddings(), allow_dangerous_deserialization=True)
                if model_choice == "Anthropic Claude 3 Haiku":
                    llm = get_claude_llm()
                    response = get_response_llm(llm, faiss_index, user_question)
                elif model_choice == "Amazon Titan Text Lite":
                    llm = get_titan_text_lite_llm()
                    response = get_response_llm(llm, faiss_index, user_question)
                elif model_choice == "OpenAI GPT-4o-mini":
                    llm = get_openai_llm()
                    response = get_response_llm(llm, faiss_index, user_question)
                else:
                    model_id = get_ai21_llm()
                    context = ""  # Replace with the context extracted from the documents
                    response = invoke_bedrock_model(f"Use the following context to answer the question: {context}\n\nQuestion: {user_question}", model_id, max_tokens=4096)

                st.markdown(f"### {model_choice}'s Response")
                st.write(response)
                st.success("Response generated successfully!")

if __name__ == "__main__":
    main()
