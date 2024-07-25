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
from langchain_community.embeddings import BedrockEmbeddings
from langchain_community.document_loaders import PyPDFLoader

# Initialize Bedrock client
bedrock = boto3.client(service_name="bedrock-runtime")

bedrock_embeddings = BedrockEmbeddings(
    model_id='amazon.titan-embed-text-v1',  
    client=bedrock
)

def data_ingestion(file_path):
    loader = PyPDFLoader(file_path)
    documents = loader.load()

    # Using Character splitter for better results with this PDF data set
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    docs = text_splitter.split_documents(documents)
    return docs

def get_vector_store(docs):
    vectorstore_faiss = FAISS.from_documents(docs, bedrock_embeddings)
    if not os.path.exists("faiss_index"):
        os.makedirs("faiss_index")
    vectorstore_faiss.save_local("faiss_index")

def get_claude_llm():
    llm = Bedrock(model_id="anthropic.claude-v2", client=bedrock)
    return llm

def get_llama3_llm():
    llm = Bedrock(model_id="meta.llama3-8b-instruct-v1:0", client=bedrock)
    return llm

def get_ai21_llm():
    return "ai21.jamba-instruct-v1:0"

def invoke_bedrock_model(prompt, model_id):
    body = json.dumps({
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 256,
        "top_p": 0.8,
        "temperature": 0.7,
    })

    response = bedrock.invoke_model(
        body=body,
        modelId=model_id,
        accept="application/json",
        contentType="application/json"
    )
    result = json.loads(response.get('body').read())
    return result['choices'][0]['message']['content']

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
    st.set_page_config(page_title="DocQuery AI", layout="wide")

    st.title("📄 DocQuery AI: AI-powered PDF Document Assistant 💼")

    st.sidebar.title("Select AI Model")
    model_choice = st.sidebar.radio("Choose a model to generate responses:", ("Claude", "LLaMA3", "AI21"))

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

                if model_choice == "Claude":
                    llm = get_claude_llm()
                    model = ConversationChain(llm=llm, verbose=True, memory=ConversationBufferMemory())
                    result = model.predict(input=ai_prompt)
                elif model_choice == "LLaMA3":
                    llm = get_llama3_llm()
                    model = ConversationChain(llm=llm, verbose=True, memory=ConversationBufferMemory())
                    result = model.predict(input=ai_prompt)
                else:
                    model_id = get_ai21_llm()
                    result = invoke_bedrock_model(ai_prompt, model_id)

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

        if st.button("Get Response"):
            if not os.path.exists("faiss_index/index.faiss"):
                st.error("FAISS index file not found. Please update vectors first.")
                return

            with st.spinner(f"Generating response with {model_choice}..."):
                faiss_index = FAISS.load_local("faiss_index", bedrock_embeddings, allow_dangerous_deserialization=True)
                if model_choice == "Claude":
                    llm = get_claude_llm()
                    response = get_response_llm(llm, faiss_index, user_question)
                elif model_choice == "LLaMA3":
                    llm = get_llama3_llm()
                    response = get_response_llm(llm, faiss_index, user_question)
                else:
                    model_id = get_ai21_llm()
                    context = ""  # Replace with the context extracted from the documents
                    response = invoke_bedrock_model(f"Use the following context to answer the question: {context}\n\nQuestion: {user_question}", model_id)

                st.markdown(f"### {model_choice}'s Response")
                st.write(response)
                st.success("Response generated successfully!")

if __name__ == "__main__":
    main()