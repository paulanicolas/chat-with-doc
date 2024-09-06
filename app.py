import os
import time
import json
import boto3
import streamlit as st
import numpy as np
import pickle
import faiss
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
from openai import OpenAI
from anthropic_bedrock import AnthropicBedrock

# Initialize Bedrock client
bedrock_us_east_1 = boto3.client(service_name="bedrock-runtime", region_name="us-east-1")
bedrock_us_west_2 = boto3.client(service_name="bedrock-runtime", region_name="us-west-2")

# Initialize OpenAI client
client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),
)

# Token cost per model (cost per 1000 tokens)
token_costs = {
    "anthropic.claude-3-5-sonnet-20240620-v1:0": {"input": 0.003 / 1000, "output": 0.015 / 1000},
    "anthropic.claude-3-haiku-20240307-v1:0": {"input": 0.00025 / 1000, "output": 0.00125 / 1000}
}

def count_tokens_claude(prompt):
    """Count the number of tokens in a Claude prompt (using a simple word count as a proxy)."""
    #return len(prompt.split())
    return len(prompt) // 4

def calculate_cost(model_id, input_tokens, output_tokens):
    """Calculate total cost based on token usage for Claude models."""
    input_cost = token_costs[model_id]["input"] * input_tokens
    output_cost = token_costs[model_id]["output"] * output_tokens
    total_cost = input_cost + output_cost
    return total_cost

def calculate_cost_claude(model_id, input_prompt, output):
    input_tokens = count_tokens_claude(input_prompt)
    output_tokens = count_tokens_claude(output)

    # Calculate cost based on token usage
    total_cost = calculate_cost(model_id, input_tokens, output_tokens)

    return total_cost, input_tokens, output_tokens

def calculate_cost_claude_new(model_id, input_prompt, output):
    client = AnthropicBedrock()
    input_tokens = client.count_tokens(input_prompt)
    output_tokens = client.count_tokens(output)
    
    # Calculate cost based on token usage
    total_cost = calculate_cost(model_id, input_tokens, output_tokens)

    return total_cost, round(input_tokens), round(output_tokens)

def data_ingestion(file_path):
    """Load and split PDF documents into chunks."""
    loader = PyPDFLoader(file_path)
    documents = loader.load()

    # Use character splitter for better results with this PDF dataset
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    docs = text_splitter.split_documents(documents)
    return docs

def get_vector_store(docs):
    """Generate and save a FAISS vector store from documents."""
    openai_embeddings = OpenAIEmbeddings()
    vectorstore_faiss = FAISS.from_documents(docs, openai_embeddings)
    if not os.path.exists("faiss_index"):
        os.makedirs("faiss_index")
    vectorstore_faiss.save_local("faiss_index")

 # Save the documents list for later retrieval
    with open("faiss_index/documents.pkl", "wb") as f:
        pickle.dump(docs, f)

def generate_embedding(text):
    openai_embeddings = OpenAIEmbeddings()
    embedding = openai_embeddings.embed_query(text)
    return np.array(embedding).astype('float32')

def get_content_of_vectore_store(query):
    # Load the FAISS index
    index = faiss.read_index("faiss_index/index.faiss")

    # Load the document metadata
    with open("faiss_index/documents.pkl", "rb") as f:
        documents = pickle.load(f)

    # Define a query and generate its embedding
    query_text = query
    query_embedding = generate_embedding(query_text).reshape(1, -1)

    # Search the FAISS index
    k = 3  # Number of nearest neighbors to retrieve
    distances, indices = index.search(query_embedding, k)

    # Extract the documents using the indices
    retrieved_documents = [documents[idx] for idx in indices[0]]

    # Prepare the documents for the A121 Jamba Instruct model
    context = "Context for A121 Jamba Instruct Model:\n"
    for doc in retrieved_documents:
        context += f"\nContent: {doc.page_content}\n\n"

    return doc.page_content


def get_claude_llm(model_id):
    """Initialize Claude LLM."""
    llm = BedrockChat(model_id=model_id, client=bedrock_us_east_1)
    return llm

def get_ai21_llm():
    """Return AI21 model ID."""
    return "ai21.jamba-instruct-v1:0"

def get_meta_llama_llm():
    """Return Meta LLaMA model ID."""
    llm = Bedrock(model_id="meta.llama3-1-405b-instruct-v1:0", client=bedrock_us_west_2)
    return llm

def get_openai_llm():
    """Return OpenAI model ID."""
    return "gpt-4o-mini"

def invoke_bedrock_model(prompt, model_id, max_tokens=250000):
    """Invoke Bedrock model with given prompt."""
    if model_id == 'meta.llama3-1-405b-instruct-v1:0':
        client = bedrock_us_west_2
    else:
        client = bedrock_us_east_1

    body = json.dumps({
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "top_p": 0.8,
        "temperature": 0.3,
    })

    response = client.invoke_model(
        body=body,
        modelId=model_id,
        accept="application/json",
        contentType="application/json"
    )
    result = json.loads(response.get('body').read())
    return result['choices'][0]['message']['content']

def invoke_openai_model(prompt, memory):
    """Invoke OpenAI model with given prompt."""
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        #messages=[{"role": "user", "content": prompt}],
        messages=memory + [{"role": "user", "content": prompt}],
        max_tokens=4096,
        n=1,
        stop=None,
        temperature=0.3,
        top_p=0.8
    )
    return response.choices[0].message.content.strip()

prompt_template_llama = """
<|begin_of_text|><|start_header_id|>system<|end_header_id|>
The following is a conversation between a human and an AI assistant.
The assistant is designed to provide concise, accurate, and precise answers related to Environmental, Social, and Governance (ESG) topics. The assistant will not include any additional commentary, chit-chat, or unnecessary information. It will answer the questions directly and clearly. Below are examples of the type of answers expected:

<example>
Example Questions and Answers:

Q: What is the reporting period in which the climate-related scenario analysis was carried out?  
A: January 1, 2022 to December 31, 2022

Q: What are the Scope 1 greenhouse gas emissions for the organization?  
A: 100,000 metric tons of CO2e

Q: Does the entity apply a carbon price in decision-making, and if so, how?  
A: Yes, the entity applies a carbon price of $50 per metric ton of CO2e in its investment decisions and scenario analysis.

Q: Is the greenhouse gas emissions target a gross or net target, and if net, what is the associated gross target?  
A: The target is a net greenhouse gas emissions target of 50,000 metric tons of CO2e, with an associated gross target of 75,000 metric tons of CO2e.
</example>

Adhere strictly to this format.

You will play the role of the assistant.

You have access to the following pieces of context:
<context>
{context}
</context>

Use this context to provide a concise answer to the question at the end. Do not summarize the results. If you don't know the answer, just say that you don't know, don't try to make up an answer.

<|eot_id|>
<|start_header_id|>user<|end_header_id|>
Question: {question}
<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
Thought: I will review the provided context to answer the question.
Final Answer: 
"""

prompt_template_old = """
Human: You are an AI designed to provide concise, accurate, and precise answers related to Environmental, Social, and Governance (ESG) topics. Do not include any additional commentary, chit-chat, or unnecessary information. Answer the questions directly and clearly. The answer should always be in paragraph only, do not make it in bullet points. Below are examples of the type of answers I expect:
Example Questions and Answers:

<example>
Q: What is the reporting period in which the climate-related scenario analysis was carried out? A: January 1, 2022 to December 31, 2022
Q: What are the Scope 1 greenhouse gas emissions for the organization? A: 100,000 metric tons of CO2e
Q: Does the entity apply a carbon price in decision-making, and if so, how? A: Yes, the entity applies a carbon price of $50 per metric ton of CO2e in its investment decisions and scenario analysis.
Q: Is the greenhouse gas emissions target a gross or net target, and if net, what is the associated gross target? A: The target is a net greenhouse gas emissions target of 50,000 metric tons of CO2e, with an associated gross target of 75,000 metric tons of CO2e.
</example>
Adhere strictly to this format. 

Use the following pieces of context to provide a concise answer to the question at the end. Do not summarize the results. If you don't know the answer, just say that you don't know, don't try to make up an answer.
<context>
{context}
</context>

Question: {question}

Assistant:
"""

prompt_template = """
I'm going to give you a IFRS Sustainability Disclosures report. Read the report carefully, because I'm going to ask you a question about it. Here is the report.
<report>{context}</report>

Please read the report carefully and analyze all aspects.

Then, I will ask you some questions about this report. Please answer based on your analysis of the report. Your answers should meet the following requirements:

- Answers should be based on the content of the report itself, avoiding excessive embellishment or speculation.
- Provide direct answer from the report. Do not include any additional commentary, chit-chat, or unnecessary information.
- If a question goes beyond the scope discussed in the report, please clearly state that it cannot be determined from the report's content.

Question: {question}

Assistant:
"""

PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

def get_response_llm(llm, vectorstore_faiss, query, model_choice = ''):
    if model_choice == 'Llama 3.1 405B Instruct':
        PROMPT = PromptTemplate(template=prompt_template_llama, input_variables=["context", "question"])
    else:
        PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

    """Generate response using LLM and vector store."""
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore_faiss.as_retriever(search_type="similarity", search_kwargs={"k": 3}),
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT}
    )
    answer = qa({"query": query})
    return answer['result']

def get_response_openai(vectorstore_faiss, query, memory):
    retriever = vectorstore_faiss.as_retriever(search_type="similarity", search_kwargs={"k": 3})
    docs = retriever.get_relevant_documents(query)
    context = " ".join([doc.page_content for doc in docs])
    prompt = PROMPT.format(context=context, question=query)
    response = invoke_openai_model(prompt, memory)
    return response

def main():
    st.set_page_config(page_title="Nicole AI", layout="wide")

    st.title("📄 Nicole AI: AI-powered Assistant 💼")

    st.sidebar.title("Select AI Model")
    if "model_choice" not in st.session_state:
        st.session_state.model_choice = "OpenAI GPT-4o-mini"

    model_choice = st.sidebar.radio("Choose a model to generate responses:", ("Anthropic Claude 3.5 Sonnet", "Anthropic Claude 3 Haiku", "Llama 3.1 405B Instruct", "AI21 Jamba-Instruct v1", "OpenAI GPT-4o-mini"))

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

                if model_choice == "Anthropic Claude 3.5 Sonnet":
                    llm = get_claude_llm('anthropic.claude-3-5-sonnet-20240620-v1:0')
                    model = ConversationChain(llm=llm, verbose=True, memory=st.session_state.memory)
                    result = model.predict(input=ai_prompt)
                    cost, number_of_input_tokens, number_of_output_tokens = calculate_cost_claude('anthropic.claude-3-5-sonnet-20240620-v1:0', ai_prompt, result)
                elif model_choice == "Anthropic Claude 3 Haiku":
                    llm = get_claude_llm('anthropic.claude-3-haiku-20240307-v1:0')
                    model = ConversationChain(llm=llm, verbose=True, memory=st.session_state.memory)
                    result = model.predict(input=ai_prompt)
                    cost, number_of_input_tokens, number_of_output_tokens = calculate_cost_claude('anthropic.claude-3-haiku-20240307-v1:0', ai_prompt, result)
                elif model_choice == "Llama 3.1 405B Instruct":
                    llm = get_meta_llama_llm()
                    model = ConversationChain(llm=llm, verbose=True, memory=st.session_state.memory)
                    result = model.predict(input=ai_prompt)
                elif model_choice == "OpenAI GPT-4o-mini":
                    memory = [{"role": msg["role"], "content": msg["content"]} for msg in st.session_state.ai_messages]
                    result = invoke_openai_model(ai_prompt, memory)
                else:
                    model_id = get_ai21_llm()
                    result = invoke_bedrock_model(ai_prompt, model_id, max_tokens=4096)

                if model_choice in ["Anthropic Claude 3.5 Sonnet", "Anthropic Claude 3 Haiku"]:
                    st.write(f"Number of input tokens: {number_of_input_tokens}")
                    st.write(f"Number of output tokens: {number_of_output_tokens}")
                    st.write(f"### Total cost: ${cost:.6f}")

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
                if model_choice == "Anthropic Claude 3.5 Sonnet":
                    llm = get_claude_llm('anthropic.claude-3-5-sonnet-20240620-v1:0')
                    response = get_response_llm(llm, faiss_index, user_question)
                    cost, number_of_input_tokens, number_of_output_tokens = calculate_cost_claude('anthropic.claude-3-5-sonnet-20240620-v1:0', user_question, response)
                elif model_choice == "Anthropic Claude 3 Haiku":
                    llm = get_claude_llm('anthropic.claude-3-haiku-20240307-v1:0')
                    response = get_response_llm(llm, faiss_index, user_question)
                    cost, number_of_input_tokens, number_of_output_tokens = calculate_cost_claude('anthropic.claude-3-haiku-20240307-v1:0', user_question, response)
                elif model_choice == "Llama 3.1 405B Instruct":
                    llm = get_meta_llama_llm()
                    response = get_response_llm(llm, faiss_index, user_question, model_choice)
                elif model_choice == "OpenAI GPT-4o-mini":
                    #response = get_response_openai(faiss_index, user_question)
                    memory = [{"role": msg["role"], "content": msg["content"]} for msg in st.session_state.ai_messages]
                    response = get_response_openai(faiss_index, user_question, memory)
                else:   
                    model_id = get_ai21_llm()
                    context = get_content_of_vectore_store(user_question)  # Replace with the context extracted from the documents
                    response = invoke_bedrock_model(f""" You are an AI designed to provide concise, accurate, and precise answers related to Environmental, Social, and Governance (ESG) topics. Do not include any additional commentary, chit-chat, or unnecessary information. Answer the questions directly and clearly. Below are examples of the type of answers I expect:
Example Questions and Answers:
 
Q: What is the reporting period in which the climate-related scenario analysis was carried out? A: January 1, 2022 to December 31, 2022
Q: What are the Scope 1 greenhouse gas emissions for the organization? A: 100,000 metric tons of CO2e
Q: Does the entity apply a carbon price in decision-making, and if so, how? A: Yes, the entity applies a carbon price of $50 per metric ton of CO2e in its investment decisions and scenario analysis.
Q: Is the greenhouse gas emissions target a gross or net target, and if net, what is the associated gross target? A: The target is a net greenhouse gas emissions target of 50,000 metric tons of CO2e, with an associated gross target of 75,000 metric tons of CO2e.
Adhere strictly to this format. Use the following context to answer the question: {context}\n\nQuestion: {user_question}""", model_id, max_tokens=4096)


                if model_choice in ["Anthropic Claude 3.5 Sonnet", "Anthropic Claude 3 Haiku"]:
                    st.write(f"Number of input tokens: {number_of_input_tokens}")
                    st.write(f"Number of output tokens: {number_of_output_tokens}")
                    st.write(f"### Total cost: ${cost:.6f}")

                st.markdown(f"### {model_choice}'s Response")
                st.write(response)
                st.success("Response generated successfully!")

if __name__ == "__main__":
    main()
