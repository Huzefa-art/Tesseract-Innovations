import utils.transport
import utils.session
from pathlib import Path
import pickle
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langgraph.prebuilt import create_react_agent
from langchain_core.runnables import RunnableLambda
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.chat_models import ChatOpenAI 
import getpass
import os
from langchain_groq import ChatGroq
from langchain_core.tools import tool
# from langchain.schema import AIMessage
from langchain_core.messages import HumanMessage, AIMessage
if "GROQ_API_KEY" not in os.environ:
    os.environ["GROQ_API_KEY"] = getpass.getpass("Enter your Groq API key: ")

def load_doc_and_embeddings():
    model_name = "BAAI/bge-small-en"
    model_kwargs = {"device": "cpu"}
    encode_kwargs = {"normalize_embeddings": True}

    VECTOR_STORE_FILE = "vector_store.pkl"

    if Path(VECTOR_STORE_FILE).exists():
        with open(VECTOR_STORE_FILE, "rb") as f:
            vector_store = pickle.load(f)
        print("Vector store loaded from disk.")
    else:

        embeddings = HuggingFaceEmbeddings(
        model_name=model_name, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs
    )

        loader = TextLoader("tesseract_data.txt")
        docs = loader.load()
        print(f"Loaded {len(docs)} raw documents.")

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        all_splits = text_splitter.split_documents(docs)
        print(f"Split into {len(all_splits)} chunks.")

        vector_store = InMemoryVectorStore(embedding=embeddings)
        vector_store.add_documents(all_splits)

        with open(VECTOR_STORE_FILE, "wb") as f:
            pickle.dump(vector_store, f)
        print("Vector store saved to disk.")
    return vector_store

vector_store= load_doc_and_embeddings()

def retrieve(state: dict):
    question = state["question"]
    docs = vector_store.similarity_search(question)
    return {"context": docs}

def generate(state: dict):
    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0.5,
        max_tokens=None,
        # reasoning_format="parsed",
        timeout=None,
        max_retries=2,
    )

    docs_content = "\n\n".join(doc.page_content for doc in state["context"])
    # print("retrieved",docs_content)
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant for answering company questions given the context. IF you don't know the answer just reply I am sorry I am not aware about that"),
        ("human", "Question: {question}\n\nContext:\n{context}")
    ])
    messages = prompt.invoke({"question": state["question"], "context": docs_content})
    response = llm.invoke(messages)
    return {"answer": response.content}


@tool
def company_QA(question: str) -> str:
    """
    Answers questions related to the company Tesseract based on internal knowledge and documentation.

    This tool should be used when the user asks static or dynamic questions about the company, such as:
    - Company technologies and core offerings
    - Number of employees or organizational structure
    - Leadership (e.g., CEO)
    - Vision, mission, or goals
    - Products, services, or internal tools
    - Other relevant details about the company

    Input:
        question (str): The user's question about the company.

    Output:
        A string containing the best-matched answer from Tesseract's internal documentation,
        or a fallback message if the answer cannot be confidently generated.
    """
    try:
        state = {"question": question}
        retrieved = retrieve(state)
        state.update(retrieved)
        result = generate(state)
        return result["answer"]
    except Exception as e:
        return f"Sorry, I couldn't find the answer due to an internal issue. ({str(e)})"


@tool
def Project_Proposal_Generator(question: str) -> str:
    """
    This tool is if the question includes solution related to AI, like: "I need an AI solution to detect fraud in
    banking,"
    Generates a templated AI project proposal based on the user's input.

    Returns a structured output with:
    - Problem Understanding
    - Proposed AI Solution
    - Estimated Cost
    - Required Timeline
    """
    # print(f"[Tool Triggered] Generating proposal for: {question}")

    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0.5,
        max_tokens=None,
        timeout=None,
        max_retries=2,
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system",
        "You are an expert AI consultant. Based on the user's request, fill out the following template with appropriate content:\n\n"
        "**Problem Understanding**:\n<Write the problem here>\n\n"
        "**Proposed AI Solution**:\n<Write proposed solution here, mention AI techniques>\n\n"
        "**Estimated Cost (USD)**:\n<Give a reasonable cost range>\n\n"
        "**Required Timeline (weeks)**:\n<Give a realistic timeline>\n\n"
        "Be concise and avoid omitting any section."
        ),
        ("human", "{input_problem}")
    ])
    # Create the message
    messages = prompt.invoke({"input_problem": question})
    response = llm.invoke(messages)

    return response.content


@tool
def HR_Forwarding(question: str) -> str:
    """
    Handles user queries related to hiring, job seeking, or HR-related communication.

    This tool should be triggered when the user expresses interest in employment opportunities
    or HR-related discussions, such as:
    - "I'm looking for a job."
    - "Can I talk to HR?"
    - "Here's my resume..."
    - "I want to apply for a position."

    Behavior:
        - The agent must ask the user for confirmation:
          "Do you want me to forward this to our HR department?"
        - Based on the user's response:
            - If 'yes', reply with: "Message has been sent to HR."
            - If 'no', reply with: "Message was not sent to HR."

    Input:
        question (str): The user's HR-related query or intent.

    Output:
        A confirmation prompt, followed by a simulated acknowledgement message based on user input.
    """


    user_input = question.strip().lower()

    if user_input == "yes":
        return "Message has been sent to HR."
    elif user_input == "no":
        return "Message was not sent to HR."
    else:
        return "Do you want me to forward this to our HR department? (Yes or No)"

if __name__ == "__main__":


    model = ChatGroq(
        model="meta-llama/llama-4-maverick-17b-128e-instruct",
        temperature=0.5,
        max_tokens=None,
        # reasoning_format="parsed",
        timeout=None,
        max_retries=0,
    )
    tools = [company_QA, Project_Proposal_Generator, HR_Forwarding]
    agent_executor = create_react_agent(model, tools)



    print("Welcome to the QA Agent! Type 'exit' to quit.")
    chat_history = []

    while True:
        user_input = input("You: ")
        if user_input.lower() in ("exit", "quit"):
            print("Goodbye!")
            break

        chat_history.append(HumanMessage(content=user_input))

        response = agent_executor.invoke({"messages": chat_history})


        ai_messages = [
            m for m in response["messages"]
            if isinstance(m, AIMessage) and hasattr(m, "content")
        ]

        if ai_messages:
            print("AI:", ai_messages[-1].content)
            chat_history.append(ai_messages[-1])