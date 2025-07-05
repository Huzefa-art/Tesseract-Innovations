from langchain_community.utilities import SQLDatabase
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from typing_extensions import Annotated
from langchain_groq import ChatGroq
import os 
import getpass
from typing_extensions import TypedDict
from langchain_community.tools.sql_database.tool import QuerySQLDatabaseTool

from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
from fastapi.responses import JSONResponse
from langchain_huggingface import HuggingFaceEmbeddings

# embedding model for pgvector
embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True}
)


# Environment Setup
# Set GROQ API key
if "GROQ_API_KEY" not in os.environ:
    os.environ["GROQ_API_KEY"] = getpass.getpass("Enter your Groq API key: ")

# Database Configuration
username = "admin"
password = "postgres"
host = "localhost"
port = "5432"
database_name = "agent_db"
db_uri = f"postgresql+psycopg2://{username}:{password}@{host}:{port}/{database_name}"

# Initialize SQL database
db = SQLDatabase.from_uri(db_uri)

# print(db.dialect)
# print(db.get_usable_table_names())
# print(db.run("SELECT * FROM customers LIMIT 10;"))

# LLM Initialization
llm = ChatGroq(
    model="meta-llama/llama-4-maverick-17b-128e-instruct",
    temperature=0.5,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)

# Prompt Template Setup
system_message = """
Given an input question, create a syntactically correct {dialect} query to
run to help find the answer. Unless the user specifies in his question a
specific number of examples they wish to obtain, always limit your query to
at most {top_k} results. You can order the results by a relevant column to
return the most interesting examples in the database.

Never query for all the columns from a specific table, only ask for a the
few relevant columns given the question.

Pay attention to use only the column names that you can see in the schema
description. Be careful to not query for columns that do not exist. Also,
pay attention to which column is in which table.

Only use the following tables:
{table_info}
"""

user_prompt = "Question: {input}"

query_prompt_template = ChatPromptTemplate(
    [("system", system_message), ("user", user_prompt)]
)

class QuestionRequest(BaseModel):
    question: str

class QueryResponse(BaseModel):
    query: str
    result: str
    answer: str


class State(TypedDict):
    question: str
    query: str
    result: str
    answer: str

class QueryOutput(TypedDict):
    """Generated SQL query."""

    query: Annotated[str, ..., "Syntactically valid SQL query."]


# def write_query(state: State):
#     """Generate SQL query to fetch information."""
#     try:
#         prompt = query_prompt_template.invoke(
#             {
#                 "dialect": db.dialect,
#                 "top_k": 10,
#                 "table_info": db.get_table_info(),
#                 "input": state["question"],
#             }
#         )
#         structured_llm = llm.with_structured_output(QueryOutput)
#         result = structured_llm.invoke(prompt)
#         return {"query": result["query"]}
#     except Exception as e:
#         # LLM responded with chatty message, not SQL
#         if "tool_use_failed" in str(e) or "Failed to call a function" in str(e):
#             raise ValueError("Sorry, your input doesn't look like a valid database-related question.")
#         raise RuntimeError(f"Error generating SQL query: {str(e)}")

def write_query(state: State):
    question = state["question"].lower()
    if "similar" in question or "related" in question:
        # Generate embedding of the question
        query_embedding = embeddings.embed_query(state["question"])
        # Format embedding for SQL (Postgres vector array style)
        embedding_str = "[" + ",".join(map(str, query_embedding)) + "]"
        
        query = f"""
        SELECT id, name, email, embedding
        FROM customers
        ORDER BY embedding <=> '{embedding_str}'
        LIMIT 10
        """
        return {"query": query}

    # Else do regular SQL generation via your existing prompt to the LLM
    try:
        prompt = query_prompt_template.invoke(
            {
                "dialect": db.dialect,
                "top_k": 10,
                "table_info": db.get_table_info(),
                "input": state["question"],
            }
        )
        structured_llm = llm.with_structured_output(QueryOutput)
        result = structured_llm.invoke(prompt)
        return {"query": result["query"]}
    except Exception as e:
        if "tool_use_failed" in str(e) or "Failed to call a function" in str(e):
            raise ValueError("Sorry, your input doesn't look like a valid database-related question.")
        raise RuntimeError(f"Error generating SQL query: {str(e)}")

def execute_query(state: State):
    try:
        execute_query_tool = QuerySQLDatabaseTool(db=db)
        return {"result": execute_query_tool.invoke(state["query"])}
    except Exception as e:
        raise RuntimeError(f"Error executing SQL query: {str(e)}")


def generate_answer(state: State):
    try:
        prompt = (
            "Given the following user question, SQL query, and SQL result, "
            "give a short, direct, user-friendly answer with no explanation.\n\n"
            f'Question: {state["question"]}\n'
            f'SQL Query: {state["query"]}\n'
            f'SQL Result: {state["result"]}\n\n'
            "Answer:"
        )
        response = llm.invoke(prompt)
        return {"answer": response.content}
    except Exception as e:
        raise RuntimeError(f"Error generating answer: {str(e)}")


app = FastAPI(title="Natural Language to SQL API")

@app.post("/query", response_model=QueryResponse)
def process_question(request: QuestionRequest):
    question = request.question
    state = {"question": question, "query": "", "result": "", "answer": ""}

    try:
        state.update(write_query(state))
        state.update(execute_query(state))
        state.update(generate_answer(state))

        return QueryResponse(
            query=state["query"],
            result=state["result"],
            answer=state["answer"],
        )
    except ValueError as ve:
        return JSONResponse(
            status_code=400,
            content={"query": "", "result": "", "answer": str(ve)}
        )
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"query": "", "result": "", "answer": "An unexpected error occurred."}
        )


if __name__ == '__main__':
    question = "How many customers are there?"
    
    state = {"question": question, "query": "", "result": "", "answer": ""}
    state.update(write_query(state))
    state.update(execute_query(state))
    state.update(generate_answer(state))
    
    print("Final Answer:", state["answer"])
