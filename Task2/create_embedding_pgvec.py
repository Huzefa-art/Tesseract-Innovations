import psycopg2
from psycopg2.extras import execute_values
from langchain_huggingface import HuggingFaceEmbeddings

conn = psycopg2.connect(
    dbname="agent_db",
    user="admin",
    password="postgres",
    host="localhost",
    port="5432"
)
cursor = conn.cursor()

model_name = "BAAI/bge-small-en"
model_kwargs = {"device": "cpu"}
encode_kwargs = {"normalize_embeddings": True}

embeddings = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs,
)

cursor.execute("SELECT id, name, email FROM customers")
rows = cursor.fetchall()

update_values = []
for row in rows:
    cust_id, name, email = row
    text = f"{name} {email}"
    embedding_vector = embeddings.embed_query(text) 
    
    update_values.append((embedding_vector, cust_id))

sql = """
    UPDATE customers
    SET embedding = data.embedding
    FROM (VALUES %s) AS data(embedding, id)
    WHERE customers.id = data.id;
"""

execute_values(cursor, sql, update_values, template="(%s::vector, %s)")

conn.commit()
cursor.close()
conn.close()
print("Embeddings generated and saved in the database.")
