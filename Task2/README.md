# 🧠 Natural Language Querying in Langchain Database:

This project is an Database Agent. Follow the instructions below to set up and run the project.  
You will need access to the **ChatGroq API**.


## ⚙️ Setup and Run Instructions

Make sure you have **Anaconda** or **Miniconda** and **Docker** installed before proceeding.

### 1. Create and Activate Conda Environment

conda create -n agentic_ai_chatbot python=3.12
conda activate agentic_ai_chatbot

## 2. Install Python Dependencies
pip install -r requirements.txt

## 3. Running Database
sudo docker compose up -d

## 4. create and insert data using psql
sudo apt update
sudo apt install postgresql-client -y
PGPASSWORD=postgres psql -h localhost -U admin -d agent_db

-- Create the table
CREATE TABLE customers (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100),
    email VARCHAR(100),
    age INTEGER
);
-- Populate Data
INSERT INTO customers (name, email, age) VALUES
('Alice Johnson', 'alice@example.com', 30),
('Bob Smith', 'bob@example.com', 25),
('Charlie Davis', 'charlie@example.com', 35),
('Diana Prince', 'diana@example.com', 28),
('Ethan Hunt', 'ethan@example.com', 40),
('Fiona Glenanne', 'fiona@example.com', 33),
('George Miller', 'george@example.com', 38),
('Hannah Wells', 'hannah@example.com', 27),
('Ian Malcolm', 'ian@example.com', 45),
('Jane Eyre', 'jane@example.com', 32);

-- Verify
SELECT * FROM customers;

--integrate pgvector
CREATE EXTENSION IF NOT EXISTS vector;
ALTER TABLE customers ADD COLUMN IF NOT EXISTS embedding vector(384); 
python create_embedding_pgvec.py

## 6. Run the Chatbot with Fast API
uvicorn database_Agent:app --reload
post request with body 
http://127.0.0.1:8000/query
{
  "question": "How many customers are there?"
}

<!-- python database_Agent.py -->
