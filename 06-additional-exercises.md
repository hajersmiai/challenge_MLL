# 📚 LLM & GenAI One-Day Project Guidelines

This document contains **four independent exercises**.  
Each exercise is designed to be completed in **1 or 2 days** and uses LLMs or GenAI concepts.  

## 1️⃣ Bias Detection in AI Output

**Objective:**  
Investigate potential biases in an LLM’s responses and propose strategies to reduce them.

**Constraints:**  
- Must use at least two different prompt formulations to test bias.  
- Must test on at least two different LLM models.  
- Report must include both quantitative (counts, frequencies) and qualitative observations.

**Suggested Datasets & Domains:**  
- Occupation stereotypes  
- Political statements  
- Gender pronouns in profession descriptions  
- News headline rewording

**Steps:**
1. Choose a bias domain (e.g., gendered language in tech jobs).  
2. Write **5–10 test prompts** designed to reveal bias (e.g., "Describe a nurse" vs. "Describe a software engineer").  
3. Run prompts on multiple LLMs (OpenAI GPT-4, local LLaMA, etc.).  
4. Compare responses for differences in tone, descriptors, or assumptions.  
5. Document patterns and propose at least one bias mitigation approach (e.g., neutral prompt rewriting).

**Resources:**
- [Google AI Fairness Guidelines](https://ai.google/responsibilities/responsible-ai-practices/)
- [Datacamp bias lesson](https://campus.datacamp.com/courses/generative-ai-for-business/generative-ai-solutions?ex=5)
- [GeeksForGeeks AI bias and fairness](https://www.geeksforgeeks.org/artificial-intelligence/fairness-and-bias-in-artificial-intelligence/)
- [Hugging Face: Datasets for Bias Testing](https://huggingface.co/datasets?search=bias)

## 2️⃣ Code Explainer Bot (Error-Focused)

**Objective:**  
Build a bot that explains **errors** in Python code without giving full solutions or generating new code. See this exercice as an opportunity to design a proof of concept of a "Becode code assistant" that would help coaches explain code errors without giving the solution to the learner.

**Constraints:**  
- The bot **must not** generate working code.  
- Must clearly identify **why** the error occurred.  
- Must suggest conceptual fixes (e.g., "You need to initialize the variable before using it").

**Steps:**
1. Prepare a set of **faulty Python scripts** (syntax errors, logic errors, runtime errors).  
2. Write a prompt template for the LLM:  

```plaintext
You are a teaching assistant. Given a Python error and code snippet, explain:

    What the error means
    Why it happened
    How the student can fix it (conceptually, not by giving working code)
    Do not provide any full code solutions.
```


3. Connect a Python script that:
- Reads code files with errors
- Sends them to the LLM
- Outputs explanations to a markdown file
4. Test with at least 5 different error cases.

**Resources:**
- [Python Common Errors](https://realpython.com/python-exceptions/)

## 3️⃣ Natural Language to SQL Queries

**Objective:**  
Create a system that converts plain English questions into SQL queries and retrieves results from a given database.

**Constraints:**  
- Must work on a **predefined SQLite database**.  
- Must return both the **SQL query** and the **query results**.  
- No direct hardcoding of queries allowed.

**Dataset Ideas:**
- [Chinook SQLite Database](https://github.com/lerocha/chinook-database) (music store data)  
- [Sakila Sample Database](https://dev.mysql.com/doc/sakila/en/) (movies rental) 
- Or any database you already have or already know in order to save some time 

**Steps:**
1. Load the database into SQLite and explore the schema.  
2. Create a prompt template for the LLM:

```plaintext
You are a SQL assistant. Given a database schema and a natural language request,
write the correct SQL query to answer it. Only use the given schema.
```

3. Parse the LLM’s SQL output and execute it in Python.  
4. Return the SQL query + table of results to the user.  
5. Test with at least 5 different natural language queries.

**Resources:**
- [SQLite Python Docs](https://docs.python.org/3/library/sqlite3.html)
- [LangChain SQL Database Agent](https://python.langchain.com/docs/integrations/tools/sql_database/)

---

## 4️⃣ Mini RAG (Retrieval-Augmented Generation) System

**Objective:**  
Build a small chatbot that answers questions based only on a provided document set.

**Constraints:**  
- Must not allow the LLM to answer from general knowledge (only from provided docs).  
- Must use an embedding-based retrieval step before generation.  
- Dataset should have at least 10+ documents.

**Dataset Ideas:**
- Lecture notes (PDF → text)
- Wikipedia articles on a single topic
- Product manuals

**Steps:**
1. Prepare your document set (convert to `.txt` or `.md`).  
2. Use an embedding model (e.g., OpenAI `text-embedding-ada-002` or sentence-transformers) to vectorize documents.  
3. Store embeddings in a vector database (e.g., FAISS, ChromaDB).  
4. On a query:
- Convert question to embedding
- Retrieve top relevant docs
- Pass retrieved text to LLM with instruction:  
  ```
  Answer the question using only the provided context. 
  If the answer is not in the context, say "I don't know."
  ```
5. Test with 5+ queries.

**Resources:**
- [FAISS Documentation](https://faiss.ai)
- [LangChain RAG Tutorial](https://python.langchain.com/docs/tutorials/rag/)
- [Sentence Transformers](https://www.sbert.net/)


## Deliverables for All Projects
- **Notebook or Python script** with working code  
- **README** explaining setup, usage, and findings  
- **Short report** summarizing results and challenges

