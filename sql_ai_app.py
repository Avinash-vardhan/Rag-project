import streamlit as st
import mysql.connector
import ollama

st.title("AI SQL Assistant")

# Connect MySQL
conn = mysql.connector.connect(
    host="localhost",
    user="root",
    password="root123",
    database="company_db"
)

cursor = conn.cursor()

question = st.text_input("Ask a question about the database")

if st.button("Ask"):

    prompt = f"""
You are a MySQL expert.

Return ONLY SQL query.

Database schema:
employees(id, name, department, salary)

Question: {question}

SQL:
"""

    response = ollama.chat(
        model="llama3",
        messages=[{"role": "user", "content": prompt}]
    )

    sql = response["message"]["content"]

    sql = sql.replace("```sql", "").replace("```", "").strip()
    sql = sql.split(";")[0] + ";"

    st.write("Generated SQL:")
    st.code(sql)

    cursor.execute(sql)
    results = cursor.fetchall()

    st.write("Results:")
    st.write(results)