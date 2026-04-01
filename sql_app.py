import streamlit as st
import mysql.connector
import ollama

st.title("Vanna AI - Text to SQL Assistant")

# Connect MySQL
conn = mysql.connector.connect(
    host="localhost",
    user="root",
    password="root123",
    database="company_db"
)

cursor = conn.cursor()

# Input
question = st.text_input("Ask a question about the database")

if st.button("Ask"):

    prompt = f"""
You are a MySQL expert.

Return ONLY SQL query.
No explanation.

Table:
employees(id, name, department, salary)

Question: {question}

SQL:
"""

    response = ollama.chat(
        model="llama3",
        messages=[{"role": "user", "content": prompt}]
    )

    sql = response["message"]["content"]

    # Clean SQL
    sql = sql.replace("```sql", "").replace("```", "")
    sql = sql.strip()
    sql = sql.split(";")[0] + ";"

    st.subheader("Generated SQL:")
    st.code(sql)

    try:
        cursor.execute(sql)
        results = cursor.fetchall()

        st.subheader("Results:")

        # Clean decimal values
        clean_results = []
        for row in results:
            clean_row = []
            for val in row:
                if hasattr(val, "quantize"):
                    clean_row.append(float(val))
                else:
                    clean_row.append(val)
            clean_results.append(clean_row)

        st.table(clean_results)

    except Exception as e:
        st.error(f"Error: {e}")