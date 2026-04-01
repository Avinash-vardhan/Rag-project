import mysql.connector
import ollama

# Connect MySQL
conn = mysql.connector.connect(
    host="localhost",
    user="root",
    password="root123",
    database="company_db"
)

cursor = conn.cursor()

while True:

    question = input("\nAsk a question (or type exit): ")

    if question.lower() == "exit":
        break

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

    # clean SQL
    sql = sql.replace("```sql", "").replace("```", "")
    sql = sql.strip()
    sql = sql.split(";")[0] + ";"

    print("\nGenerated SQL:")
    print(sql)

    cursor.execute(sql)

    results = cursor.fetchall()

    print("\nResults:\n")

    for row in results:
        print(row)

conn.close()