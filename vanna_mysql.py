import mysql.connector
import ollama

# Connect to MySQL
conn = mysql.connector.connect(
    host="localhost",
    user="root",
    password="root123",
    database="company_db"
)

cursor = conn.cursor()

# User question
question = "Show employees in Engineering"

# Prompt for the LLM
prompt = f"""
You are an expert SQL generator.
return ONLY the SQL query.
DO NOT explain anything.
DO NOT add text.
DO NOT use markdown.

Database schema:
employees(id, name, department, salary)



Question: {question}

SQL query:
"""

# Ask Ollama
response = ollama.chat(
    model="llama3",
    messages=[{"role": "user", "content": prompt}]
)

sql = response["message"]["content"]
sql=sql.replace("'''sql","").replace("'''","")
sql=sql.replace("The answer is:","").strip()
sql=sql.split(";")[0]+";"
print("\nGenerated SQL:\n", sql)

# Run SQL
cursor.execute(sql)

print("\nResults:\n")

for row in cursor.fetchall():
    print(row)

conn.close()