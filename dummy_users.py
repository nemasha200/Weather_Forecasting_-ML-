import sqlite3

conn = sqlite3.connect('users.db')
c = conn.cursor()

# Insert dummy users
c.execute("INSERT INTO users (username, email, full_name) VALUES (?, ?, ?)",
          ("john_doe", "john@example.com", "John Doe"))
c.execute("INSERT INTO users (username, email, full_name) VALUES (?, ?, ?)",
          ("jane_smith", "jane@example.com", "Jane Smith"))

conn.commit()
conn.close()

print("Test users added.")
