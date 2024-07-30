import sqlite3

class Category:
    def __init__(self, name):
        self.name = name

class Factor:
    def __init__(self, name, category, description=''):
        self.name = name
        self.category = category
        self.description = description

class DatabaseHandler:
    def __init__(self, db_name='forex_factors.db'):
        self.conn = sqlite3.connect(db_name)
        self.create_tables()

    def create_tables(self):
        with self.conn:
            self.conn.execute('''CREATE TABLE IF NOT EXISTS Category (
                                    id INTEGER PRIMARY KEY,
                                    name TEXT UNIQUE NOT NULL
                                 )''')
            self.conn.execute('''CREATE TABLE IF NOT EXISTS Factor (
                                    id INTEGER PRIMARY KEY,
                                    name TEXT NOT NULL,
                                    category_id INTEGER,
                                    description TEXT,
                                    FOREIGN KEY (category_id) REFERENCES Category (id)
                                 )''')

    def add_category(self, category):
        with self.conn:
            cursor = self.conn.cursor()
            cursor.execute('INSERT OR IGNORE INTO Category (name) VALUES (?)', (category.name,))
            self.conn.commit()
            return cursor.lastrowid

    def add_factor(self, factor):
        with self.conn:
            cursor = self.conn.cursor()
            cursor.execute('SELECT id FROM Category WHERE name = ?', (factor.category.name,))
            category_id = cursor.fetchone()
            if category_id:
                category_id = category_id[0]
            else:
                category_id = self.add_category(factor.category)
            cursor.execute('INSERT INTO Factor (name, category_id, description) VALUES (?, ?, ?)',
                           (factor.name, category_id, factor.description))
            self.conn.commit()

    def close(self):
        self.conn.close()
