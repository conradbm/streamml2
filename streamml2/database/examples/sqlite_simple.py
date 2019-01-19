import sqlite3
conn = sqlite3.connect('restaurantmenu.db')

c = conn.cursor()
c.execute('''
	  CREATE TABLE restaurant
	  (id INTEGER PRIMARY KEY ASC, name varchar(250) NOT NULL)
	  ''')
c.execute('''
	  CREATE TABLE menu_items
	  (id INTEGER PRIMARY KEY ASC, name varchar(250), price varchar(250),
	  description varchar(250) NOT NULL, restaurant_id INTEGER NOT NULL,
	  FOREIGN KEY(restaurant_id) REFERENCES restaurant(id))
	  ''')

conn.commit()
conn.close()
