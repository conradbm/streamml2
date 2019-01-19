from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from database_setup import Base, Restaurant, MenuItem

# Re-create the database
engine = create_engine('sqlite:///restaurantmenu.db')

# Relate Tables to DB
Base.metadata.bind = engine

# SQL Session Wrapper
DBSession = sessionmaker(bind=engine)
session = DBSession()

# Insert Restaurant Record
myFirstRestaurant = Restaurant(name = "Pizza Palace")
session.add(myFirstRestaurant) # in the staging zone
session.commit()
session.query(Restaurant).all()
#[<database_setup.Restaurant object at 0x7f2516554590>]

# Insert MenuItem Record
cheesepizza = MenuItem(name = "Cheese Pizza", description= "Made with all natural ingredients and fresh mozzarella",$
peppizza = MenuItem(name = "Pepperoni Pizza", description= "Made with all natural ingredients and fresh mozzarella",$
session.add(peppizza)
session.commit()
session.query(MenuItem).all()
#[<database_setup.MenuItem object at 0x7f25165547d0>]

# Loop through multple instances Example
veggieBurgers = session.query(MenuItem).filter_by(name = "Veggie Burger")
for vb in veggieBurgers:
     print ("Item Before", vb.id, vb.price, vb.restaurant.name)
     if vb.price != "$2.99":
         vb.price="$2.99"
         print("Item Price Changed: ", vb.id,vb.price,vb.restaurant.name)


# Delete Example
session.query(MenuItem).filter_by(name = "Spinach Ice Cream").one()
#<database_setup.MenuItem object at 0x7f29ef413610>
spinach = session.query(MenuItem).filter_by(name = "Spinach Ice Cream").one()
print(spinach.restaurant.name)
#Auntie Ann's Diner 
session.delete(spinach)
session.commit()
spinach = session.query(MenuItem).filter_by(name="Spinach Ice Cream").one()
#Traceback (most recent call last):
#  File "<stdin>", line 1, in <module>
#  File "/usr/local/lib/python2.7/dist-packages/sqlalchemy/orm/query.py", line 2953, in one
#    raise orm_exc.NoResultFound("No row was found for one()")
#sqlalchemy.orm.exc.NoResultFound: No row was found for one()
