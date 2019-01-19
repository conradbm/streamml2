import os
import sys
from sqlalchemy import Column, ForeignKey, Integer, String
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy import create_engine

Base = declarative_base()


class Restaurant(Base):
    __tablename__ = 'restaurant'

    # PK
    id = Column(Integer, primary_key=True)
    
    name = Column(String(250), nullable=False)


class MenuItem(Base):
    __tablename__ = 'menu_item'

    # PK
    id = Column(Integer, primary_key=True)
    name = Column(String(80), nullable=False)
    description = Column(String(250))
    price = Column(String(8))
    course = Column(String(250))
    
    # FK
    restaurant_id = Column(Integer, ForeignKey('restaurant.id'))
    
    # Relationship From
    restaurant = relationship(Restaurant)


# Create the db file
engine = create_engine('sqlite:///restaurantmenu.db')


# Link Tables to DB via Base
Base.metadata.create_all(engine)
