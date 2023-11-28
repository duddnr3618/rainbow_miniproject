# models.py
from sqlalchemy import Column, Integer, LargeBinary, DateTime
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

class ImageData(Base):
    __tablename__ = 'image_data'
    id = Column(Integer, primary_key=True, index=True)
    income_time = Column(DateTime, default=False)
    outcome_time = Column(DateTime, default=False)
    image_binary = Column(LargeBinary)

class UserInfo(Base):
    __tablename__ = "user_info"
    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    phone_number = Column(str, default=False)
    image_binary = Column(LargeBinary)
