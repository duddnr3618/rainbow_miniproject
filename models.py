from sqlalchemy import create_engine, Column, Integer, String,BLOB,Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from pydantic import BaseModel
from fastapi import  UploadFile
from sqlalchemy import create_engine, Column, Integer, Text, BLOB,LargeBinary
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from pydantic import BaseModel
import os

Base = declarative_base()

current_directory = os.path.dirname(os.path.realpath(__file__))
DATABASE_URL = f"sqlite:///{os.path.join(current_directory, 'venv/user.db')}"
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


class User(Base):
    __tablename__ = "user"
    userName = Column(Text)
    userPhone = Column(Integer, primary_key=True, index=True)
    userImage = Column(BLOB)

class ImageCreate(BaseModel):
    userName: str
    userPhone: int
    profile_pic: bytes


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()