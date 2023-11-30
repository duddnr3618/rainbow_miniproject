import uuid
from sqlalchemy import Column, ForeignKey, Integer, LargeBinary, String, DateTime, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.sql import func

Base = declarative_base()

class Image_Data(Base):
    __tablename__ = 'image2'
    id = Column(String, primary_key=True, default=str(uuid.uuid4()), index=True)
    webcamName = Column(Text)
    phoneNumber = Column(Integer)
    capturedImage = Column(LargeBinary)
    
class Attendance(Base):
    __tablename__ = 'attendance'
    id = Column(String, primary_key=True, default=str(uuid.uuid4()), index=True)
    image_id = Column(String, ForeignKey('image2.id'))
    webcam_name = Column(String, ForeignKey('image2.webcamName'))
    check_in_time = Column(DateTime(timezone=True), server_default=func.now())
    check_out_time = Column(DateTime(timezone=True))
        
        
