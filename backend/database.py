from sqlalchemy import create_engine, Column, Integer, String, Text, ForeignKey, Float, JSON
from sqlalchemy.orm import declarative_base, sessionmaker, relationship
import json
import time

Base = declarative_base()

class AudienceModel(Base):
    __tablename__ = 'audiences'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String)
    type = Column(String) # B2B, B2C, etc.
    description = Column(String)
    
    personas = relationship("PersonaModel", back_populates="audience")

class PersonaModel(Base):
    __tablename__ = 'personas'
    
    id = Column(String, primary_key=True)
    audience_id = Column(Integer, ForeignKey('audiences.id')) # Link to audience
    name = Column(String)
    role = Column(String)
    company = Column(String)
    avatar = Column(String)
    psychographics = Column(Text)
    past_behavior = Column(Text)
    
    audience = relationship("AudienceModel", back_populates="personas")
    
    def to_dict(self):
        return {
            "id": self.id,
            "name": self.name,
            "role": self.role,
            "company": self.company,
            "avatar": self.avatar,
            "psychographics": self.psychographics,
            "pastBehavior": self.past_behavior
        }

class SimulationModel(Base):
    __tablename__ = 'simulations'
    
    id = Column(String, primary_key=True)
    timestamp = Column(Integer)
    subject = Column(String)
    body = Column(Text)
    cta = Column(String)
    audience_target = Column(String)
    
    # Metrics stored as JSON
    metrics = Column(JSON)
    insights = Column(JSON)
    
    responses = relationship("ResponseModel", back_populates="simulation")

class ResponseModel(Base):
    __tablename__ = 'responses'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    simulation_id = Column(String, ForeignKey('simulations.id'))
    persona_id = Column(String, ForeignKey('personas.id'))
    
    action = Column(String)
    sentiment = Column(String)
    comment = Column(Text)
    detailed_reasoning = Column(Text)
    
    simulation = relationship("SimulationModel", back_populates="responses")
    persona = relationship("PersonaModel")

import os
from dotenv import load_dotenv

load_dotenv()

# Database Setup
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./email_predictor.db")
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def init_db():
    Base.metadata.create_all(bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
