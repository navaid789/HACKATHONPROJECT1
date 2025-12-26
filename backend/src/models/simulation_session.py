from sqlalchemy import Column, Integer, String, Text, DateTime, Boolean, ForeignKey, JSON
from sqlalchemy.ext.declarative import declarative_base
from datetime import datetime

Base = declarative_base()

class SimulationSession(Base):
    __tablename__ = "simulation_sessions"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    simulation_name = Column(String(200), nullable=False)  # Name of the simulation
    simulation_type = Column(String(100), nullable=False)  # gazebo, isaac, custom, etc.
    parameters = Column(JSON, nullable=True)  # Simulation parameters
    results = Column(Text, nullable=True)  # Simulation results or output
    status = Column(String(50), default="pending")  # pending, running, completed, failed
    started_at = Column(DateTime, nullable=True)
    ended_at = Column(DateTime, nullable=True)
    duration = Column(Integer, nullable=True)  # Duration in seconds
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    def __repr__(self):
        return f"<SimulationSession(id={self.id}, user_id={self.user_id}, simulation_name={self.simulation_name})>"