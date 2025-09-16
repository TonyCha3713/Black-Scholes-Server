# app/models.py
from sqlalchemy import Column, Integer, Float, DateTime, ForeignKey
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship
from .db import Base

class Calculation(Base):
    __tablename__ = "calculations"
    id = Column(Integer, primary_key=True)
    spot = Column(Float, nullable=False)
    strike = Column(Float, nullable=False)
    time_to_expiry = Column(Float, nullable=False)   # years
    volatility = Column(Float, nullable=False)
    risk_free_rate = Column(Float, nullable=False)
    purchase_call = Column(Float, nullable=True)
    purchase_put  = Column(Float, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    heatmap_points = relationship("HeatmapPoint", back_populates="calc", cascade="all, delete")

class HeatmapPoint(Base):
    __tablename__ = "heatmap_points"
    id = Column(Integer, primary_key=True)
    calculation_id = Column(Integer, ForeignKey("calculations.id"), index=True)
    spot_shock = Column(Float, nullable=False)
    vol = Column(Float, nullable=False)
    call_value = Column(Float, nullable=True)
    put_value  = Column(Float, nullable=True)
    call_pnl   = Column(Float, nullable=True)
    put_pnl    = Column(Float, nullable=True)
    calc = relationship("Calculation", back_populates="heatmap_points")
