import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from db.models import Base

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://megaai:megaai123@localhost:5432/megaai")

engine = create_engine(DATABASE_URL, pool_pre_ping=True)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def init_db():
    Base.metadata.create_all(bind=engine)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
