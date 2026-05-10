import os
import logging
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import NullPool
from db.models import Base

logger = logging.getLogger(__name__)

DATABASE_URL = os.getenv("DATABASE_URL", "")

if not DATABASE_URL:
    logger.warning("DATABASE_URL not set — using SQLite fallback")
    DATABASE_URL = "sqlite:////tmp/megaai.db"
    engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
else:
    # Railway provides postgres:// but SQLAlchemy requires postgresql://
    if DATABASE_URL.startswith("postgres://"):
        DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)
    engine = create_engine(
        DATABASE_URL,
        pool_pre_ping=True,
        pool_timeout=10,
        connect_args={"connect_timeout": 10},
    )

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def init_db():
    try:
        Base.metadata.create_all(bind=engine)
        logger.info("Database initialized successfully")
    except Exception as e:
        logger.error(f"Database initialization failed: {e}")


def get_db():
    db = SessionLocal()
    try:
        yield db
    except Exception as e:
        logger.error(f"DB session error: {e}")
        db.rollback()
        raise
    finally:
        db.close()
