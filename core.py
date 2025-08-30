from sqlalchemy import create_engine, Column, Integer, String, Text
from sqlalchemy.orm import sessionmaker, declarative_base

# --- Database Setup ---
DATABASE_URL = "sqlite:///./users.db"  # SQLite database file

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False}  # Required for SQLite in multi-threaded apps like Streamlit
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# --- Models ---
class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True, nullable=False)
    email = Column(String, unique=True, index=True, nullable=False)
    password = Column(String, nullable=False)

class Resume(Base):
    __tablename__ = "resumes"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, nullable=False)
    content = Column(Text, nullable=False)  # stores resume text
    skills = Column(Text, nullable=True)    # extracted skills

# --- Init Function ---
def init_db():
    """
    Creates tables if they don't exist.
    Run this before starting the app to ensure database is ready.
    """
    Base.metadata.create_all(bind=engine)

