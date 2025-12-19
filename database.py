from sqlalchemy import create_engine, Column, Integer, String, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

# 1. SETUP SQLITE
DATABASE_URL = "sqlite:///./heritage.db" 
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# 2. DEFINE THE TABLE
class CulturalItem(Base):
    __tablename__ = "cultural_items"
    
    id = Column(Integer, primary_key=True, index=True)
    
    # --- CHANGED 'filename' TO 'name' TO FIX YOUR ERROR ---
    # This matches the Keras class names (e.g., "Peksi_batik")
    name = Column(String, unique=True, index=True) 
    
    title = Column(String)
    category = Column(String)
    place_of_discovery = Column(String)
    origin = Column(String)
    description = Column(Text)
    fun_fact = Column(Text)
    
    # This holds the file path like "peksi.jpg"
    image_url = Column(String)

# 3. CREATE TABLES
def init_db():
    Base.metadata.create_all(bind=engine)

if __name__ == "__main__":
    print("Creating database tables...")
    init_db()
    print("Tables created successfully!")