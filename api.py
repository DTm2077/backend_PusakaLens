import io
import tensorflow as tf
import numpy as np
from PIL import Image
from fastapi import FastAPI, UploadFile, File, HTTPException, Depends
from pydantic import BaseModel
from sqlalchemy import create_engine, Column, Integer, String, Text
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.ext.declarative import declarative_base

# FastAPI instance
app = FastAPI()

# Model Loading
MODEL_PATH = 'artifact_classifier.keras'
model = tf.keras.models.load_model(MODEL_PATH)

# Class Names for Batik, Patung, Wayang (use the exact names used during training)
CLASS_NAMES = ['Batik_Salvia_batik', 'Garuda_Wisnu_Kencana', 'Patung_Dirgantara',
               'patung_surabaya', 'Peksi_batik', 'wayang_madya', 'Wayang_Purwa', 
               'Wayang_Rama']

# Helper function to process the image
def process_image(image_bytes: bytes) -> np.ndarray:
    """Process image into the format for model prediction."""
    IMG_SIZE = 180  # Image size expected by the model
    img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    img = img.resize((IMG_SIZE, IMG_SIZE))  # Resize image
    img_array = tf.keras.utils.img_to_array(img)  # Convert image to array
    img_array = tf.expand_dims(img_array, 0)  # Add batch dimension
    return img_array

# Database Setup
DATABASE_URL = "sqlite:///./test.db"
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Dependency to get the database session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Database models for cultural items (Batik, Patung, Wayang)
class CulturalItem(Base):
    __tablename__ = "cultural_items"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True)  # Example: batik-bali
    category = Column(String)
    description = Column(Text)
    image_url = Column(String)  # URL to the image (assets/images/... or external URL)

# Pydantic models for response
class CulturalItemSchema(BaseModel):
    id: int
    name: str
    category: str
    description: str
    image_url: str

    class Config:
        orm_mode = True

class PredictionResponseSchema(BaseModel):
    motif: str
    confidence: float
    item_details: CulturalItemSchema

@app.post("/scan_predict", response_model=PredictionResponseSchema)
async def scan_predict(image: UploadFile = File(...), db: Session = Depends(get_db)):
    """API endpoint to predict artifact category from an image."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")
    
    # Read the image bytes
    image_bytes = await image.read()
    
    # Process the image
    processed_image = process_image(image_bytes)
    
    # Run prediction
    predictions = model.predict(processed_image)
    score = tf.nn.softmax(predictions[0])  # Get probability scores
     
    predicted_motif = CLASS_NAMES[np.argmax(score)]
    confidence = 100 * np.max(score)  # Confidence as percentage

    # Fetch item details from the database based on predicted class name
    item_details = db.query(CulturalItem).filter(CulturalItem.name == predicted_motif).first()

    if item_details is None:
        raise HTTPException(status_code=404, detail="Item not found in the database.")
    
    return {"motif": predicted_motif, "confidence": confidence, "item_details": item_details}
