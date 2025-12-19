import io
import os  # <--- NEW IMPORT (Crucial for finding folders)
import numpy as np
import tensorflow as tf
from PIL import Image
from fastapi import FastAPI, UploadFile, File, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles  # <--- NEW IMPORT (To serve images)
from pydantic import BaseModel
from sqlalchemy import create_engine, Column, Integer, String, Text
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.ext.declarative import declarative_base
import uvicorn

# 1. SETUP DATABASE & MODELS
DATABASE_URL = "sqlite:///./heritage.db"
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class CulturalItem(Base):
    __tablename__ = "cultural_items"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, unique=True, index=True)
    title = Column(String)
    category = Column(String)
    place_of_discovery = Column(String)
    origin = Column(String)
    description = Column(Text)
    fun_fact = Column(Text)
    image_url = Column(String)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# 2. SETUP AI MODEL & APP
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

script_dir = os.path.dirname(os.path.abspath(__file__))
static_folder_path = os.path.join(script_dir, "static_images")

if os.path.isdir(static_folder_path):
    print(f"✅ Serving images from: {static_folder_path}")
    app.mount("/static", StaticFiles(directory=static_folder_path), name="static")
else:
    print(f"❌ ERROR: Folder not found at {static_folder_path}")
    print("   -> Please create a folder named 'static_images' next to main.py")

MODEL_PATH = 'artifact_classifier.keras'
print("Loading Keras Model...")
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print("Model loaded successfully.")
except Exception as e:
    print(f"ERROR LOADING MODEL: {e}")
    model = None

CLASS_NAMES = [
    'Batik_Salvia_batik', 'Garuda_Wisnu_Kencana', 'Patung_Dirgantara',
    'patung_surabaya', 'Peksi_batik', 'wayang_madya', 
    'Wayang_Purwa', 'Wayang_Rama'
]

def process_image(image_bytes: bytes) -> np.ndarray:
    IMG_SIZE = 180
    img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    img = img.resize((IMG_SIZE, IMG_SIZE))
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)
    return img_array

# 3. API RESPONSE SCHEMAS
class ItemDetailsSchema(BaseModel):
    title: str
    category: str
    place_of_discovery: str
    origin: str
    description: str
    fun_fact: str
    image_url: str  # This will hold the full HTTP link

    class Config:
        from_attributes = True # updated for Pydantic V2

class PredictionResponseSchema(BaseModel):
    motif: str
    confidence: float
    item_details: ItemDetailsSchema

# 4. THE ENDPOINT
@app.post("/scan_predict", response_model=PredictionResponseSchema)
async def scan_predict(image: UploadFile = File(...), db: Session = Depends(get_db)):
    
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")

    # A. Predict
    image_bytes = await image.read()
    processed_image = process_image(image_bytes)
    predictions = model.predict(processed_image)
    score = tf.nn.softmax(predictions[0])
    
    predicted_motif = CLASS_NAMES[np.argmax(score)]
    confidence = float(100 * np.max(score))

    print(f"Predicted: {predicted_motif} ({confidence:.2f}%)")

    # B. Query DB
    item_details = db.query(CulturalItem).filter(CulturalItem.name == predicted_motif).first()

    if item_details is None:
        raise HTTPException(status_code=404, detail=f"Item '{predicted_motif}' not found in DB.")

    #base_url = "http://10.10.10.73:8000/static/" #for emu
    base_url = "http://192.168.170.102:8000/static/" #phone
    
    # If the DB already has a full link, keep it. Otherwise, add the base_url.
    final_image_url = item_details.image_url
    if final_image_url and not final_image_url.startswith("http"):
        final_image_url = base_url + final_image_url

    # Create response object
    response_details = ItemDetailsSchema(
        title=item_details.title,
        category=item_details.category,
        place_of_discovery=item_details.place_of_discovery,
        origin=item_details.origin,
        description=item_details.description,
        fun_fact=item_details.fun_fact,
        image_url=final_image_url # <--- Sending the full link!
    )

    return {
        "motif": predicted_motif,
        "confidence": confidence,
        "item_details": response_details
    }

# ==========================================
# 5. RUN SERVER
# ==========================================
if __name__ == "__main__":
    import uvicorn
    # host="0.0.0.0" adalah KUNCI agar HP bisa masuk
    # port=8000 sesuaikan dengan port Anda
    uvicorn.run(app, host="0.0.0.0", port=8000)