from sqlalchemy.orm import Session
from ml_backend import CulturalItem, SessionLocal

# Create a session to interact with the database
db = SessionLocal()

# Add a sample item to the database
new_item = CulturalItem(
    name="Batik_Salvia_batik",
    category="Batik",
    description="This is a sample batik pattern from Salvia.",
    image_url="path/to/batik_salvia_image.jpg"
)
new_item = CulturalItem(
    name="Batik_Salvia_batik",
    category="Batik",
    description="This is a sample batik pattern from Salvia.",
    image_url="path/to/batik_salvia_image.jpg"
)
new_item = CulturalItem(
    name="Batik_Salvia_batik",
    category="Batik",
    description="This is a sample batik pattern from Salvia.",
    image_url="path/to/batik_salvia_image.jpg"
)
new_item = CulturalItem(
    name="Batik_Salvia_batik",
    category="Batik",
    description="This is a sample batik pattern from Salvia.",
    image_url="path/to/batik_salvia_image.jpg"
)
new_item = CulturalItem(
    name="Batik_Salvia_batik",
    category="Batik",
    description="This is a sample batik pattern from Salvia.",
    image_url="path/to/batik_salvia_image.jpg"
)
new_item = CulturalItem(
    name="Batik_Salvia_batik",
    category="Batik",
    description="This is a sample batik pattern from Salvia.",
    image_url="path/to/batik_salvia_image.jpg"
)
new_item = CulturalItem(
    name="Batik_Salvia_batik",
    category="Batik",
    description="This is a sample batik pattern from Salvia.",
    image_url="path/to/batik_salvia_image.jpg"
)
new_item = CulturalItem(
    name="Batik_Salvia_batik",
    category="Batik",
    description="This is a sample batik pattern from Salvia.",
    image_url="path/to/batik_salvia_image.jpg"
)
# Add to the session and commit
db.add(new_item)
db.commit()
db.refresh(new_item)  # Get the updated instance with an auto-generated ID

print(f"Item added: {new_item.name}")
db.close()
