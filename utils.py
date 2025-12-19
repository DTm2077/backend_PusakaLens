import numpy as np
from PIL import Image
import tensorflow as tf
import io

# Function to process an image and prepare it for prediction by the TensorFlow model
def process_image(image_bytes: bytes) -> np.ndarray:
    """
    This function processes the image bytes, resizes the image to the correct dimensions,
    normalizes it, and converts it into a numpy array for model prediction.
    """
    IMG_SIZE = 180  # Image size expected by the model (can be adjusted depending on your model)
    
    # Open the image from bytes
    img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    
    # Resize the image to the expected size for the model
    img = img.resize((IMG_SIZE, IMG_SIZE))
    
    # Convert the image to a numpy array and scale pixel values to [0, 1]
    img_array = tf.keras.utils.img_to_array(img)  # Convert the image to a numpy array
    img_array = img_array / 255.0  # Normalize the image
    
    # Add a batch dimension (model expects 4D input: [batch_size, height, width, channels])
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array
