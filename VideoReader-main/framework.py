from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
from kokoro import KPipeline
import soundfile as sf
from pathlib import Path


# Initialize the BLIP model and processor
image2text_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
image2text_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
text2audio_model = KPipeline(lang_code='a') # <= make sure lang_code matches voice

# Function to load the image from a local file path
def load_image(file_path: str) -> Image:
    """
    Load an image from a local file path and return the PIL Image object.
    """
    img = Image.open(file_path).convert('RGB')
    return img

# Function to generate text from the image using BLIP model
def generate_text_from_image(image: Image) -> str:
    """
    Use the BLIP model to generate a text for the input image.
    """
    # Prepare the image with a prompt
    text = "a photography of"
    inputs = image2text_processor(image, text, return_tensors="pt")
    
    # Generate the text
    out = image2text_model.generate(**inputs)
    
    # Decode and return the generated text
    return image2text_processor.decode(out[0], skip_special_tokens=True)


def generate_audio_from_text(text: str, voice: str = 'af_heart', speed: float = 1) -> bytes:
    """
    Generates audio from the provided text using the Kokoro TTS model.
    
    Args:
    - text (str): The text to convert to speech.
    - voice (str): The voice to use for speech generation (default is 'bf_emma').
    - speed (float): The speed of speech generation (default is 1).

    Returns:
    - audio (bytes): The generated audio in raw byte format.
    """
    # Generate audio
    generation = text2audio_model(text, voice=voice, speed=speed)
    
    # Get the first audio output (assuming there is only one)
    for i, (gs, ps, audio) in enumerate(generation):
        return audio  # Return the audio data as bytes


# Function to save the audio to a specified file path
def save_audio_to_file(audio: bytes, file_path: str, sample_rate: int = 24000) -> None:
    """
    Saves the generated audio to a specified file path as a .wav file.
    
    Args:
    - audio (bytes): The audio data to save.
    - file_path (str): The path where the audio file should be saved.
    - sample_rate (int): The sample rate of the audio (default is 24000).
    """
    # Save audio as a .wav file
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)
    sf.write(file_path, audio, sample_rate)
    print(f"Audio saved as {file_path}")


if __name__ == "__main__":
    # Load the image from a local file
    file_path = 'images/demo2.jpg'  # Change to your file path
    image = load_image(file_path)

    # Generate text for the image
    text = generate_text_from_image(image)
    print(text)

    # Generate audio from the text
    audio = generate_audio_from_text(text)

    # Save the audio to a file
    save_audio_to_file(audio, "audio/output.wav")