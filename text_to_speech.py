'''# text_to_speech.py

from elevenlabs.client import ElevenLabs
from elevenlabs import play
import os
from dotenv import load_dotenv

# Load environment variables (API key)
load_dotenv()

# Initialize ElevenLabs client
client = ElevenLabs(api_key=os.getenv("ELEVENLABS_API_KEY"))

def speak_text(text, voice="Antoni", model="eleven_monolingual_v1"):
    audio = client.generate(
        text=text,
        voice=voice,
        model=model
    )
    play(audio)
if __name__=="__main__":
    speak_text("Hey!! This is tesing 1,2,3")'''
from gtts import gTTS
import os
from playsound import playsound


def speak_text(input_text):
    #Your text to convert
    

    # Convert to speech using gTTS
    tts = gTTS(text=input_text, lang='en',slow=False)

    # Save the audio file
    tts.save("output.mp3")

    # Play the audio
    playsound("output.mp3")

if __name__=="__main__":
    speak_text("Hello this is mike testing")