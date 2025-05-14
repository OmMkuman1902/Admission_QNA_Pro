'''from elevenlabs.client import ElevenLabs
from elevenlabs import play
import os
from dotenv import load_dotenv

load_dotenv()

client = ElevenLabs(api_key=os.getenv("ELEVENLABS_API_KEY"))

audio = client.generate(
    text="Hello, welcome to the voice QA agent!",
    voice="Rachel",
    model="eleven_monolingual_v1"
)
play(audio)'''

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

    
'''if __name__=="__main__":
    speak_text("Hello, this Om Mangesh Kuman, now testing the software")'''

