import whisper
import sounddevice as sd
import scipy.io.wavfile as wav

model = whisper.load_model("base")

def speech_to_text():
    print("🎙️ Speak your question (5 seconds)...")
    duration = 8
    fs = 44100
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=1)
    sd.wait()
    wav.write("input.wav", fs, recording)

    result = model.transcribe("input.wav")
    
    return result['text']


