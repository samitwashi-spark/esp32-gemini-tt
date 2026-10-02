import os
import asyncio
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
import edge_tts
from pydub import AudioSegment
from langdetect import detect

app = FastAPI()

TEMP_MP3 = "/tmp/temp_response.mp3"
FINAL_WAV = "/tmp/final_response.wav"

def get_voice_by_langdetect(text: str) -> str:
    """Detects language grammar using langdetect and assigns the neural voice"""
    try:
        # Detect the language shortcode (e.g., 'bn', 'en', 'hi', 'ar')
        lang = detect(text)
        print(f"langdetect identified language code: {lang}")
        
        if lang == 'bn':
            print("Voice selected: Bangla -> Nabanita")
            return "bn-BD-NabanitaNeural"
        elif lang == 'hi':
            print("Voice selected: Hindi -> Swara")
            return "hi-IN-SwaraNeural"
        elif lang == 'ar':
            print("Voice selected: Arabic -> Fatima")
            return "ar-AE-FatimaNeural"
        elif lang == 'es':
            print("Voice selected: Spanish -> Elvira")
            return "es-ES-ElviraNeural"
        elif lang == 'fr':
            print("Voice selected: French -> Denise")
            return "fr-FR-DeniseNeural"
            
    except Exception as e:
        print(f"langdetect failed or text was obscure: {e}. Falling back to English.")
        
    # Default fallback to English if detection fails or language isn't explicitly mapped
    return "en-US-EmmaNeural"

async def generate_true_wav(text: str, voice_model: str):
    """Generates the speech audio file and converts it into a raw uncompressed PCM WAV container"""
    communicate = edge_tts.Communicate(text, voice_model)
    await communicate.save(TEMP_MP3)
    
    # Read the temporary compressed file from Edge TTS
    sound = AudioSegment.from_mp3(TEMP_MP3)
    
    # Force configurations to perfectly stream straight into the built-in ESP32 hardware I2S buffer
    sound = sound.set_frame_rate(24000) # 24kHz Sample Rate
    sound = sound.set_channels(1)       # Mono Audio Output
    sound = sound.set_sample_width(2)   # 16-bit sound depth (2 bytes per sample)
    
    sound.export(FINAL_WAV, format="wav")

@app.get("/ping")
def ping():
    return {"status": "alive"}

@app.get("/tts")
async def text_to_speech_wav(text: str):
    if not text:
        raise HTTPException(status_code=400, detail="Text parameter cannot be empty")
        
    try:
        # Step 1: Detect the language using langdetect library
        selected_voice = get_voice_by_langdetect(text)

        # Step 2: Convert text using the specific voice mapping
        await generate_true_wav(text, selected_voice)

        # Step 3: Stream the output back down to the microcontroller
        def iterfile():
            with open(FINAL_WAV, mode="rb") as file_like:
                yield from file_like

        return StreamingResponse(iterfile(), media_type="audio/wav")

    except Exception as e:
        print(f"Server Processing Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
