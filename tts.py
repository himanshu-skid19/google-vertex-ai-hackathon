import aiohttp
from config import API_URL_TTS, headers_tts,ELEVENLABS_API_KEY,ELEVENLABS_VOICE_ID
import httpx
from io import BytesIO

# async def query(payload):
#     async with aiohttp.ClientSession() as session:
#         async with session.post(API_URL_TTS, headers=headers_tts, json=payload) as response:
#             if response.status == 200:
#                 return await response.read()  # Return the raw audio bytes
#             else:
#                 raise Exception(f"Failed to get audio: {response.status} - {await response.text()}")

# async def text_to_speech(text):
#     return await query({"text": text})  # Return raw audio bytes directly

async def text_to_speech(text: str, mime_type: str):
    CHUNK_SIZE = 1024

    url = f"https://api.elevenlabs.io/v1/text-to-speech/{ELEVENLABS_VOICE_ID}"

    headers = {
    "Accept": mime_type,
    "Content-Type": "application/json",
    "xi-api-key": ELEVENLABS_API_KEY
    }

    data = {
        "text": text,
        "model_id": "eleven_monolingual_v1",
        "voice_settings": {
            "stability": 0.5,
            "similarity_boost": 0.5
        }
    }
    
    async with httpx.AsyncClient(timeout=25.0) as client:
        response = await client.post(url, json=data, headers=headers)
        response.raise_for_status()  # Ensure we notice bad responses

        buffer = BytesIO()
        buffer.name = f"output_audio.{mime_type.split('/')[1]}"

        async for chunk in response.aiter_bytes(chunk_size=CHUNK_SIZE):
            if chunk:
                buffer.write(chunk)
        
        buffer.seek(0)
        return buffer.name, buffer.read()