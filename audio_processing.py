from pydub import AudioSegment
from transformers import pipeline
import soundfile as sf
import librosa
import io
import logging

# Set up basic configuration for logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# ASR Pipeline
asr_pipeline = pipeline("automatic-speech-recognition", model="jonatasgrosman/wav2vec2-large-xlsr-53-english")

def process_audio(audio_buffer,mime_type):
    wav_buffer = None
    try: 
        # Attempt to read and process the file
        if  "mp4" in mime_type:
            audio_segment = AudioSegment.from_file(audio_buffer, format="mp4")
        elif "webm" in mime_type:
            audio_segment = AudioSegment.from_file(audio_buffer, format="webm")
        else:
            logging.error(f"Unsupported file format")
            return None, None
        
        mono_audio_segment = audio_segment.set_channels(1)
        wav_buffer = io.BytesIO()
        mono_audio_segment.export(wav_buffer, format="wav")
        wav_buffer.seek(0)

        # Read and process the WAV buffer
        audio, samplerate = sf.read(wav_buffer, dtype='float32')
        if samplerate != 16000:
            audio = librosa.resample(audio, orig_sr=samplerate, target_sr=16000)

        # Use ASR pipeline to generate transcript
        audio_input = {"raw": audio, "sampling_rate": 16000}
        transcript = asr_pipeline(audio_input)['text']
        user_question = transcript
    except FileNotFoundError:
        logging.error(f"File not found: {audio_buffer}")
        return None, None
    except Exception as e:
        logging.error(f"Failed during conversion or read: {e}")
        return None, None
    finally:
        # Ensure resources are cleaned up
        if wav_buffer:
            wav_buffer.close()
    return user_question, transcript
