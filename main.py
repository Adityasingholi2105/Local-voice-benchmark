from faster_whisper import WhisperModel
import time
from ollama import chat
import wave
from piper import PiperVoice

def main():
    # Speech to Text 
    model = WhisperModel("tiny", device="cpu", compute_type="int8")
    
    stt_start = time.perf_counter()
    segments, info = model.transcribe("data/sample.m4a",language="en", beam_size=5)
    segments = list(segments)
   
    stt_time = time.perf_counter() - stt_start
    transcript = "".join(segment.text.strip() for segment in segments )
    print(f"STT time: {stt_time:.2f} seconds")
    print(f"Detected language: {info.language}")
    print("You:", transcript)

    #Send the transcript to rhe local LLM 
    llm_start = time.perf_counter()
    response = chat(
        model="qwen2.5:1.5b",
        messages=[
            
            {"role": "user", "content": f"Reply in one short sentence: {transcript}"},
        

        ],
    )
    llm_time = time.perf_counter() - llm_start

    reply = response.message.content
    print("Assistant:", reply)
    print(f"LLM time: {llm_time:.2f} seconds")

    # Convert the LLm reply to speech 
    voice = PiperVoice.load("voices/en_US-lessac-medium.onnx")

    tts_start = time.perf_counter()
    with wave.open("data/reply.wav", "wb") as wav_file:
        voice.synthesize_wav(response.message.content,wav_file)
    tts_time = time.perf_counter() - tts_start

    print(f"TTS time; {tts_time:.2f} seconds")
    print("Spoken reply saved to data/reply.wav")

if __name__ == "__main__":
    main()
   
    

