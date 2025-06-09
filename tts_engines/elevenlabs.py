# tts_engines/elevenlabs.py

import requests

class ElevenLabs:
    def __init__(self, api_key):
        self.api_key = api_key
        self.voice_id = "21m00Tcm4TlvDq8ikWAM"  # Rachel, adjust as needed

    def synthesize(self, text, output_path="output_eleven.wav"):
        url = f"https://api.elevenlabs.io/v1/text-to-speech/{self.voice_id}"
        headers = {
            "xi-api-key": self.api_key,
            "Content-Type": "application/json"
        }
        data = {
            "text": text,
            "model_id": "eleven_monolingual_v1",
            "voice_settings": {
                "stability": 0.75,
                "similarity_boost": 0.75
            }
        }
        response = requests.post(url, headers=headers, json=data)
        if response.status_code == 200:
            with open(output_path, "wb") as f:
                f.write(response.content)
            return output_path
        else:
            raise RuntimeError(f"ElevenLabs TTS request failed: {response.status_code}, {response.text}")