# tts_engines/azure_tts.py

import azure.cognitiveservices.speech as speechsdk

class AzureTTS:
    def __init__(self, key, region):
        self.speech_key = key
        self.service_region = region
        self.voice = "en-US-JennyNeural"  # Change to desired voice
        self.audio_format = speechsdk.SpeechSynthesisOutputFormat.Riff16Khz16BitMonoPcm

    def synthesize(self, text, output_file="azure_tts_output.wav"):
        speech_config = speechsdk.SpeechConfig(subscription=self.speech_key, region=self.service_region)
        speech_config.speech_synthesis_voice_name = self.voice
        speech_config.set_speech_synthesis_output_format(self.audio_format)

        audio_config = speechsdk.audio.AudioOutputConfig(filename=output_file)

        synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config, audio_config=audio_config)
        result = synthesizer.speak_text_async(text).get()

        if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
            print(f"[Azure TTS] Speech synthesized to {output_file}")
            return output_file
        else:
            print(f"[Azure TTS] Error: {result.reason}")
            return None