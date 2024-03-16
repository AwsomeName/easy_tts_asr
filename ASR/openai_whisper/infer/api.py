from transformers import WhisperProcessor, WhisperForConditionalGeneration
import numpy as np
from pydub import AudioSegment
import torch

class ASREngine():
    def __init__(
        self,
        model_path = "/home/lc/models/openai/whisper-large-v3",
    ):
        self.model_path = model_path
        self.processor = WhisperProcessor.from_pretrained(self.model_path)
        self.model = WhisperForConditionalGeneration.from_pretrained(
            self.model_path, device_map="auto")
            # self.model_path, torch_dtype=torch.float16, device_map="auto")
        self.model.config.forced_decoder_ids = self.processor.get_decoder_prompt_ids(
            language="chinese", task="transcribe"
        )
        
    def load_mp3(self, audio_path):
        audio = AudioSegment.from_mp3(audio_path)
        samples = np.array(audio.get_array_of_samples())
        sr = audio.frame_rate
        if audio.channels == 2:
            samples = samples.reshape((-1, 2))
        return np.float32(samples) / 2**15, sr
        
    def test(self, audio_path):
        data, sr = self.load_mp3(audio_path)
        
        input_features = self.processor(data, sampling_rate=sr, return_tensors="pt").input_features.cuda()
        predicted_ids = self.model.generate(input_features)
        transcription = self.processor.batch_decode(predicted_ids, skip_special_tokens=False)
        print(transcription)
        return transcription


    
if __name__ == "__main__":
    file_path = "/home/lc/data/radio/yuanshen_zh/雷电将军/sr16000/997ee8950033ce78a1e1204cef725d51226_145647756923704119.mp3"
    asrE = ASREngine()
    asrE.test(file_path)