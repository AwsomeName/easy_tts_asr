from pydub import AudioSegment
import numpy as np
from transformers import WhisperProcessor, WhisperForConditionalGeneration

# m_path = "/home/lc/models/openai/whisper-base"
m_path = "/home/lc/models/openai/whisper-large-v3"
# load model and processor
processor = WhisperProcessor.from_pretrained(m_path)
model = WhisperForConditionalGeneration.from_pretrained(m_path)
model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(language="chinese", task="transcribe")

# print("prompt:", model.config.forced_decoder_ids)
# exit()

data_path = "/home/lc/data/radio/yuanshen_zh/雷电将军/sr16000/997ee8950033ce78a1e1204cef725d51226_145647756923704119.mp3"
audio = AudioSegment.from_mp3(data_path)
samples = np.array(audio.get_array_of_samples())
sr = audio.frame_rate
if audio.channels == 2:
    samples = samples.reshape((-1, 2))

input_features = processor(np.float32(samples) / 2**15, sampling_rate=sr, return_tensors="pt").input_features 

# generate token ids
predicted_ids = model.generate(input_features)
# decode token ids to text
transcription = processor.batch_decode(predicted_ids, skip_special_tokens=False)
print("t:", transcription)
print("===================")

transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)
print("===================")
print("t:", transcription)