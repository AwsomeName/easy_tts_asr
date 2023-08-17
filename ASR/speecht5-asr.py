from transformers import SpeechT5Processor, SpeechT5ForSpeechToText
from datasets import load_dataset

local_path = "/home/lc/models/"
# local_path = ""
local_data = "/home/lc/hf_dataset/"
# local_data = ""

dataset = load_dataset(local_data + "hf-internal-testing/librispeech_asr_demo", "clean", split="validation")
dataset = dataset.sort("id")
sampling_rate = dataset.features["audio"].sampling_rate
example_speech = dataset[0]["audio"]["array"]

# 注意这两个应该是一样的，可以把音频或文本转为emb，作为Decoder的输入
processor = SpeechT5Processor.from_pretrained(local_path + "microsoft/speecht5_tts")
# processor = SpeechT5Processor.from_pretrained(local_path + "microsoft/speecht5_asr")

model = SpeechT5ForSpeechToText.from_pretrained(local_path + "microsoft/speecht5_asr")

inputs = processor(audio=example_speech, sampling_rate=sampling_rate, return_tensors="pt")

predicted_ids = model.generate(**inputs, max_length=100)

transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)
print(transcription[0])
