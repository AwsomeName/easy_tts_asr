# Following pip packages need to be installed:
# !pip install git+https://github.com/huggingface/transformers sentencepiece datasets

from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
from datasets import load_dataset
import torch
import soundfile as sf
from datasets import load_dataset

local_path = "/home/lc/models/"
# local_path = ""
local_data = "/home/lc/hf_dataset/"
# local_data = ""

processor = SpeechT5Processor.from_pretrained(local_path + "microsoft/speecht5_tts")
model = SpeechT5ForTextToSpeech.from_pretrained(local_path + "microsoft/speecht5_tts")
vocoder = SpeechT5HifiGan.from_pretrained(local_path + "microsoft/speecht5_hifigan")

inputs = processor(text="Hello, my dog is cute", return_tensors="pt")

# load xvector containing speaker's voice characteristics from a dataset
embeddings_dataset = load_dataset(local_data + "Matthijs/cmu-arctic-xvectors", split="validation")
speaker_embeddings = torch.tensor(embeddings_dataset[7306]["xvector"]).unsqueeze(0)

speech = model.generate_speech(inputs["input_ids"], speaker_embeddings, vocoder=vocoder)

sf.write("speech.wav", speech.numpy(), samplerate=16000)
