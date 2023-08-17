from transformers import SpeechT5Processor, SpeechT5ForSpeechToSpeech, SpeechT5HifiGan
from datasets import load_dataset

local_path = "/home/lc/models/"
# local_path = ""
local_data = "/home/lc/hf_dataset/"
# local_data = ""

dataset = load_dataset(local_data + "hf-internal-testing/librispeech_asr_demo", "clean", split="validation")
dataset = dataset.sort("id")
sampling_rate = dataset.features["audio"].sampling_rate
example_speech = dataset[0]["audio"]["array"]

processor = SpeechT5Processor.from_pretrained(local_path + "microsoft/speecht5_vc")
model = SpeechT5ForSpeechToSpeech.from_pretrained(local_path + "microsoft/speecht5_vc")
vocoder = SpeechT5HifiGan.from_pretrained(local_path + "microsoft/speecht5_hifigan")

inputs = processor(audio=example_speech, sampling_rate=sampling_rate, return_tensors="pt")

# load xvector containing speaker's voice characteristics from a file
import numpy as np
import torch
# speaker_embeddings = np.load("xvector_speaker_embedding.npy")
speaker_embeddings = np.load("/home/lc/hf_dataset/Matthijs/cmu-arctic-xvectors/spkrec-xvect/cmu_us_rms_arctic-wav-arctic_b0088.npy")
speaker_embeddings = torch.tensor(speaker_embeddings).unsqueeze(0)
# print("emb shape right:", speaker_embeddings.shape)
# speaker_embeddings = np.load("/home/lc/code/easy_tts_asr/Speaker_emb/test_data/spkrec-xvect/Speaker_emb-test_data-liuchang-16k.npy")
# speaker_embeddings = torch.tensor(speaker_embeddings).unsqueeze(0)
# print("emb shape test:", speaker_embeddings.shape)

# embeddings_dataset = load_dataset(local_data + "Matthijs/cmu-arctic-xvectors", split="validation")
# speaker_embeddings = torch.tensor(embeddings_dataset[7306]["xvector"]).unsqueeze(0)

speech = model.generate_speech(inputs["input_values"], speaker_embeddings, vocoder=vocoder)

import soundfile as sf
sf.write("speech.wav", speech.numpy(), samplerate=16000)
