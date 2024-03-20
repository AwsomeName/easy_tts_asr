# 该脚本并未验证，只是博客中的代码摘录，作为研读

from datasets import load_dataset, DatasetDict, load_from_disk
from datasets.utils.info_utils import VerificationMode

common_voice = DatasetDict()
# data_path = "/home/lc/data_a/mozilla-foundation/common_voice_11_0/"
data_path = "/home/lc/data_a/mozilla-foundation/common_voice_11_0"
# data_path = "/home/lc/data_a/mozilla-foundation/common_voice_11_0_bk"
# data_path = "/home/lc/data_a/mozilla-foundation/common_voice_11_0_test"

# common_voice["train"] = load_dataset(data_path, "hi", split="train+validation", trust_remote_code=True)
# common_voice["test"] = load_dataset(data_path, "hi", split="test", trust_remote_code=True)
# common_voice["train"] = load_dataset(data_path, "default", split="train+validation", trust_remote_code=True)
# common_voice["test"] = load_dataset(
# load_from_disk(data_path)
# old_data = load_dataset("/home/lc/code/database_papers/arxiv_dataset", data_dir="/home/lc/Arxiv/164/", verification_mode=VerificationMode.NO_CHECKS)
print("-----------------")
common_voice["test"] = load_dataset(data_path, name="hi", data_dir=data_path, verification_mode=VerificationMode.NO_CHECKS)
# exit()
# common_voice["test"] = load_dataset(
#     "audiofolder", 
#     # "webdataset",
#     data_dir=data_path,
#     name="as",
#     split="test", 
#     num_proc=20, 
#     trust_remote_code=True,
#     verification_mode=VerificationMode.NO_CHECKS)

# common_voice['train'] = load_dataset(
#     "audiofolder", 
#     data_dir=data_path,
#     name="as",
#     split="train", 
#     num_proc=20, 
#     trust_remote_code=True,
# )
    # download_mode="reuse_dataset_if_exists")
# common_voice["train"] = load_dataset(data_path, "default", split="validation", trust_remote_code=True)

print(common_voice)
exit()
"""DatasetDict({
    train: Dataset({
        features: ['client_id', 'path', 'audio', 'sentence', 'up_votes', 'down_votes', 'age', 'gender', 'accent', 'locale', 'segment'],
        num_rows: 6540
    })
    test: Dataset({
        features: ['client_id', 'path', 'audio', 'sentence', 'up_votes', 'down_votes', 'age', 'gender', 'accent', 'locale', 'segment'],
        num_rows: 2894
    })
})
"""

# common_voice = common_voice.remove_columns(["accent", "age", "client_id", "down_votes", "gender", "locale", "path", "segment", "up_votes"])

# 特征提取器
from transformers import WhisperFeatureExtractor

model_path = "/home/lc/models/openai/whisper-small"

# feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-small")
feature_extractor = WhisperFeatureExtractor.from_pretrained(model_path)

# tokenizer
from transformers import WhisperTokenizer

tokenizer = WhisperTokenizer.from_pretrained(model_path, language="Hindi", task="transcribe")



input_str = common_voice["train"][0]["sentence"]
labels = tokenizer(input_str).input_ids
decoded_with_special = tokenizer.decode(labels, skip_special_tokens=False)
decoded_str = tokenizer.decode(labels, skip_special_tokens=True)

print(f"Input: {input_str}")
print(f"Decoded w/ special: {decoded_with_special}")
print(f"Decoded w/out special: {decoded_str}")
print(f"Are equal: {input_str == decoded_str}")


# 这个processor封装了以上两者
from transformers import WhisperProcessor

processor = WhisperProcessor.from_pretrained(model_path, language="Hindi", task="transcribe")

# 打印一个数据样本
print(common_voice["train"][0])
"""{'audio': {'path': '/home/sanchit_huggingface_co/.cache/huggingface/datasets/downloads/extracted/607848c7e74a89a3b5225c0fa5ffb9470e39b7f11112db614962076a847f3abf/cv-corpus-11.0-2022-09-21/hi/clips/common_voice_hi_25998259.mp3',
           'array': array([0.0000000e+00, 0.0000000e+00, 0.0000000e+00, ..., 9.6724887e-07,
       1.5334779e-06, 1.0415988e-06], dtype=float32),
           'sampling_rate': 48000},
 'sentence': 'खीर की मिठास पर गरमाई बिहार की सियासत, कुशवाहा ने दी सफाई'}
"""

# 由于现在输入音频的采样率为 48kHz，所以在将其馈送给 Whisper 特征提取器之前，我们需要将其 _下采样_至 16kHz。

# 我们将使用 dataset 的 cast_column 方法将输入音频转换至所需的采样率。该方法仅指示 datasets 让其在首次加载音频时 _即时地_对数据进行重采样，因此并不会改变原音频数据:
    
from datasets import Audio

common_voice = common_voice.cast_column("audio", Audio(sampling_rate=16000))

print(common_voice["train"][0])


def prepare_dataset(batch):
    # load and resample audio data from 48 to 16kHz
    audio = batch["audio"]

    # compute log-Mel input features from input audio array
    batch["input_features"] = feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]

    # encode target text to label ids
    batch["labels"] = tokenizer(batch["sentence"]).input_ids
    return batch

common_voice = common_voice.map(prepare_dataset, remove_columns=common_voice.column_names["train"], num_proc=4)
print("prepare_data done ...")

import torch

from dataclasses import dataclass
from typing import Any, Dict, List, Union

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need different padding methods
        # first treat the audio inputs by simply returning torch tensors
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        # get the tokenized label sequences
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        # pad the labels to max length
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        # if bos token is appended in previous tokenization step,
        # cut bos token here as it's append later anyways
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels

        return batch


data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

print("data_collator done")

import evaluate

metric = evaluate.load("wer")

def compute_metrics(pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    # replace -100 with the pad_token_id
    label_ids[label_ids == -100] = tokenizer.pad_token_id

    # we do not want to group tokens when computing the metrics
    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    wer = 100 * metric.compute(predictions=pred_str, references=label_str)

    return {"wer": wer}

from transformers import WhisperForConditionalGeneration

model = WhisperForConditionalGeneration.from_pretrained(model_path)

model.config.forced_decoder_ids = None
model.config.suppress_tokens = []

from transformers import Seq2SeqTrainingArguments

training_args = Seq2SeqTrainingArguments(
    output_dir="./whisper-small-hi", # change to a repo name of your choice
    per_device_train_batch_size=16,
    gradient_accumulation_steps=1, # increase by 2x for every 2x decrease in batch size
    learning_rate=1e-5,
    warmup_steps=500,
    max_steps=4000,
    gradient_checkpointing=True,
    fp16=True,
    evaluation_strategy="steps",
    per_device_eval_batch_size=8,
    predict_with_generate=True,
    generation_max_length=225,
    save_steps=1000,
    eval_steps=1000,
    logging_steps=25,
    report_to=["tensorboard"],
    load_best_model_at_end=True,
    metric_for_best_model="wer",
    greater_is_better=False,
    # push_to_hub=True,
)

from transformers import Seq2SeqTrainer
print("trainer init done ...")
trainer = Seq2SeqTrainer(
    args=training_args,
    model=model,
    train_dataset=common_voice["train"],
    eval_dataset=common_voice["test"],
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    tokenizer=processor.feature_extractor,
)

trainer.train()







