import os
import glob
import numpy
import argparse
import torchaudio
from speechbrain.pretrained import EncoderClassifier
import torch
from tqdm import tqdm
import torch.nn.functional as F

spk_model = {
    "speechbrain/spkrec-xvect-voxceleb": 512, 
    "speechbrain/spkrec-ecapa-voxceleb": 192,
}

def f2embed(wav_file, classifier, size_embed):
    signal, fs = torchaudio.load(wav_file)
    assert fs == 16000, fs
    with torch.no_grad():
        embeddings = classifier.encode_batch(signal)
        embeddings = F.normalize(embeddings, dim=2)
        embeddings = embeddings.squeeze().cpu().numpy()
        
    print(embeddings)
    print("----")
    print(embeddings[0])
    # raw_emb = embeddings
    new_emb = embeddings[0]
    print("---check:", embeddings.shape, "---:",  size_embed, "----", embeddings.shape[0])
    # assert embeddings.shape[0] == size_embed, embeddings.shape[0]
    print("new_emb:", new_emb.shape)
    assert new_emb.shape[0] == size_embed, new_emb.shape[0]
    # return embeddings
    return new_emb

def process(args):
    wavlst = []
    # for split in args.splits.split(","):
    #     wav_dir = os.path.join(args.arctic_root, split)
    #     print("wav_dir:", wav_dir)
    #     wavlst_split = glob.glob(os.path.join(wav_dir, "wav", "*.wav"))
    #     wavlst_split = ["/home/lc/code/easy_tts_asr/Speaker_emb/test_data/liuchang.wav"]
    #     print("wavlist_split:", wavlst_split)
    #     print(f"{split} {len(wavlst_split)} utterances.")
    #     wavlst.extend(wavlst_split)
    wavlst = ["/home/lc/code/easy_tts_asr/Speaker_emb/test_data/liuchang-16k.wav"]

    spkemb_root = args.output_root
    if not os.path.exists(spkemb_root):
        print(f"Create speaker embedding directory: {spkemb_root}")
        os.mkdir(spkemb_root)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    classifier = EncoderClassifier.from_hparams(source=args.speaker_embed, run_opts={"device": device}, savedir=os.path.join('/tmp', args.speaker_embed))
    size_embed = spk_model[args.speaker_embed]
    for utt_i in tqdm(wavlst, total=len(wavlst), desc="Extract"):
        # TODO rename speaker embedding
        utt_id = "-".join(utt_i.split("/")[-3:]).replace(".wav", "")
        utt_emb = f2embed(utt_i, classifier, size_embed)
        numpy.save(os.path.join(spkemb_root, f"{utt_id}.npy"), utt_emb)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arctic-root", "-i", required=True, type=str, help="LibriTTS root directory.")
    parser.add_argument("--output-root", "-o", required=True, type=str, help="Output directory.")
    parser.add_argument("--speaker-embed", "-s", type=str, required=True, choices=["speechbrain/spkrec-xvect-voxceleb", "speechbrain/spkrec-ecapa-voxceleb"],
                        help="Pretrained model for extracting speaker emebdding.")
    parser.add_argument("--splits",  type=str, help="Split of four speakers seperate by comma.", default="liuchang")
                        # default="cmu_us_bdl_arctic,cmu_us_clb_arctic,cmu_us_rms_arctic,cmu_us_slt_arctic")
    args = parser.parse_args()
    print(f"Loading utterances from {args.arctic_root}/{args.splits}, "
        + f"Save speaker embedding 'npy' to {args.output_root}, "
        + f"Using speaker model {args.speaker_embed} with {spk_model[args.speaker_embed]} size.")
    process(args)

if __name__ == "__main__":
    """
    python utils/prep_cmu_arctic_spkemb.py \
        -i ./test_data \
        -o ./test_data/spkrec-xvect \
        -s speechbrain/spkrec-xvect-voxceleb
    """
    main()