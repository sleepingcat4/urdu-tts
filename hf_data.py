from datasets import load_dataset
import os
import json
import soundfile as sf

def hf_audio(dataset_name):
    dataset = load_dataset(dataset_name)
    audio_dir = os.path.join(os.getcwd(), "audio")
    os.makedirs(audio_dir, exist_ok=True)
    jsonl_path = os.path.join(os.getcwd(), "data.jsonl")

    with open(jsonl_path, "w", encoding="utf-8") as f:
        for item in dataset["train"]:
            filename = item["filename"] + ".wav"
            audio_path = os.path.join(audio_dir, filename)
            sf.write(audio_path, item["audio"]["array"], item["audio"]["sampling_rate"])
            f.write(json.dumps({
                "audiofilename": filename,
                "audio_path": audio_path,
                "text": item["text"]
            }, ensure_ascii=False) + "\n")

hf_audio('muhammadsaadgondal/urdu-tts')
