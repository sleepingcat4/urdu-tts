import torch
from pyannote.audio import Pipeline
from pyannote.audio.pipelines.utils.hook import ProgressHook
import os
import json
from pydub import AudioSegment
from tqdm import tqdm
import whisper

whisper_model = whisper.load_model("medium")

def transcribe_audio_local(audio_path):
    result = whisper_model.transcribe(audio_path)
    return result["text"]

def diarized_audio(data_jsonl):
    output_dir = "data"
    os.makedirs(output_dir, exist_ok=True)
    pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-community-1")
    pipeline.to(torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))

    jsonl_path = os.path.join(output_dir, "train_data.jsonl")

    with open(data_jsonl, "r", encoding="utf-8") as f:
        total_lines = sum(1 for _ in f)

    with open(data_jsonl, "r", encoding="utf-8") as f_in, \
         open(jsonl_path, "w", encoding="utf-8") as f_out, \
         tqdm(total=total_lines, desc="Processing audio") as pbar:

        for line in f_in:
            item = json.loads(line)
            audio_path = item["audio_path"]
            audio = AudioSegment.from_file(audio_path)

            with ProgressHook() as hook:
                output = pipeline(audio_path, hook=hook)

            speaker_map = {}
            next_speaker_id = 0

            for i, (turn, speaker_label) in enumerate(output.speaker_diarization):
                if speaker_label not in speaker_map:
                    speaker_map[speaker_label] = next_speaker_id
                    next_speaker_id += 1
                speaker_id = speaker_map[speaker_label]

                start_ms = int(turn.start * 1000)
                end_ms = int(turn.end * 1000)
                segment_audio = audio[start_ms:end_ms]
                segment_filename = f"{os.path.splitext(os.path.basename(audio_path))[0]}_seg{i:03d}.wav"
                segment_path = os.path.join(output_dir, segment_filename)
                segment_audio.export(segment_path, format="wav")

                transcript_text = transcribe_audio_local(segment_path)

                f_out.write(json.dumps({
                    "text": f"Speaker {speaker_id}: {transcript_text}",
                    "audio": segment_path
                }, ensure_ascii=False) + "\n")

            pbar.update(1)
