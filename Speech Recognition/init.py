from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq
distil_model_id = "distil-whisper/distil-large-v2"
distil_model_path = Path(distil_model_id.split("/")[-1])
quantized_distil_model_path = Path(f"{distil_model_path}_quantized")
processor = AutoProcessor.from_pretrained(distil_model_id) 