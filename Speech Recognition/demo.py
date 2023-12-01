#DEMO
from transformers.pipelines.audio_utils import ffmpeg_read
import gradio as gr
import urllib.request

urllib.request.urlretrieve(
    url="https://huggingface.co/spaces/distil-whisper/whisper-vs-distil-whisper/resolve/main/assets/example_1.wav",
    filename="example_1.wav",
)

BATCH_SIZE = 16
MAX_AUDIO_MINS = 30  # maximum audio input in minutes


ov_pipe = pipeline(
    "automatic-speech-recognition",
    model=ov_distil_model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    max_new_tokens=128,
    chunk_length_s=15,
    generate_kwargs={"language": "en", "task": "transcribe"},
)
ov_pipe_forward = ov_pipe._forward

if to_quantize.value:
    ov_quantized_distil_model.generation_config = ov_distil_model.generation_config
    ov_quantized_pipe = pipeline(
        "automatic-speech-recognition",
        model=ov_quantized_distil_model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        max_new_tokens=128,
        chunk_length_s=15,
        generate_kwargs={"language": "en", "task": "transcribe"},
    )
    ov_quantized_pipe_forward = ov_quantized_pipe._forward


def transcribe(inputs, quantized=False):
    pipe = ov_quantized_pipe if quantized else ov_pipe
    pipe_forward = ov_quantized_pipe_forward if quantized else ov_pipe_forward

    if inputs is None:
        raise gr.Error(
            "No audio file submitted! Please record or upload an audio file before submitting your request."
        )

    with open(inputs, "rb") as f:
        inputs = f.read()

    inputs = ffmpeg_read(inputs, pipe.feature_extractor.sampling_rate)
    audio_length_mins = len(inputs) / pipe.feature_extractor.sampling_rate / 60

    if audio_length_mins > MAX_AUDIO_MINS:
        raise gr.Error(
            f"To ensure fair usage of the Space, the maximum audio length permitted is {MAX_AUDIO_MINS} minutes."
            f"Got an audio of length {round(audio_length_mins, 3)} minutes."
        )

    inputs = {"array": inputs, "sampling_rate": pipe.feature_extractor.sampling_rate}

    def _forward_ov_time(*args, **kwargs):
        global ov_time
        start_time = time.time()
        result = pipe_forward(*args, **kwargs)
        ov_time = time.time() - start_time
        ov_time = round(ov_time, 2)
        return result

    pipe._forward = _forward_ov_time
    ov_text = pipe(inputs.copy(), batch_size=BATCH_SIZE)["text"]
    return ov_text, ov_time


with gr.Blocks() as demo:
    gr.HTML(
        """
                <div style="text-align: center; max-width: 700px; margin: 0 auto;">
                  <div
                    style="
                      display: inline-flex; align-items: center; gap: 0.8rem; font-size: 1.75rem;
                    "
                  >
                    <h1 style="font-weight: 900; margin-bottom: 7px; line-height: normal;">
                      OpenVINO Distil-Whisper demo
                    </h1>
                  </div>
                </div>
            """
    )
    audio = gr.components.Audio(type="filepath", label="Audio input")
    with gr.Row():
        button = gr.Button("Transcribe")
        if to_quantize.value:
            button_q = gr.Button("Transcribe quantized")
    with gr.Row():
        infer_time = gr.components.Textbox(
            label="OpenVINO Distil-Whisper Transcription Time (s)"
        )
        if to_quantize.value:
            infer_time_q = gr.components.Textbox(
                label="OpenVINO Quantized Distil-Whisper Transcription Time (s)"
            )
    with gr.Row():
        transcription = gr.components.Textbox(
            label="OpenVINO Distil-Whisper Transcription", show_copy_button=True
        )
        if to_quantize.value:
            transcription_q = gr.components.Textbox(
                label="OpenVINO Quantized Distil-Whisper Transcription", show_copy_button=True
            )
    button.click(
        fn=transcribe,
        inputs=audio,
        outputs=[transcription, infer_time],
    )
    if to_quantize.value:
        button_q.click(
            fn=transcribe,
            inputs=[audio, gr.Number(value=1, visible=False)],
            outputs=[transcription_q, infer_time_q],
        )
    gr.Markdown("## Examples")
    gr.Examples(
        [["./example_1.wav"]],
        audio,
        outputs=[transcription, infer_time],
        fn=transcribe,
        cache_examples=False,
    )
# if you are launching remotely, specify server_name and server_port
# demo.launch(server_name='your server name', server_port='server port in int')
# Read more in the docs: https://gradio.app/docs/
try:
    demo.launch(debug=True)
except Exception:
    demo.launch(share=True, debug=True)