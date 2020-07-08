from encoder.params_model import model_embedding_size as speaker_embedding_size
from utils.argutils import print_args
from synthesizer.inference import Synthesizer
from encoder import inference as encoder
from vocoder import inference as vocoder
from pathlib import Path
import numpy as np
import soundfile as sf
import librosa
import argparse
import torch
import sys
from flask import Flask
from flask import request

@app.route('/uploadfile1', methods=['POST'])
def uploadfile1():
    if request.method == 'POST':
        f = request.files['the_file']
        f.save('i.wav')
        return render_template('success')

@app.route('/downloadfile', methods=['POST'])
def downloadfile():
    if request.method == 'POST':
        s = request.string['the_stirng']
        parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
        parser.add_argument("-e", "--enc_model_fpath", type=Path, 
                            default="encoder/saved_models/pretrained.pt",
                            help="Path to a saved encoder")
        parser.add_argument("-s", "--syn_model_dir", type=Path, 
                            default="synthesizer/saved_models/logs-pretrained/",
                            help="Directory containing the synthesizer model")
        parser.add_argument("-v", "--voc_model_fpath", type=Path, 
                            default="vocoder/saved_models/pretrained/pretrained.pt",
                            help="Path to a saved vocoder")
        parser.add_argument("--low_mem", action="store_true", help=\
            "If True, the memory used by the synthesizer will be freed after each use. Adds large "
            "overhead but allows to save some GPU memory for lower-end GPUs.")
        parser.add_argument("--no_sound", action="store_true", help=\
            "If True, audio won't be played.")
        parser.add_argument("--cpu", help="Use CPU.", action="store_true")
        args = parser.parse_args()
        if not args.no_sound:
            import sounddevice as sd
        if args.cpu:
            print("")
        elif torch.cuda.is_available():
            device_id = torch.cuda.current_device()
            gpu_properties = torch.cuda.get_device_properties(device_id)
        else:
            quit(-1)
        encoder.load_model(args.enc_model_fpath)
        synthesizer = Synthesizer(args.syn_model_dir.joinpath("taco_pretrained"), low_mem=args.low_mem)
        vocoder.load_model(args.voc_model_fpath)
        encoder.embed_utterance(np.zeros(encoder.sampling_rate))
        embed = np.random.rand(speaker_embedding_size)
        embed /= np.linalg.norm(embed)
        embeds = [embed, np.zeros(speaker_embedding_size)]
        texts = ["test 1", "test 2"]
        mels = synthesizer.synthesize_spectrograms(texts, embeds)
        mel = np.concatenate(mels, axis=1)
        no_action = lambda *args: None
        vocoder.infer_waveform(mel, target=200, overlap=50, progress_callback=no_action)
        in_fpath = Path(f.replace("\"", "").replace("\'", ""))
        preprocessed_wav = encoder.preprocess_wav(in_fpath)
        original_wav, sampling_rate = librosa.load(str(in_fpath))
        preprocessed_wav = encoder.preprocess_wav(original_wav, sampling_rate)
        embed = encoder.embed_utterance(preprocessed_wav)
        text = s
        texts = [text]
        embeds = [embed]
        specs = synthesizer.synthesize_spectrograms(texts, embeds)
        spec = specs[0]
        generated_wav = vocoder.infer_waveform(spec)
        generated_wav = np.pad(generated_wav, (0, synthesizer.sample_rate), mode="constant")
        if not args.no_sound:
            sd.stop()
            sd.play(generated_wav, synthesizer.sample_rate)
        filename = "demo_output_%02d.wav" % num_generated
        sf.write(filename, generated_wav.astype(np.float32), synthesizer.sample_rate)
        num_generated += 1
        print("\nSaved output as %s\n\n" % filename)
        return render_template(filename,success);

@app.route('/uploadfile2', methods=['POST'])
def uploadfile2():
    if request.method == 'POST':
        f = request.files['the_file']
        f.save('d.wav')
        return render_template(ifisfake)



if __name__ == '__main__':
    app.run()


