#!/bin/bash

FOLDER="/media/clic/tts/corpus_ubterm"
INPUTS="${FOLDER}/dades.csv"
OUTPUTS="${FOLDER}/corpus_sintetic"

python3 synthetic_speech.py --texts_file $INPUTS --output_dir $OUTPUTS
# python3 synthetic_speech.py --texts_file prova.txt --output_dir $OUTPUTS

echo "Converting: wav to 16KHz mp3"
for wav_file in "$OUTPUTS"/*.wav; do
    [ ! -f "$wav_file" ] && continue
    mp3_file="${wav_file%.wav}.mp3"
    ffmpeg -i "$wav_file" -ar 16000 -b:a 128k -y "$mp3_file" -hide_banner -loglevel error && rm "$wav_file"
done
echo "Done!"
