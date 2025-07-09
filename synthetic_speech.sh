#!/bin/bash

FOLDER="/media/clic/tts/corpus_ubterm/bio"
INPUTS="${FOLDER}/sentences/words/biologia_celular.csv"
OUTPUTS="${FOLDER}/sentences/tts/biologia_celular"
SPEED=0.9

python3 synthetic_speech.py --texts_file $INPUTS --output_dir $OUTPUTS --length_scale $SPEED

echo "Converting: wav to 16KHz mp3"
parallel -j 8 'ffmpeg -i "{}" -ar 16000 -b:a 128k -y "{.}.mp3" -hide_banner -loglevel error' ::: "$OUTPUTS"/*.wav
rm "$OUTPUTS"/*.wav
echo "Done!"
