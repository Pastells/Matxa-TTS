#!/bin/bash

FOLDER="/media/clic/tts/corpus_ubterm/bio"
INPUTS="${FOLDER}/sentences/words/biologia_celular.csv"
# INPUTS="${FOLDER}/plans_docents.csv"
OUTPUTS="${FOLDER}/sentences/tts/biologia_celular/second_sentence"
# OUTPUTS="${FOLDER}/plans_docents/tts"
BATCH_SIZE=32
# BATCH_SIZE=8
SPEED=0.9

python3 synthetic_speech.py --texts_file $INPUTS --output_dir $OUTPUTS --length_scale $SPEED --batch_size $BATCH_SIZE

echo "Converting: wav to 16KHz mp3"
parallel -j 8 'ffmpeg -i "{}" -ar 16000 -b:a 128k -y "{.}.mp3" -hide_banner -loglevel error' ::: "$OUTPUTS"/*.wav
rm "$OUTPUTS"/*.wav
echo "Done!"
