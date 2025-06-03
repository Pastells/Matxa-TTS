#!/bin/bash

OUTPUTS="outputs"
python3 synthetic_speech.py --texts_file frases.txt --output_dir $OUTPUTS

for wav_file in "$OUTPUTS"/*.wav; do
    [ ! -f "$wav_file" ] && continue
    mp3_file="${wav_file%.wav}.mp3"
    echo "Converting: $(basename "$wav_file")"
    ffmpeg -i "$wav_file" -ar 16000 -b:a 128k -y "$mp3_file" -v quiet && rm "$wav_file"
done
echo "Done!"
