#!/bin/bash

# Run the first program
python3 folds_whisper_ko.py --repo_name 'whisper-small-voice-conversion-korean' --fold_num 4

# Run the second program
python3 folds_whisper_ko.py --repo_name 'whisper-small-voice-conversion-korean-20min' --fold_num 3

# Run the third program
python3 folds_whisper_ko.py --repo_name 'whisper-small-voice-conversion-korean-10min' --fold_num 2

echo "All programs have finished running."