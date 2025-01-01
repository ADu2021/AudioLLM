#!/bin/bash

# Loop through all files matching the pattern "audio-stream (*).mp3"
for i in {1..21}; do
  # Check if the file exists
  if [ -f "audio-stream ($i).mp3" ]; then
    # Rename the file to replace spaces and parentheses with dashes
    mv "audio-stream ($i).mp3" "audio-stream-$i.mp3"
  fi
done

echo "File renaming completed."