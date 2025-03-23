#!/bin/bash
echo "Starting setup.sh" > /tmp/setup_log.txt
python -V >> /tmp/setup_log.txt 2>&1  # Log Python version
pip list | grep spacy >> /tmp/setup_log.txt 2>&1  # Confirm spaCy is installed
python -m spacy download en_core_web_sm >> /tmp/setup_log.txt 2>&1
if [ $? -eq 0 ]; then
    echo "Successfully installed en_core_web_sm" >> /tmp/setup_log.txt
else
    echo "Failed to install en_core_web_sm with exit code $?" >> /tmp/setup_log.txt
fi
echo "Finished setup.sh" >> /tmp/setup_log.txt
