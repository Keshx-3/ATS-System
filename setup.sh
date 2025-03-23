#!/bin/bash
echo "Starting setup.sh" > /tmp/setup_log.txt
python -m spacy download en_core_web_sm 2>> /tmp/setup_log.txt
if [ $? -eq 0 ]; then
    echo "Successfully installed en_core_web_sm" >> /tmp/setup_log.txt
else
    echo "Failed to install en_core_web_sm" >> /tmp/setup_log.txt
fi
echo "Finished setup.sh" >> /tmp/setup_log.txt
