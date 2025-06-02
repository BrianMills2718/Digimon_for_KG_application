#!/bin/bash
# Test script for DIGIMON CLI interactive mode

# Create a simple input file for testing
echo -e "What is machine learning?\nWhat is NLP?\nexit" > test_input.txt

# Source conda and run the CLI in interactive mode with input
source ~/miniconda3/etc/profile.d/conda.sh
conda activate digimon
python digimon_cli.py -c testing/test_documents -i < test_input.txt

# Clean up
rm test_input.txt
