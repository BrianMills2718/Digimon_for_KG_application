import os
import json
import argparse
from pathlib import Path

def create_corpus_from_text_files(input_dir: Path, output_dir: Path):
    """
    Reads all .txt files from an input directory, processes them into a corpus format,
    and writes the output to Corpus.json in the output directory.

    Each document in the corpus will have a 'title' (derived from filename),
    'content' (text content of the file), and a unique 'doc_id'.
    """
    corpus_data = []
    doc_id_counter = 0

    if not input_dir.is_dir():
        print(f"Error: Input directory '{input_dir}' not found or is not a directory.")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    output_file_path = output_dir / "Corpus.json"

    print(f"Reading text files from: {input_dir}")
    # Iterate over .txt files in the input directory
    for filepath in sorted(input_dir.glob("*.txt")): # Sort for consistent doc_ids
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Use filename (without extension) as title
            title = filepath.stem 
            
            corpus_entry = {
                "title": title,
                "content": content,
                "doc_id": doc_id_counter
            }
            corpus_data.append(corpus_entry)
            doc_id_counter += 1
            print(f"Processed: {filepath.name} (doc_id: {doc_id_counter-1})")

        except Exception as e:
            print(f"Error processing file {filepath.name}: {e}")

    if not corpus_data:
        print("No .txt files found or processed in the input directory.")
        return

    # Write to Corpus.json in JSON Lines format
    try:
        with open(output_file_path, 'w', encoding='utf-8') as outfile:
            for entry in corpus_data:
                json.dump(entry, outfile)
                outfile.write('\n')
        print(f"\nSuccessfully created Corpus.json with {len(corpus_data)} documents at: {output_file_path}")
    except Exception as e:
        print(f"Error writing Corpus.json: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create Corpus.json from a directory of text files.")
    parser.add_argument(
        "--input_dir", 
        type=str, 
        required=True, 
        help="Directory containing the .txt files to process."
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        required=True, 
        help="Directory where Corpus.json will be saved."
    )

    args = parser.parse_args()

    input_directory = Path(args.input_dir)
    output_directory = Path(args.output_dir)

    create_corpus_from_text_files(input_directory, output_directory)

    # Example Usage from your project root:
    # python scripts/prepare_corpus.py --input_dir ./Data/MyRawTextDataset/ --output_dir ./Data/MyRawTextDataset/
    # This would look for .txt files in ./Data/MyRawTextDataset/ and create ./Data/MyRawTextDataset/Corpus.json
    #
    # To test with your HotpotQAsmallest (assuming it has raw text files, which it might not directly):
    # If you had raw files for HotpotQAsmallest in, say, ./Data/HotpotQAsmallest_raw_text/
    # python scripts/prepare_corpus.py --input_dir ./Data/HotpotQAsmallest_raw_text/ --output_dir ./Data/HotpotQAsmallest/
    # This would create/overwrite ./Data/HotpotQAsmallest/Corpus.json
