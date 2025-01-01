from llama.tokenizer import Tokenizer
import pickle
import os
import argparse
from tqdm import tqdm

def clean(input_folder, output_folder, tokenizer):
    """
    For each .pkl file in input_folder, load with pickle as data;
    Then for each data['text'], which is a list, check the last token of each element, decode with tokenizer;
    If the decoded text ends with \n, then replace the token with the one encoded without \n.
    Make sure that we are replacing exactly one token to exactly another token.
    """

    # Ensure the output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Iterate over all .pkl files in the input folder
    for filename in os.listdir(input_folder):
        if filename.endswith('.pkl'):
            input_filepath = os.path.join(input_folder, filename)
            output_filepath = os.path.join(output_folder, filename)

            # Load the data from the pickle file
            with open(input_filepath, 'rb') as f:
                data = pickle.load(f)

            # Get the list of texts
            text_list = data['text']

            # Process each text in the list
            new_text_list = []
            for text in tqdm(text_list):
                # Get the last token
                last_token = text[-1]

                # Decode the last token
                decoded_last_token = tokenizer.decode([last_token])

                # Check if the decoded last token ends with a newline character
                if decoded_last_token.endswith('\n'):
                    # Remove the newline character
                    new_decoded_token = decoded_last_token.rstrip('\n')

                    # Encode the new token without the newline character
                    new_token_ids = tokenizer.encode(new_decoded_token, bos=False, eos=False)

                    # Ensure we replace exactly one token with one token
                    if len(new_token_ids) == 1:
                        # Replace the last token
                        text[-1] = new_token_ids[0]
                        new_text = text
                    else:
                        # If we can't replace with exactly one token, keep the original text
                        new_text = text
                else:
                    # No newline at the end, keep the original text
                    new_text = text

                import ipdb; ipdb.set_trace();

                # Append the processed text to the new list
                new_text_list.append(new_text)

            # Update the data
            data['text'] = new_text_list

            # Save the (possibly modified) data to the output folder
            # with open(output_filepath, 'wb') as f:
            #     pickle.dump(data, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clean .pkl files by removing trailing newline characters from tokens.")
    parser.add_argument('input_folder', help='Path to the input folder containing .pkl files')
    parser.add_argument('output_folder', help='Path to the output folder for cleaned .pkl files')
    args = parser.parse_args()

    # Initialize the tokenizer
    DEFAULT_TOKENIZER_URI = "/afs/cs.stanford.edu/u/duyy/data/models/llama3/Meta-Llama-3-8B-Instruct/tokenizer.model"
    tokenizer = Tokenizer(DEFAULT_TOKENIZER_URI)

    # Run the clean function
    clean(args.input_folder, args.output_folder, tokenizer)
