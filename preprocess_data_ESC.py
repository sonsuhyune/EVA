import os
import json
import argparse

from collections import defaultdict

# Convert a dialogue into turns where each turn contains a set of utterances by the same speaker
def convert_to_turns(dialog):
    flattened_dialog = []  # List to hold the converted dialogue
    turn = defaultdict(list)  # Dictionary to hold the current turn

    prev_speaker = None
    for utterance in dialog:
        cur_speaker = utterance['speaker']  # Current speaker

        # If the current speaker is 'sys' and the previous was 'usr', a turn is complete
        if cur_speaker == 'sys' and prev_speaker == 'usr':
            flattened_dialog.append(turn)  # Add the completed turn to the list
            turn = defaultdict(list)  # Start a new turn

        turn[cur_speaker].append(utterance['text'])  # Add the utterance text to the current turn
        prev_speaker = cur_speaker  # Update the previous speaker

    flattened_dialog.append(turn)  # Add the last turn
    return flattened_dialog

# Parse the input dialogue into a specific format with role names replaced
def parse_input(raw_context):
    mapper = {'sys': 'supporter', 'usr': 'help_seeker'}
    input_text = ''

    if type(raw_context) != list:  # If not a list, iterate through the keys
        for key in raw_context:
            input_text += mapper[key] + ": " + " ".join(raw_context[key])
    else:  # If it's a list, iterate through each context
        for context in raw_context:
            for key in context:
                input_text += mapper[key] + ": " + " ".join(context[key])
                input_text += ' '

    return input_text.strip()  # Return the parsed input as a single string

# Generate a windowed dataset based on a given window size
def gen_windowed_dataset(flattened_dialog, window_size):
    windowed_dialogs = {'inputs': [], 'outputs': []}

    for i in range(len(flattened_dialog) - window_size + 1):
        window = flattened_dialog[i:i + window_size]
        inputs = parse_input(window[:window_size - 1])  # Prepare inputs by parsing the window
        output = ".".join(window[-1]['sys'])  # Join the system outputs with a period
        windowed_dialogs['inputs'].append(inputs)
        windowed_dialogs['outputs'].append(output)

    return windowed_dialogs

# Main script execution
if __name__ == '__main__':
    test_dataset = []

    # Open and read the original ESConv dataset file
    with open(os.path.join("raw_data", "test.txt"), "r") as f:  # original ESConv dataset path
        dia_id = 0  # Dialogue ID

        for line in f:
            

            raw_data = json.loads(line)  # Load the JSON data from the line
            raw_dialog = raw_data['dialog']  # Extract the dialogue part
            flattened_dialog = convert_to_turns(raw_dialog)  # Convert the dialogue into turns

            flag = False
            result = []
            inputs = []

            turn_id = 0  # Turn ID
            for turn in flattened_dialog:
                with open("ESC_dataset/prepro_test_w_idx_ver2.jsonl", "a") as f:  # Save path
                    for utt in turn.items():  # Iterate through the items in the turn
                        tmp = {}
                        speaker, text = utt

                        if speaker == "sys":  # System role
                            tmp['role'] = 1
                            tmp['utt'] = text[0]
                        elif speaker == "usr":  # User role
                            tmp['role'] = 0
                            tmp['utt'] = text[0]

                        inputs.append(tmp)  # Append the turn to inputs

                        if not flag:  # If flag is False, set it to True (start of a dialogue)
                            flag = True
                        else:
                            if speaker == "sys":  # Save the turn only if the speaker is 'sys'
                                save = {"idx": f"{dia_id}_{turn_id}", "dialog": inputs}
                                json.dump(save, f)  # Dump the JSON object to the file
                                f.write("\n")
                                turn_id += 1  # Increment the turn ID

            dia_id += 1  # Increment the dialogue ID
        print(dia_id)
        print(turn_id)