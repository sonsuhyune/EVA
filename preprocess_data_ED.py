
import csv
import json
# Open the raw dataset file (CSV format) for reading
with open("empatheticdialogues/train.csv", "r") as file: ### raw dataset path
    reader = csv.reader(file)
    next(reader)

    # Initialize variables
    last_idx = ""
    conversation = []
    conversations = []
    speakers = []
    dia_id =0
    turn_id = 0
    # Iterate over each line in the CSV file
    for line in reader:

        conversation_idx = line[0]  # Extract the conversation index
        utterance = line[5]  # Extract the utterance (dialogue text)
        utterance = utterance.replace("_comma_", ",")  # Replace placeholder for commas
        speaker = line[4]  # Extract the speaker

        # If the conversation index changes, reset the speaker list
        if last_idx != conversation_idx:
            speakers = []
        # Add the speaker to the list if not already present
        if speaker in speakers:
            pass
        else:
            speakers.append(speaker)
        #         print(speakers)

        # Append the utterance to the conversation
        if last_idx == conversation_idx:
            conversation.append(utterance)
        if last_idx != conversation_idx: # If it's a new conversation, save the previous one
            if last_idx != "":
                conversations.append(conversation)
                conversation = []
            conversation.append(utterance)
            last_idx = conversation_idx
            dia_id += 1
            turn_id = 0
        #         print(line)
        print(conversation)
        print(speakers)
        print(speaker)

        # If there are two speakers and the current speaker is the last one added
        if len(speakers) == 2 and speaker == speakers[-1]:
            turn_id += 1  # Increment turn ID
            conv = []  # Initialize a temporary list for the conversation

            # Iterate over each utterance in the conversation
            for utt_idx, utt in enumerate(conversation):
                # Assign roles based on the index (0 for even, 1 for odd)
                if utt_idx % 2 == 0:
                    role = 0
                else:
                    role = 1

                # Create a dictionary with role and utterance, and add to conv list
                tmp = {"role": role, "utt": utt}
                conv.append(tmp)

            # Save the conversation to a JSONL file with a unique index
            with open("empatheticdialogues/prepro_train_w_idx.jsonl", "a") as f:  # Save path
                json.dump({"idx": f"{dia_id - 1}_{turn_id - 1}", "dialog": conv}, f)
                f.write("\n")  # Write the JSON object and add a newline
#         breakpoint()