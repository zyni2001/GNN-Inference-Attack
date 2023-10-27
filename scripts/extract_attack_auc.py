import os
import re
import csv

# Directory where your log files are located
log_directory = '../temp_data/log'

# Output CSV file
output_csv = './output.csv'

# Regular expression to match the attack auc
regex = r'attack auc \[([0-9.]+),'

# Open the CSV file for writing
with open(output_csv, 'w') as csv_file:
    writer = csv.writer(csv_file)
    # Write the header
    writer.writerow(['dataset_name', 'target_model', 'sample_node_ratio', 'sample_method', 'attack_auc'])

    # Iterate over all log files in the directory
    for filename in os.listdir(log_directory):
        if filename.endswith('.log'):
            # Remove the '.log' extension
            filename = filename[:-4]

            # Find the second last '.' in the filename
            split_index = filename.rfind('.', 0, filename.rfind('.'))

            # Split the filename at the second last '.'
            first_part = filename[:split_index]
            sample_node_ratio_and_method = filename[split_index+1:]

            # The first part of the filename can be split normally
            parts = first_part.split('.')
            dataset_name, target_model = parts

            # Use regular expression to separate sample_node_ratio and sample_method
            match = re.search(r'([0-9.]+)([a-z_]+)', sample_node_ratio_and_method)
            if match:
                sample_node_ratio = match.group(1)
                sample_method = match.group(2)

                # Open the log file
                with open(os.path.join(log_directory, filename + '.log'), 'r') as log_file:
                    # Read the file content
                    content = log_file.read()

                    # Find the attack auc using regex
                    match = re.search(regex, content)

                    if match:
                        attack_auc = match.group(1)
                        # Write the configuration and the attack auc to the CSV file
                        writer.writerow([dataset_name, target_model, sample_node_ratio, sample_method, attack_auc])
