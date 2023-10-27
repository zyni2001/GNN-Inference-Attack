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
            # Parse the filename to get the configuration
            parts = filename.split('.')
            dataset_name = parts[0]
            target_model = parts[1]
            sample_node_ratio_and_method = parts[2]

            # Use regular expression to separate sample_node_ratio and sample_method
            match = re.search(r'([0-9.]+)([a-z_]+)', sample_node_ratio_and_method)
            if match:
                sample_node_ratio = match.group(1)
                sample_method = match.group(2)

                # Open the log file
                with open(os.path.join(log_directory, filename), 'r') as log_file:
                    # Read the file content
                    content = log_file.read()

                    # Find the attack auc using regex
                    match = re.search(regex, content)

                    if match:
                        attack_auc = match.group(1)
                        # Write the configuration and the attack auc to the CSV file
                        writer.writerow([dataset_name, target_model, sample_node_ratio, sample_method, attack_auc])
