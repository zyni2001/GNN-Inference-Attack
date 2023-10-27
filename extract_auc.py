import os
import re
import csv

# Specify the directory containing the log files
log_dir = './'  # replace this with your log directory

# Initialize a list to store results
results = []

# Regular expression to match the max attack auc in the log files
regex = r"attack auc ([0-9\.]+)"

# Iterate over all subdirectories (configs)
for root, dirs, files in os.walk(log_dir):
    if root.endswith('log') and root.startswith('./temp_data'):
        for file in files:
            if file.endswith('.log'):
                # Extract config name from the root name:./temp_data_DD_0.8_mean_pool_snow_ball/log --> DD_0.8_mean_pool_snow_ball
                config = root.split('/')[1]

                with open(os.path.join(root, file), 'r') as f:
                    content = f.read()
                    matches = re.findall(regex, content)
                    print(matches)
                    # Convert matches to float and get the maximum
                    if matches is not None and len(matches) > 0:
                        max_auc = max(map(float, matches))
                    else:
                        max_auc = 0
                    # Split the config string into components
                    config_parts = config.split('_')

                    # Assign the first few parts to individual variables
                    temp, data, dataset, sample_ratio = config_parts[:4]

                    # model
                    model = '_'.join(config_parts[4:6])

                    # sample method
                    method = '_'.join(config_parts[6:])

                    results.append([dataset, sample_ratio, model, method, max_auc])

# Save the results to a CSV file
with open('new_result.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['dataset', 'sample_ratio', 'model', 'method', 'max_auc'])  # writing header
    writer.writerows(results)  # writing data

