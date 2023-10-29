import matplotlib.pyplot as plt
import re
import os
import pickle

if __name__ == '__main__':
    """
    This script is used to plot the inference attack accuracy vs graph sparsity 
    and the inductive UGS classification accuracy vs graph sparsity.
    """

    input_directory = '/home/zhiyu/GNN-Embedding-Leaks-DD/temp_data_DD_0.2_diff_pool_snow_ball/embed_log_correct_setting/'
    if os.path.exists(input_directory) is False:
        raise Exception('input_directory does not exist')
        exit()

    final_test_acc_list = []
    graph_sparsity = pickle.load(open('/home/zhiyu/GNN-Embedding-Leaks-DD/acc-graph_sparsity-pkl/graph_sparsity_DD_diffpool.pkl', 'rb'))
    # save no decimal point of graph_sparsity
    graph_sparsity = [int(i) for i in graph_sparsity]
    final_test_acc_list_DD = pickle.load(open('/home/zhiyu/GNN-Embedding-Leaks-DD/acc-graph_sparsity-pkl/final_test_acc_list_DD_diffpool.pkl', 'rb'))

    fig, ax = plt.subplots(figsize=(10, 6))
    # Lists to accumulate means and stds
    means = []
    stds = []

    # Assuming the number of sparsity levels is the maximum filename index
    num_sparsity_levels = len(graph_sparsity)
    root = input_directory

    # extract the number of repeats from input_directory: the max number using '_' in filename
    repeats = 0
    for files in os.listdir(input_directory):
        if files.endswith(".log"):
            repeats = max(repeats, int(files.split('_')[2].split('.')[0]))
    repeats = repeats + 1

    for sparsity_index in range(0, num_sparsity_levels):
        accuracies_for_sparsity = []

        for repeat in range(0, repeats):  # Assuming there are 4 repeats for each level
            filename = f"{sparsity_index}_repeat_{repeat}.log"
            file_path = os.path.join(root, filename)
            if os.path.exists(file_path) is False:
                continue
            with open(file_path, 'r') as infile:
                for line in infile:
                    final_test_acc_match = re.search(r'element_l2, max attack acc: (\d+\.\d+)', line)
                    if final_test_acc_match:
                        final_test_acc = float(final_test_acc_match.group(1))*100
                        accuracies_for_sparsity.append(final_test_acc)
        
        # Compute the mean and std for this sparsity level
        mean_acc = sum(accuracies_for_sparsity) / len(accuracies_for_sparsity)
        std_acc = (sum([(x - mean_acc)**2 for x in accuracies_for_sparsity]) / len(accuracies_for_sparsity))**0.5

        means.append(mean_acc)
        stds.append(std_acc)

    # Use errorbar to plot the means with std as error bars
    # ax.errorbar(graph_sparsity, means, yerr=stds, label='Mean Inference Attack Accuracy after pruning', fmt='-o')
    # Plot the mean curve
    ax.plot(graph_sparsity, means, '-', label='Inference Attack Accuracy', color='blue', linewidth=4)

    # baseline = 0.826114473*100
    baseline = 0.83*100
    # Plot a horizontal line
    ax.axhline(y=baseline, color='k', linestyle='--', linewidth=4, label='Inference Attack Accuracy (Baseline)')

    # Shade the error area
    ax.fill_between(graph_sparsity, 
                    [m - s for m, s in zip(means, stds)], 
                    [m + s for m, s in zip(means, stds)], 
                    color='gray', alpha=0.5)

    # ax.plot(graph_sparsity, final_test_acc_list, label='Inference Attack Accuracy after pruning')
    ax.plot(graph_sparsity, final_test_acc_list_DD, label='Inductive UGS Classification Accuracy', color='orange', linewidth=4)
    ax.set_xticks(graph_sparsity) 
    # Get current y-ticks
    current_yticks = ax.get_yticks()
    # remove the last tick
    current_yticks = current_yticks[:-1]

    # Determine the new tick
    new_tick = [min(current_yticks) - 5, min(current_yticks) - 10]

    # Add the new tick to the list of y-ticks
    all_yticks = sorted(list(current_yticks) + new_tick)

    # Set the y-ticks
    ax.set_yticks(all_yticks)

    ax.set_xlabel('Graph sparsity (%)', fontsize=20)
    ax.set_ylabel('Inference Attack Accuracy (%)', fontsize=20, color='blue')
    ax2 = ax.twinx()

    # Set the label for the second y-axis
    ax2.set_ylabel('Inductive UGS Accuracy (%)', fontsize=20, color='orange')

    ax2.set_ylim(ax.get_ylim())
    ax2.set_yticks(ax.get_yticks())

    ax.set_facecolor("#f5f5f5")  # This is a very light grey

    # Display and customize the grid
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax.grid(True, which='major', color='0.6')  # Adjusted to a slightly darker grey for better contrast

    # Customize ticks (if needed)
    ax.tick_params(axis='both', which='both', direction='in', color='0.6')


    handles, labels = ax.get_legend_handles_labels()

    # Create a custom order
    # order = ['Inductive UGS Classification Accuracy']

    # # # Reorder handles and labels
    # ordered_handles = [handles[labels.index(label)] for label in order if label in labels]
    # ordered_labels = [label for label in order if label in labels]

    # # Create a single legend for the whole figure
    # fig.legend(handles=ordered_handles, labels=ordered_labels, loc='upper center', fontsize=10, ncol=1, bbox_to_anchor=(0.5, 0.1))

    ax.legend(loc='lower left', fontsize=18)
    # plt.tight_layout(rect=[0, 0.06, 1, 1])
    plt.savefig('DD_diffpool.png')
    print('save figure to DD_diffpool.png')