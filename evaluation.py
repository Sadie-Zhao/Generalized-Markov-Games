import matplotlib.pyplot as plt


def plot_loss_dict(losses, title = ""):
    num_losses = len(losses)
    num_rows = (num_losses - 1) // 3 + 1
    fig, axs = plt.subplots(num_rows, 3, figsize=(10, 3*num_rows))
    for i, (name, loss) in enumerate(losses.items()):
        row = i // 3
        col = i % 3
        if num_rows == 1:
            axs[col].plot(loss)
            axs[col].set_title(name)
            continue
        axs[row, col].plot(loss)
        axs[row, col].set_title(name)

    fig.suptitle(title, fontsize=16)  # Customize the title and font size
    plt.tight_layout(rect=[0, 0, 1, 1])

    plt.show()

def plot_loss_overlay_dict(losses, title = ""):
    num_plots = len(losses)  # Number of subplots needed
    num_cols = 3  # Number of columns per row
    num_rows = (num_plots - 1) // num_cols + 1

    fig, axs = plt.subplots(num_rows, num_cols, figsize=(10, 3 * num_rows))

    # Flatten axs array for easy indexing and handle cases where axs is not an array
    if num_rows * num_cols == 1:
        axs = [axs]
    else:
        axs = axs.flatten()

    # Iterate over the plots and their corresponding data
    for i, (plot_title, curves_dict) in enumerate(losses.items()):
        ax = axs[i]
        for legend_title, loss_values in curves_dict.items():
            ax.plot(loss_values, label=legend_title)
        ax.set_title(plot_title)
        ax.legend()

    # Remove any unused subplots
    for j in range(num_plots, len(axs)):
        fig.delaxes(axs[j])

    fig.suptitle(title, fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 1])
    plt.show()