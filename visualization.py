import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from logic import process_audio_files

# Process audio files and get the data
all_data = process_audio_files()

# Define batch size for plotting
batch_size = 3

# Create a figure for plotting
fig, axs = plt.subplots(3, batch_size, figsize=(15, 10))
fig.tight_layout(pad=3.0)

# Function to update the plots
def update_plots(start_idx):
    for i in range(batch_size):
        if start_idx + i < len(all_data):
            audio_file, audio_data, inverted_audio, merged_audio = all_data[start_idx + i]
            
            axs[0, i].cla()
            axs[0, i].plot(audio_data)
            axs[0, i].set_title(f'Original Audio: {audio_file}')
            
            axs[1, i].cla()
            axs[1, i].plot(inverted_audio)
            axs[1, i].set_title(f'Inverted Audio: {audio_file}')
            
            axs[2, i].cla()
            axs[2, i].plot(merged_audio)
            axs[2, i].set_title(f'Merged Audio: {audio_file}')
        else:
            axs[0, i].cla()
            axs[1, i].cla()
            axs[2, i].cla()

    fig.canvas.draw()

# Add buttons for navigation
axprev = plt.axes([0.7, 0.01, 0.1, 0.075])
axnext = plt.axes([0.81, 0.01, 0.1, 0.075])
bprev = Button(axprev, 'Previous')
bnext = Button(axnext, 'Next')

# Define current index for pagination
current_idx = [0]

# Callback functions for buttons
def next_page(_):
    if current_idx[0] + batch_size < len(all_data):
        current_idx[0] += batch_size
        update_plots(current_idx[0])

def prev_page(_):
    if current_idx[0] - batch_size >= 0:
        current_idx[0] -= batch_size
        update_plots(current_idx[0])

bnext.on_clicked(next_page)
bprev.on_clicked(prev_page)

# Initial plot update
update_plots(0)

# Maximize the window
manager = plt.get_current_fig_manager()
manager.window.state('zoomed')  # For Tkinter backend

# Show the plot
plt.show()
