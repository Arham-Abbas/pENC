import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from logic import process_audio_files_multithreaded

# Process audio files and get the data
all_data = process_audio_files_multithreaded()
data_len = len(all_data)

# Create a figure for plotting
fig, axs = plt.subplots(5, 1, figsize=(10, 15))
fig.tight_layout(pad=3.0)

# Initial plot setup
for ax in axs:
    ax.plot([])

# Function to update the plots
def update_plots(index):
    if index < data_len:
        audio_file, audio_data, inverted_audio, stereo_audio = all_data[index]
        
        axs[0].lines[0].set_data(range(len(audio_data)), audio_data)
        axs[0].set_title(f'Original Audio: {audio_file}')
        
        axs[1].lines[0].set_data(range(len(inverted_audio)), inverted_audio)
        axs[1].set_title(f'Inverted Audio: {audio_file}')
        
        axs[2].lines[0].set_data(range(len(audio_data)), audio_data + inverted_audio)
        axs[2].set_title(f'Merged Audio: {audio_file}')
        
        axs[3].lines[0].set_data(range(len(stereo_audio[:, 0])), stereo_audio[:, 0])
        axs[3].set_title(f'Mixed Audio (Left Channel): {audio_file}')
        
        axs[4].lines[0].set_data(range(len(stereo_audio[:, 1])), stereo_audio[:, 1])
        axs[4].set_title(f'Mixed Audio (Right Channel): {audio_file}')
        
        for ax in axs:
            ax.relim()
            ax.autoscale_view()
        
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
    if current_idx[0] + 1 < data_len:
        current_idx[0] += 1
        update_plots(current_idx[0])

def prev_page(_):
    if current_idx[0] - 1 >= 0:
        current_idx[0] -= 1
        update_plots(current_idx[0])

bnext.on_clicked(next_page)
bprev.on_clicked(prev_page)

# Initial plot update
update_plots(0)

# Maximize the window
manager = plt.get_current_fig_manager()
try:
    manager.window.state('zoomed')  # For Tkinter backend
except AttributeError:
    try:
        manager.full_screen_toggle()  # For other backends
    except AttributeError:
        pass

# Show the plot
plt.show()
