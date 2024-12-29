import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from logic import process_audio_files

# Process audio files and get the data
all_data = process_audio_files()

# Create a figure for plotting
fig, axs = plt.subplots(5, 1, figsize=(10, 15))
fig.tight_layout(pad=3.0)

# Function to update the plots
def update_plots(index):
    if index < len(all_data):
        audio_file, audio_data, inverted_audio, stereo_audio = all_data[index]
        
        axs[0].cla()
        axs[0].plot(audio_data)
        axs[0].set_title(f'Original Audio: {audio_file}')
        
        axs[1].cla()
        axs[1].plot(inverted_audio)
        axs[1].set_title(f'Inverted Audio: {audio_file}')
        
        axs[2].cla()
        axs[2].plot(audio_data + inverted_audio)
        axs[2].set_title(f'Merged Audio: {audio_file}')
        
        axs[3].cla()
        axs[3].plot(stereo_audio[:, 0])
        axs[3].set_title(f'Mixed Audio (Left Channel): {audio_file}')
        
        axs[4].cla()
        axs[4].plot(stereo_audio[:, 1])
        axs[4].set_title(f'Mixed Audio (Right Channel): {audio_file}')

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
    if current_idx[0] + 1 < len(all_data):
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
