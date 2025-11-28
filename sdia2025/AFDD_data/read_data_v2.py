import tkinter as tk
from tkinter import filedialog, messagebox
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

class PklViewerApp:
    def __init__(self, master):
        self.master = master
        master.title("PKL File Viewer and Labeler")

        self.current_directory = tk.StringVar()
        self.current_directory.set("No directory selected")

        self.opened_file_path = None
        self.loaded_data = None
        self.waveform_data = None # 'Waveform' channel data
        self.all_channel_data = {} # To store all 4 channel data for plotting
        self.labels = None # Stores labels for each 150-sample segment
        self.samples_per_segment = 150 # Renamed for clarity

        # Variables for interactive selection on continuous labeling canvas
        self.start_x = None
        self.current_rect_id = None
        self.selected_start_sample = -1
        self.selected_end_sample = -1

        self.labeling_window = None # To hold the Toplevel window for continuous labeling
        self.zoom_plot_window = None # To hold the Toplevel window for Matplotlib plot

        # --- Directory Selection Frame ---
        self.dir_frame = tk.LabelFrame(master, text="Directory Selection")
        self.dir_frame.pack(padx=10, pady=5, fill="x")

        self.browse_button = tk.Button(self.dir_frame, text="Browse Directory", command=self.browse_directory)
        self.browse_button.pack(side="left", padx=5, pady=5)

        # New Refresh button
        self.refresh_button = tk.Button(self.dir_frame, text="Refresh Files", command=self.refresh_file_list)
        self.refresh_button.pack(side="left", padx=5, pady=5)

        self.dir_label = tk.Label(self.dir_frame, textvariable=self.current_directory, wraplength=400)
        self.dir_label.pack(side="left", padx=5, pady=5)

        # --- File List Frame ---
        self.file_list_frame = tk.LabelFrame(master, text="PKL Files in Directory")
        self.file_list_frame.pack(padx=10, pady=5, fill="both", expand=True)

        self.file_list_scrollbar = tk.Scrollbar(self.file_list_frame)
        self.file_list_scrollbar.pack(side="right", fill="y")

        self.file_listbox = tk.Listbox(self.file_list_frame, yscrollcommand=self.file_list_scrollbar.set)
        self.file_listbox.pack(side="left", fill="both", expand=True, padx=5, pady=5)
        self.file_list_scrollbar.config(command=self.file_listbox.yview)

        self.file_listbox.bind("<<ListboxSelect>>", self.on_file_select)

        # --- Opened File Display Frame ---
        self.opened_file_frame = tk.LabelFrame(master, text="Opened File")
        self.opened_file_frame.pack(padx=10, pady=5, fill="x")

        self.opened_file_label = tk.Label(self.opened_file_frame, text="No file opened.")
        self.opened_file_label.pack(padx=5, pady=5)

        self.open_button = tk.Button(self.opened_file_frame, text="Open Selected File", command=self.open_selected_file)
        self.open_button.pack(padx=5, pady=5)

        # --- Plotting & Labeling Controls Frame ---
        self.action_buttons_frame = tk.Frame(master)
        self.action_buttons_frame.pack(padx=10, pady=5, fill="x")

        self.show_labeling_button = tk.Button(self.action_buttons_frame, text="Show Continuous Labeler", command=self.display_labeling_interface, state=tk.DISABLED)
        self.show_labeling_button.pack(side="left", padx=5, pady=5)

        self.show_zoom_plot_button = tk.Button(self.action_buttons_frame, text="Show Zoomable Plot", command=self.show_zoomable_plot, state=tk.DISABLED)
        self.show_zoom_plot_button.pack(side="right", padx=5, pady=5)


    def browse_directory(self):
        print("Browse Directory button clicked.")
        directory = filedialog.askdirectory()
        if directory:
            self.current_directory.set(directory)
            print(f"Selected directory: {directory}")
            self.list_pkl_files(directory)
            # Disable action buttons until a file is opened
            self.show_labeling_button.config(state=tk.DISABLED)
            self.show_zoom_plot_button.config(state=tk.DISABLED)
            # Also reset opened file path if directory changes
            self.opened_file_path = None
            self.opened_file_label.config(text="No file opened.")

    def refresh_file_list(self):
        """Refreshes the list of PKL files in the currently selected directory."""
        print("Refresh Files button clicked.")
        current_dir = self.current_directory.get()
        if os.path.isdir(current_dir):
            print(f"Refreshing PKL files in: {current_dir}")
            self.list_pkl_files(current_dir)
        else:
            messagebox.showwarning("No Directory Selected", "Please browse and select a directory first.")
            print("Cannot refresh: No valid directory selected.")


    def list_pkl_files(self, directory):
        print(f"Listing PKL files in: {directory}")
        self.file_listbox.delete(0, tk.END)
        found_files = False
        for filename in os.listdir(directory):
            if filename.endswith(".pkl"):
                self.file_listbox.insert(tk.END, filename)
                found_files = True
        if not found_files:
            print("No .pkl files found in this directory.")
        else:
            print("Finished listing .pkl files.")

    def on_file_select(self, event):
        print("on_file_select triggered!")
        selected_indices = self.file_listbox.curselection()
        print(f"Selected indices from listbox: {selected_indices}")
        if selected_indices:
            index = selected_indices[0]
            selected_file = self.file_listbox.get(index)
            full_path = os.path.join(self.current_directory.get(), selected_file)
            self.opened_file_path = full_path
            self.opened_file_label.config(text=f"Selected: {selected_file}")
            print(f"self.opened_file_path set to: {self.opened_file_path}")
        else:
            self.opened_file_path = None
            self.opened_file_label.config(text="No file selected.")
            print("No file selected in listbox (on_file_select).")


    def open_selected_file(self):
        print("open_selected_file triggered!")
        # If no file is explicitly selected in the listbox, but files exist,
        # automatically select the first one to make it more intuitive.
        if not self.file_listbox.curselection() and self.file_listbox.size() > 0:
            print("No explicit selection, attempting to auto-select first file.")
            self.file_listbox.selection_set(0) # Select the first item programmatically

            # Manually set self.opened_file_path since we just programmatically selected the first item
            first_file = self.file_listbox.get(0)
            self.opened_file_path = os.path.join(self.current_directory.get(), first_file)
            self.opened_file_label.config(text=f"Selected: {first_file}")
            print(f"Auto-selected file: {self.opened_file_path}")

        if not self.opened_file_path:
            print("Error: self.opened_file_path is None, showing warning.")
            messagebox.showwarning("No File Selected", "Please select a .pkl file from the list, or ensure a directory with .pkl files is browsed.")
            return

        print(f"Attempting to open file: {self.opened_file_path}")
        try:
            with open(self.opened_file_path, 'rb') as f:
                self.loaded_data = pickle.load(f)
            print("Pickle data loaded successfully!")

            required_channels = ['Low Frequency', 'Mid Frequency', 'High Frequency', 'Waveform']
            data_ok = True
            self.all_channel_data = {} # Clear previous data

            # Check if loaded_data is a dictionary, as expected
            if not isinstance(self.loaded_data, dict):
                messagebox.showerror("Data Error", "The .pkl file content is not a dictionary as expected.")
                data_ok = False
            else:
                for channel in required_channels:
                    if channel not in self.loaded_data:
                        messagebox.showerror("Data Error", f"The opened PKL file does not contain '{channel}' data.")
                        data_ok = False
                        break
                    # Store all channel data, ensure it's a numpy array for consistency
                    # Handle cases where data might be a list or other iterable
                    try:
                        self.all_channel_data[channel] = np.array(self.loaded_data[channel])
                        if self.all_channel_data[channel].size == 0:
                             print(f"Warning: Channel '{channel}' has empty data.")
                             # You might want to consider empty data an error for plotting later
                             # For now, we'll allow it but print a warning.
                    except Exception as arr_err:
                        messagebox.showerror("Data Conversion Error", f"Could not convert data for channel '{channel}' to a numpy array: {arr_err}")
                        data_ok = False
                        break


            if data_ok:
                self.waveform_data = self.all_channel_data['Waveform'] # 'Waveform' specifically for continuous labeling
                # Ensure waveform_data is not empty before calculating num_segments
                if len(self.waveform_data) > 0:
                    num_segments = int(np.ceil(len(self.waveform_data) / self.samples_per_segment))

                    # --- FIX: NEW LOGIC FOR LOADING EXISTING LABELS ---
                    if 'Labels' in self.loaded_data and isinstance(self.loaded_data['Labels'], np.ndarray):
                        # Verify if the loaded labels array matches the expected number of segments
                        if len(self.loaded_data['Labels']) == num_segments:
                            self.labels = self.loaded_data['Labels']
                            print("Loaded existing labels from file.")
                        else:
                            print("Warning: Existing 'Labels' array in file has inconsistent length. Re-initializing labels.")
                            self.labels = np.full(num_segments, -1, dtype=int) # Initialize new labels
                    else:
                        # If no 'Labels' key or it's not a numpy array, initialize new labels
                        self.labels = np.full(num_segments, -1, dtype=int) # Original initialization for new labels
                        print("No existing labels found or invalid, initializing new labels.")
                    # --- END FIX ---

                    print(f"Waveform data length: {len(self.waveform_data)}, Number of segments: {num_segments}")
                else:
                    messagebox.showwarning("Empty Waveform Data", "The 'Waveform' channel is empty. Labeling interface may not function correctly.")
                    self.labels = np.array([]) # Empty labels array
                    print("Waveform data is empty.")

                messagebox.showinfo("File Loaded", f"Successfully loaded '{os.path.basename(self.opened_file_path)}'. Data available for plotting and labeling.")
                self.opened_file_label.config(text=f"Opened: {os.path.basename(self.opened_file_path)}")
                self.show_labeling_button.config(state=tk.NORMAL)
                self.show_zoom_plot_button.config(state=tk.NORMAL)
                print("Action buttons enabled.")

                # Close previous plot windows if any
                if self.labeling_window and self.labeling_window.winfo_exists():
                    self.labeling_window.destroy()
                    print("Previous labeling window destroyed.")
                if self.zoom_plot_window and self.zoom_plot_window.winfo_exists():
                    self.zoom_plot_window.destroy()
                    print("Previous zoom plot window destroyed.")

            else:
                print("Data check failed, resetting state.")
                self.loaded_data = None
                self.waveform_data = None
                self.all_channel_data = {}
                self.labels = None
                self.show_labeling_button.config(state=tk.DISABLED)
                self.show_zoom_plot_button.config(state=tk.DISABLED)


        except FileNotFoundError:
            print(f"Error: File '{self.opened_file_path}' not found (Python error).")
            messagebox.showerror("Error", f"File '{self.opened_file_path}' not found.")
        except pickle.UnpicklingError as pe:
            print(f"Error: Could not unpickle file (corrupted/wrong format): {pe}")
            messagebox.showerror("Pickle Error", f"Could not unpickle the file. It might be corrupted or not a valid .pkl file: {pe}")
        except Exception as e:
            print(f"An unexpected error occurred while loading the data: {e} (Python error).")
            messagebox.showerror("Error", f"An unexpected error occurred while loading the data: {e}")
            self.loaded_data = None
            self.waveform_data = None
            self.all_channel_data = {}
            self.labels = None
            self.show_labeling_button.config(state=tk.DISABLED)
            self.show_zoom_plot_button.config(state=tk.DISABLED)

    def display_labeling_interface(self):
        print("Display Labeling Interface requested.")
        if self.waveform_data is None:
            messagebox.showwarning("No Data", "Please open a .pkl file with 'Waveform' data first.")
            print("Cannot display labeling interface: waveform_data is None.")
            return

        # Destroy existing window if it's open
        if self.labeling_window and self.labeling_window.winfo_exists():
            self.labeling_window.destroy()
            print("Previous labeling window destroyed before creating new one.")

        self.labeling_window = tk.Toplevel(self.master)
        self.labeling_window.title(f"Continuous Labeling: {os.path.basename(self.opened_file_path)}")
        self.labeling_window.geometry("800x400")
        print("Labeling window created.")

        # Canvas for waveform display
        self.canvas = tk.Canvas(self.labeling_window, bg="white", borderwidth=2, relief="groove")
        self.canvas.pack(side="top", fill="both", expand=True, padx=10, pady=10)
        self.canvas.bind("<Configure>", self.on_canvas_configure)
        self.canvas.bind("<ButtonPress-1>", self.on_press)
        self.canvas.bind("<B1-Motion>", self.on_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_release)
        print("Canvas created and bindings set.")

        # Controls Frame
        self.controls_frame = tk.Frame(self.labeling_window)
        self.controls_frame.pack(side="bottom", fill="x", padx=10, pady=5)

        # Label selection (0 or 1)
        self.label_value = tk.IntVar(value=0) # Default to 0
        self.label_scale = tk.Scale(self.controls_frame, from_=0, to=1, orient="horizontal", variable=self.label_value, resolution=1, length=100)
        self.label_scale.set(0) # Default label is 0. Change to 1 if you want green by default.
        self.label_scale.pack(side="left", padx=5, pady=5)
        tk.Label(self.controls_frame, text="Label (0=Red, 1=Green)").pack(side="left", padx=5, pady=5)


        self.apply_label_button = tk.Button(self.controls_frame, text="Apply Label to Selected Range", command=self.apply_selected_range_label)
        self.apply_label_button.pack(side="left", padx=5, pady=5)

        self.selected_range_label = tk.Label(self.controls_frame, text="Selected Range: None")
        self.selected_range_label.pack(side="left", padx=10, pady=5)

        self.save_button = tk.Button(self.controls_frame, text="Save Labeled Data", command=self.save_labeled_data)
        self.save_button.pack(side="right", padx=5, pady=5)
        print("Labeling window controls created.")

        # Trigger initial plot update
        # self.update_continuous_plot() # Called by on_canvas_configure anyway

    def on_canvas_configure(self, event):
        # This function is called when the canvas is resized or first appears
        # print(f"Canvas configure event: width={event.width}, height={event.height}") # Uncomment for detailed debug
        self.update_continuous_plot()

    def update_continuous_plot(self):
        # print("Updating continuous plot...") # Uncomment for detailed debug
        self.canvas.delete("all")
        if self.waveform_data is None:
            return

        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()

        # IMPORTANT: When the window first appears, winfo_width/height might be 1.
        # We need to wait for it to have actual dimensions.
        if canvas_width < 10 or canvas_height < 10: # Use a small threshold
            # print(f"Canvas too small ({canvas_width}x{canvas_height}), retrying in 100ms.") # Uncomment for detailed debug
            self.master.after(100, self.update_continuous_plot) # Try again after 100ms
            return

        # Normalize data for plotting
        min_val = np.min(self.waveform_data)
        max_val = np.max(self.waveform_data)

        if max_val == min_val: # Handle flat line data
            scaled_data = np.full_like(self.waveform_data, canvas_height / 2)
        else:
            scaled_data = (self.waveform_data - min_val) / (max_val - min_val) * (canvas_height * 0.8) + (canvas_height * 0.1)
            scaled_data = canvas_height - scaled_data # Invert y-axis for plotting

        # Draw the waveform
        if len(scaled_data) > 1:
            points = []
            for i in range(len(scaled_data)):
                # Scale x-coordinate to fit canvas width
                x = i * (canvas_width / (len(self.waveform_data) - 1)) if len(self.waveform_data) > 1 else 0
                y = scaled_data[i]
                points.extend([x, y])
            self.canvas.create_line(points, fill="blue", width=1)
        elif len(scaled_data) == 1: # Single data point
            x = canvas_width / 2
            y = scaled_data[0]
            self.canvas.create_oval(x-2, y-2, x+2, y+2, fill="blue", outline="blue")

        # Draw existing labels as colored segments
        # Ensure self.labels is not None and has the expected length before iterating
        if self.labels is not None and len(self.labels) == int(np.ceil(len(self.waveform_data) / self.samples_per_segment)):
            for i, label in enumerate(self.labels):
                if label != -1: # Only draw if labeled (0 or 1)
                    start_sample_segment = i * self.samples_per_segment
                    end_sample_segment = min((i + 1) * self.samples_per_segment, len(self.waveform_data))

                    x1 = (start_sample_segment / len(self.waveform_data)) * canvas_width
                    x2 = (end_sample_segment / len(self.waveform_data)) * canvas_width

                    color = "green" if label == 1 else "red"
                    # FIX FOR TCL ERROR: changed stipple="33" to stipple="gray25"
                    self.canvas.create_rectangle(x1, 0, x2, canvas_height, fill=color, stipple="gray25", outline="", tags="labels_highlight")


        # Redraw the selection rectangle if it exists
        if self.start_x is not None and self.current_rect_id is not None:
            # We need to ensure the rectangle is drawn on top of the signal
            self.canvas.coords(self.current_rect_id, self.start_x, 0, self.current_x, canvas_height)


    def on_press(self, event):
        self.start_x = event.x
        # Clear previous selection rectangle if any
        if self.current_rect_id:
            self.canvas.delete(self.current_rect_id)
        # Create a new rectangle (initially just a line)
        self.current_rect_id = self.canvas.create_rectangle(self.start_x, 0, self.start_x, self.canvas.winfo_height(),
                                                             outline="purple", width=2, tags="selection_rect")
        self.current_x = event.x # Keep track of current mouse position

    def on_drag(self, event):
        if self.start_x is not None and self.current_rect_id:
            self.current_x = event.x
            self.canvas.coords(self.current_rect_id, self.start_x, 0, self.current_x, self.canvas.winfo_height())

            # Update selected range label dynamically
            canvas_width = self.canvas.winfo_width()
            if canvas_width > 0 and len(self.waveform_data) > 0: # Check for empty data
                # Convert canvas X coordinates to sample indices
                temp_start_sample = int((min(self.start_x, self.current_x) / canvas_width) * len(self.waveform_data))
                temp_end_sample = int((max(self.start_x, self.current_x) / canvas_width) * len(self.waveform_data))
                # Ensure end_sample is within bounds
                temp_end_sample = min(temp_end_sample, len(self.waveform_data) -1)
                self.selected_range_label.config(text=f"Selected Range: {temp_start_sample} - {temp_end_sample}")


    def on_release(self, event):
        if self.start_x is not None:
            end_x = event.x
            canvas_width = self.canvas.winfo_width()

            if canvas_width == 0 or len(self.waveform_data) == 0:
                self.start_x = None
                self.selected_range_label.config(text="Selected Range: None")
                return

            # Convert canvas X coordinates to sample indices
            self.selected_start_sample = int((min(self.start_x, end_x) / canvas_width) * len(self.waveform_data))
            self.selected_end_sample = int((max(self.start_x, end_x) / canvas_width) * len(self.waveform_data))

            # Ensure end_sample is within bounds
            self.selected_end_sample = min(self.selected_end_sample, len(self.waveform_data) -1)


            self.selected_range_label.config(text=f"Selected Range: {self.selected_start_sample} - {self.selected_end_sample}")

            # Ensure the selection rectangle is redrawn to its final position
            if self.current_rect_id:
                self.canvas.coords(self.current_rect_id, min(self.start_x, end_x), 0, max(self.start_x, end_x), self.canvas.winfo_height())
            self.start_x = None # Reset for next selection

    def apply_selected_range_label(self):
        print("Apply Label button clicked.")
        if self.waveform_data is None:
            messagebox.showwarning("No Data", "No waveform data loaded to label.")
            print("Apply label failed: waveform_data is None.")
            return
        # Check if a valid range has been selected
        if self.selected_start_sample == -1 or self.selected_end_sample == -1 or self.selected_start_sample >= self.selected_end_sample:
            messagebox.showwarning("No Selection", "Please select a range on the waveform first by clicking and dragging.")
            print("Apply label failed: No valid selection made.")
            return

        label_value = self.label_value.get()
        print(f"Applying label {label_value} to selected range.")

        # Determine which 150-sample segments fall within the selected range
        start_segment_index = self.selected_start_sample // self.samples_per_segment
        # Calculate end_segment_index considering it's the segment containing the end_sample
        end_segment_index = self.selected_end_sample // self.samples_per_segment


        # Ensure indices are within bounds of the labels array
        start_segment_index = max(0, start_segment_index)
        end_segment_index = min(len(self.labels) - 1, end_segment_index)

        # The condition `start_segment_index > end_segment_index` might occur if the selection
        # is very small and only covers part of one segment, or if end_sample is 0 for some reason.
        # Ensure we always process at least one segment if a valid selection was made.
        if start_segment_index > end_segment_index:
            # If the selection is tiny, it might still fall within a single segment
            if (self.selected_end_sample - self.selected_start_sample) > 0 and self.samples_per_segment > 0:
                # If there's any valid range, label at least the segment it starts in
                end_segment_index = start_segment_index
            else:
                messagebox.showwarning("Invalid Selection", "Selected range is too small or invalid for segment labeling.")
                print("Apply label failed: Calculated segment range is invalid.")
                return


        for i in range(start_segment_index, end_segment_index + 1):
            self.labels[i] = label_value
            print(f"Labeled segment {i} (samples {i*self.samples_per_segment}-{min((i+1)*self.samples_per_segment, len(self.waveform_data))-1}) with {label_value}")

        messagebox.showinfo("Label Applied", f"Label {label_value} applied to segments covering samples {self.selected_start_sample} to {self.selected_end_sample}.")
        self.update_continuous_plot() # Redraw to show the new labels
        # Clear the selection after applying label
        if self.current_rect_id:
            self.canvas.delete(self.current_rect_id)
            self.current_rect_id = None
        self.selected_start_sample = -1
        self.selected_end_sample = -1
        self.selected_range_label.config(text="Selected Range: None")

    def save_labeled_data(self):
        print("Save Labeled Data button clicked.")
        if self.loaded_data is None or self.labels is None:
            messagebox.showwarning("No Data to Save", "No labeled data available to save.")
            print("Save failed: No data or labels to save.")
            return

        # The filedialog.asksaveasfilename() function automatically provides a filename input box.
        # The 'initialfile' argument just pre-fills it with a suggestion.
        # The user can freely type any name they want in the dialog.
        save_path = filedialog.asksaveasfilename(defaultextension=".pkl",
                                                 filetypes=[("PKL files", "*.pkl")],
                                                 initialfile=f"labeled_{os.path.basename(self.opened_file_path)}")
        if save_path:
            print(f"Saving labeled data to: {save_path}")
            # Create a copy of the loaded data and add the 'Labels' key
            # This is where the 'Labels' array is added to the dictionary
            data_to_save = self.loaded_data.copy()
            data_to_save['Labels'] = self.labels

            try:
                with open(save_path, 'wb') as f:
                    pickle.dump(data_to_save, f)
                messagebox.showinfo("Save Successful", f"Labeled data saved to '{os.path.basename(save_path)}'")
                print("Labeled data saved successfully.")
            except Exception as e:
                messagebox.showerror("Save Error", f"An error occurred while saving the data: {e}")
                print(f"Save failed: {e}")

    def show_zoomable_plot(self):
        print("Show Zoomable Plot button clicked.")
        if not self.all_channel_data:
            messagebox.showwarning("No Data", "Please open a PKL file with all required channel data first.")
            print("Cannot show zoomable plot: No channel data loaded.")
            return

        if self.zoom_plot_window and self.zoom_plot_window.winfo_exists():
            self.zoom_plot_window.destroy() # Close previous window if open
            print("Destroyed previous zoom plot window.")

        self.zoom_plot_window = tk.Toplevel(self.master)
        self.zoom_plot_window.title(f"Zoomable Plot: {os.path.basename(self.opened_file_path)}")
        self.zoom_plot_window.geometry("1200x800")
        print("Zoomable plot window created.")

        # Create Matplotlib Figure and Axes
        # sharex=True is important for synchronized zooming across subplots
        fig, axes = plt.subplots(4, 1, sharex=True, figsize=(10, 7))
        fig.suptitle(f"All Channels and Labels for {os.path.basename(self.opened_file_path)}")
        print("Matplotlib figure and subplots created.")

        channel_names = ['Low Frequency', 'Mid Frequency', 'High Frequency', 'Waveform']

        for i, ax in enumerate(axes):
            channel_name = channel_names[i]
            data = self.all_channel_data.get(channel_name)

            if data is not None and data.size > 0: # Check for non-empty numpy array
                ax.plot(data, label=channel_name)
                ax.set_ylabel(channel_name)
                ax.grid(True, linestyle='--', alpha=0.6)
                ax.legend(loc='upper right')
                print(f"Plotted data for channel: {channel_name}")

                # Add labels as colored spans
                # Ensure self.labels is not None and has the expected length before iterating
                if self.labels is not None and len(self.labels) > 0 and len(data) > 0 and \
                   len(self.labels) == int(np.ceil(len(self.waveform_data) / self.samples_per_segment)):
                    for j, label in enumerate(self.labels):
                        if label != -1:
                            start_sample_segment = j * self.samples_per_segment
                            end_sample_segment = min((j + 1) * self.samples_per_segment, len(data))

                            color = "green" if label == 1 else "red"
                            ax.axvspan(start_sample_segment, end_sample_segment,
                                       ymin=0, ymax=1, transform=ax.get_yaxis_transform(),
                                       facecolor=color, alpha=0.2, lw=0) # lw=0 to remove border
                    print(f"Added labels to plot for channel: {channel_name}")
            else:
                ax.text(0.5, 0.5, f"No data for {channel_name}", horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
                ax.set_ylabel(channel_name)
                print(f"No data or empty data for channel: {channel_name}")


        axes[-1].set_xlabel("Samples") # Only label the x-axis on the bottom plot
        fig.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to make space for suptitle

        # Embed plot into Tkinter window
        canvas = FigureCanvasTkAgg(fig, master=self.zoom_plot_window)
        canvas_widget = canvas.get_tk_widget()
        canvas_widget.pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        print("Matplotlib canvas embedded into Tkinter.")

        # Add Matplotlib Navigation Toolbar
        toolbar = NavigationToolbar2Tk(canvas, self.zoom_plot_window)
        toolbar.update()
        canvas_widget.pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        print("Matplotlib navigation toolbar added.")

        # Bind the window closing event to destroy the figure to free memory
        self.zoom_plot_window.protocol("WM_DELETE_WINDOW", lambda: self._on_zoom_plot_window_close(fig))

    def _on_zoom_plot_window_close(self, fig):
        print("Zoom plot window close event triggered.")
        # This function is called when the Matplotlib plot window is closed
        plt.close(fig) # Close the matplotlib figure to free memory
        self.zoom_plot_window.destroy() # Destroy the Tkinter Toplevel window
        self.zoom_plot_window = None # Reset the reference
        print("Zoom plot window destroyed and figure closed.")

def main():
    root = tk.Tk()
    app = PklViewerApp(root)
    root.geometry("600x450") # Set initial size for the main window
    root.mainloop()

if __name__ == "__main__":
    main()