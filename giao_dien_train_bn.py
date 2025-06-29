import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import threading
import queue
import sys
import os
import json
import re
from datetime import datetime
import numpy as np
from PIL import Image, ImageTk
import io
import contextlib

# Import your YOLO training module
try:
    from traindata_buonngu import YOLOTrainBuonNgu, check_dataset
except ImportError:
    print("Không thể import module traindata_buonngu.py. Đảm bảo file traindata_buonngu.py có trong cùng thư mục.")
    sys.exit(1)

class ConsoleRedirector:
    """Redirect console output to GUI with formatting"""
    def __init__(self, text_widget, queue_obj, tag="normal"):
        self.text_widget = text_widget
        self.queue = queue_obj
        self.tag = tag
        self.buffer = ""
        
    def write(self, text):
        if text:
            # Buffer text to handle partial lines
            self.buffer += text
            
            # Process complete lines
            while '\n' in self.buffer:
                line, self.buffer = self.buffer.split('\n', 1)
                if line.strip():  # Only add non-empty lines
                    formatted_line = self.format_line(line)
                    self.queue.put((formatted_line + '\n', self.determine_tag(line)))
        
    def format_line(self, line):
        """Format console line for better readability"""
        # Clean up YOLO progress bars and excessive characters
        line = re.sub(r'[█▌▍▎▏▊▋▉]+', '█', line)  # Simplify progress bars
        line = re.sub(r'#{10,}', '█████████', line)  # Replace long hash sequences
        line = re.sub(r'\x1b\[[0-9;]*m', '', line)  # Remove ANSI color codes
        
        # Add timestamp for important messages
        if any(keyword in line.lower() for keyword in ['epoch', 'error', 'complete', 'starting', 'validation']):
            timestamp = datetime.now().strftime("%H:%M:%S")
            line = f"[{timestamp}] {line}"
            
        return line
    
    def determine_tag(self, line):
        """Determine text tag based on line content"""
        line_lower = line.lower()
        
        if any(keyword in line_lower for keyword in ['error', 'failed', 'exception']):
            return "error"
        elif any(keyword in line_lower for keyword in ['warning', 'warn']):
            return "warning"
        elif any(keyword in line_lower for keyword in ['complete', 'success', 'finished']):
            return "success"
        elif 'epoch' in line_lower:
            return "epoch"
        elif any(keyword in line_lower for keyword in ['map', 'precision', 'recall']):
            return "metrics"
        else:
            return "normal"
        
    def flush(self):
        pass

class YOLOTrainingGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("YOLO Training GUI - Drowsiness Detection")
        self.root.geometry("1200x800")
        self.root.configure(bg='#2b2b2b')
        
        # Variables
        self.trainer = None
        self.training_thread = None
        self.is_training = False
        self.stop_requested = False
        self.console_queue = queue.Queue()
        
        # Training metrics for status display
        self.current_epoch = 0
        self.total_epochs = 0
        self.best_map = 0.0
        self.current_loss = 0.0
        
        self.setup_ui()
        self.setup_console_redirect()
        self.update_console()
        
    def setup_ui(self):
        """Setup the main UI"""
        # Create main frame
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Left panel for configuration
        left_frame = ttk.Frame(main_frame)
        left_frame.pack(side='left', fill='y', padx=(0, 10))
        
        # Right panel for console output
        right_frame = ttk.Frame(main_frame)
        right_frame.pack(side='right', fill='both', expand=True)
        
        self.setup_control_panel(left_frame)
        self.setup_console_panel(right_frame)
        
    def setup_control_panel(self, parent):
        """Setup training configuration and control panel"""
        # Configuration section
        config_frame = ttk.LabelFrame(parent, text="Training Configuration", padding=10)
        config_frame.pack(fill='x', pady=(0, 10))
        
        # Dataset path
        ttk.Label(config_frame, text="Dataset Path:").pack(anchor='w')
        self.dataset_path_var = tk.StringVar(value="dataset_buonngu")
        dataset_frame = ttk.Frame(config_frame)
        dataset_frame.pack(fill='x', pady=(0, 10))
        
        self.dataset_entry = ttk.Entry(dataset_frame, textvariable=self.dataset_path_var, width=40)
        self.dataset_entry.pack(side='left', fill='x', expand=True)
        
        ttk.Button(dataset_frame, text="Browse", 
                  command=self.browse_dataset).pack(side='right', padx=(5, 0))
        
        # Model save path
        ttk.Label(config_frame, text="Model Save Path:").pack(anchor='w')
        self.model_path_var = tk.StringVar(value="savemodel")
        model_frame = ttk.Frame(config_frame)
        model_frame.pack(fill='x', pady=(0, 10))
        
        self.model_entry = ttk.Entry(model_frame, textvariable=self.model_path_var, width=40)
        self.model_entry.pack(side='left', fill='x', expand=True)
        
        ttk.Button(model_frame, text="Browse", 
                  command=self.browse_model_path).pack(side='right', padx=(5, 0))
        
        # Epochs
        ttk.Label(config_frame, text="Epochs:").pack(anchor='w')
        self.epochs_var = tk.StringVar(value="100")
        epochs_spinbox = ttk.Spinbox(config_frame, from_=50, to=300, 
                                   textvariable=self.epochs_var, width=15)
        epochs_spinbox.pack(anchor='w', pady=(0, 10))
        
        # Device selection
        ttk.Label(config_frame, text="Device:").pack(anchor='w')
        self.device_var = tk.StringVar(value="auto")
        device_combo = ttk.Combobox(config_frame, textvariable=self.device_var,
                                   values=["auto", "cuda", "cpu"], state="readonly", width=15)
        device_combo.pack(anchor='w', pady=(0, 10))
        
        # Batch size
        ttk.Label(config_frame, text="Batch Size:").pack(anchor='w')
        self.batch_var = tk.StringVar(value="auto")
        batch_combo = ttk.Combobox(config_frame, textvariable=self.batch_var,
                                  values=["auto", "4", "8", "12", "16", "32"], state="readonly", width=15)
        batch_combo.pack(anchor='w', pady=(0, 10))
        
        # Control buttons
        control_frame = ttk.LabelFrame(parent, text="Training Control", padding=10)
        control_frame.pack(fill='x', pady=(0, 10))
        
        self.check_dataset_btn = ttk.Button(control_frame, text="Check Dataset",
                                          command=self.check_dataset_status)
        self.check_dataset_btn.pack(fill='x', pady=(0, 5))
        
        self.start_btn = ttk.Button(control_frame, text="Start Training",
                                   command=self.start_training, style="Accent.TButton")
        self.start_btn.pack(fill='x', pady=(0, 5))
        
        self.stop_btn = ttk.Button(control_frame, text="Stop Training",
                                  command=self.stop_training, state='disabled')
        self.stop_btn.pack(fill='x')
        
        # Status section
        status_frame = ttk.LabelFrame(parent, text="Training Status", padding=10)
        status_frame.pack(fill='both', expand=True)
        
        self.status_label = ttk.Label(status_frame, text="Ready", 
                                     foreground="green", font=("Arial", 10, "bold"))
        self.status_label.pack(anchor='w', pady=(0, 5))
        
        # Progress bar
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(status_frame, variable=self.progress_var,
                                          maximum=100, mode='determinate')
        self.progress_bar.pack(fill='x', pady=(0, 10))
        
        # Current epoch label
        self.epoch_label = ttk.Label(status_frame, text="Epoch: 0/0", font=("Arial", 9))
        self.epoch_label.pack(anchor='w', pady=(0, 5))
        
        # Training metrics display
        metrics_subframe = ttk.Frame(status_frame)
        metrics_subframe.pack(fill='x', pady=(5, 0))
        
        self.loss_label = ttk.Label(metrics_subframe, text="Loss: N/A", font=("Arial", 9))
        self.loss_label.pack(anchor='w')
        
        self.map_label = ttk.Label(metrics_subframe, text="Best mAP: N/A", font=("Arial", 9))
        self.map_label.pack(anchor='w')
        
        # Training time
        self.time_label = ttk.Label(metrics_subframe, text="Training Time: 00:00:00", font=("Arial", 9))
        self.time_label.pack(anchor='w')
        
        # Results section
        results_frame = ttk.LabelFrame(parent, text="Results", padding=10)
        results_frame.pack(fill='x', pady=(10, 0))
        
        ttk.Button(results_frame, text="Open Model Folder",
                  command=self.open_model_folder).pack(fill='x', pady=(0, 5))
        
        ttk.Button(results_frame, text="Save Training Log",
                  command=self.save_console_log).pack(fill='x')
        
    def setup_console_panel(self, parent):
        """Setup console output panel"""
        # Console output
        console_frame = ttk.LabelFrame(parent, text="Training Console", padding=5)
        console_frame.pack(fill='both', expand=True)
        
        self.setup_console_text(console_frame)
        
        # Console control buttons
        console_btn_frame = ttk.Frame(console_frame)
        console_btn_frame.pack(fill='x', pady=(5, 0))
        
        ttk.Button(console_btn_frame, text="Clear Console",
                  command=self.clear_console).pack(side='left')
        
        # Auto-scroll checkbox
        self.auto_scroll_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(console_btn_frame, text="Auto-scroll", 
                       variable=self.auto_scroll_var).pack(side='right')
        
    def setup_console_text(self, parent):
        """Setup console text widget with color formatting"""
        self.console_text = scrolledtext.ScrolledText(parent, 
                                                     bg='#1e1e1e', fg='#ffffff',
                                                     font=("Consolas", 9),
                                                     state='disabled',
                                                     wrap='word')
        self.console_text.pack(fill='both', expand=True)
        
        # Configure text tags for different message types
        self.console_text.tag_configure("normal", foreground="#ffffff")
        self.console_text.tag_configure("error", foreground="#ff6b6b", font=("Consolas", 9, "bold"))
        self.console_text.tag_configure("warning", foreground="#ffa500", font=("Consolas", 9, "bold"))
        self.console_text.tag_configure("success", foreground="#51cf66", font=("Consolas", 9, "bold"))
        self.console_text.tag_configure("epoch", foreground="#74c0fc", font=("Consolas", 9, "bold"))
        self.console_text.tag_configure("metrics", foreground="#ffd43b")
        
    def setup_console_redirect(self):
        """Setup console output redirection"""
        self.original_stdout = sys.stdout
        self.original_stderr = sys.stderr
        self.training_start_time = None
        
    def update_console(self):
        """Update console text with queued messages"""
        try:
            while True:
                message, tag = self.console_queue.get_nowait()
                self.append_to_console(message, tag)
        except queue.Empty:
            pass
        
        # Update training time if training
        if self.is_training and self.training_start_time:
            elapsed = datetime.now() - self.training_start_time
            time_str = str(elapsed).split('.')[0]  # Remove microseconds
            self.time_label.config(text=f"Training Time: {time_str}")
        
        # Schedule next update
        self.root.after(100, self.update_console)
        
    def append_to_console(self, message, tag="normal"):
        """Append message to console with formatting"""
        self.console_text.config(state='normal')
        
        # Insert message with appropriate tag
        self.console_text.insert('end', message, tag)
        
        # Auto-scroll if enabled
        if self.auto_scroll_var.get():
            self.console_text.see('end')
            
        self.console_text.config(state='disabled')
        
        # Update progress if training
        if self.is_training:
            self.extract_training_progress(message)
            
    def extract_training_progress(self, message):
        """Extract training progress from console messages"""
        try:
            # Look for epoch information in various formats
            epoch_patterns = [
                r'Epoch\s+(\d+)/(\d+)',
                r'(\d+)/(\d+)\s+epochs',
                r'Epoch:\s*(\d+)/(\d+)',
                r'\[(\d+)/(\d+)\]'
            ]
            
            for pattern in epoch_patterns:
                match = re.search(pattern, message)
                if match:
                    self.current_epoch = int(match.group(1))
                    self.total_epochs = int(match.group(2))
                    progress = (self.current_epoch / self.total_epochs) * 100
                    self.progress_var.set(progress)
                    self.epoch_label.config(text=f"Epoch: {self.current_epoch}/{self.total_epochs}")
                    break
                    
            # Extract loss information
            loss_patterns = [
                r'loss:\s*([\d.]+)',
                r'Loss:\s*([\d.]+)',
                r'train_loss:\s*([\d.]+)'
            ]
            
            for pattern in loss_patterns:
                match = re.search(pattern, message)
                if match:
                    self.current_loss = float(match.group(1))
                    self.loss_label.config(text=f"Loss: {self.current_loss:.4f}")
                    break
                    
            # Extract mAP information
            map_patterns = [
                r'mAP50:\s*([\d.]+)',
                r'mAP@0\.5:\s*([\d.]+)',
                r'map50:\s*([\d.]+)'
            ]
            
            for pattern in map_patterns:
                match = re.search(pattern, message)
                if match:
                    map_value = float(match.group(1))
                    if map_value > self.best_map:
                        self.best_map = map_value
                    self.map_label.config(text=f"Best mAP: {self.best_map:.4f}")
                    break
                    
        except Exception as e:
            pass  # Ignore parsing errors
            
    def browse_dataset(self):
        """Browse for dataset directory"""
        directory = filedialog.askdirectory(title="Select Dataset Directory")
        if directory:
            self.dataset_path_var.set(directory)
            
    def browse_model_path(self):
        """Browse for model save directory"""
        directory = filedialog.askdirectory(title="Select Model Save Directory")
        if directory:
            self.model_path_var.set(directory)
            
    def check_dataset_status(self):
        """Check dataset status"""
        dataset_path = self.dataset_path_var.get()
        
        self.append_to_console(f"\n{'='*50}\n", "normal")
        self.append_to_console(f"🔍 Checking dataset: {dataset_path}\n", "normal")
        
        try:
            is_valid = check_dataset(dataset_path)
            if is_valid:
                self.status_label.config(text="Dataset: Valid ✓", foreground="green")
                self.append_to_console("✅ Dataset is valid and ready for training!\n", "success")
                messagebox.showinfo("Dataset Check", "Dataset is valid and ready for training!")
            else:
                self.status_label.config(text="Dataset: Invalid ✗", foreground="red")
                self.append_to_console("❌ Dataset is invalid. Please check the structure.\n", "error")
                messagebox.showerror("Dataset Check", "Dataset is invalid. Please check the structure.")
        except Exception as e:
            self.append_to_console(f"❌ Error checking dataset: {e}\n", "error")
            self.status_label.config(text="Dataset: Error ✗", foreground="red")
            
    def start_training(self):
        """Start training in a separate thread"""
        if self.is_training:
            messagebox.showwarning("Training", "Training is already in progress!")
            return
            
        # Validate inputs
        try:
            epochs = int(self.epochs_var.get())
            if not (50 <= epochs <= 300):
                raise ValueError("Epochs must be between 50 and 300")
        except ValueError as e:
            messagebox.showerror("Invalid Input", f"Invalid epochs value: {e}")
            return
            
        dataset_path = self.dataset_path_var.get()
        if not os.path.exists(dataset_path):
            messagebox.showerror("Invalid Path", "Dataset path does not exist!")
            return
            
        # Reset training variables
        self.stop_requested = False
        self.current_epoch = 0
        self.total_epochs = epochs
        self.best_map = 0.0
        self.current_loss = 0.0
        self.training_start_time = datetime.now()
        
        # Update UI
        self.is_training = True
        self.start_btn.config(state='disabled')
        self.stop_btn.config(state='normal')
        self.status_label.config(text="Training in progress...", foreground="orange")
        self.progress_var.set(0)
        self.epoch_label.config(text=f"Epoch: 0/{epochs}")
        self.loss_label.config(text="Loss: N/A")
        self.map_label.config(text="Best mAP: N/A")
        
        # Clear console and add header
        self.append_to_console(f"\n{'='*60}\n", "normal")
        self.append_to_console(f"🚀 Starting YOLO Training Session\n", "success")
        self.append_to_console(f"📅 Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n", "normal")
        self.append_to_console(f"📁 Dataset: {dataset_path}\n", "normal")
        self.append_to_console(f"💾 Save Path: {self.model_path_var.get()}\n", "normal")
        self.append_to_console(f"🔄 Epochs: {epochs}\n", "normal")
        self.append_to_console(f"⚙️  Device: {self.device_var.get()}\n", "normal")
        self.append_to_console(f"{'='*60}\n\n", "normal")
        
        self.training_thread = threading.Thread(target=self.run_training, daemon=True)
        self.training_thread.start()
        
    def run_training(self):
        """Run training process"""
        try:
            # Add stop check at the beginning
            if self.stop_requested:
                return
                
            # Redirect stdout to console
            stdout_redirector = ConsoleRedirector(self.console_text, self.console_queue, "normal")
            stderr_redirector = ConsoleRedirector(self.console_text, self.console_queue, "error")
            
            sys.stdout = stdout_redirector
            sys.stderr = stderr_redirector
            
            # Initialize trainer
            self.trainer = YOLOTrainBuonNgu()
            self.trainer.dataset_path = self.dataset_path_var.get()
            self.trainer.model_save_path = self.model_path_var.get()
            
            # Add stop mechanism to trainer if possible
            if hasattr(self.trainer, 'set_stop_callback'):
                self.trainer.set_stop_callback(lambda: self.stop_requested)
            
            # Get training parameters
            epochs = int(self.epochs_var.get())
            
            # Check for stop before starting
            if self.stop_requested:
                self.console_queue.put(("\n⏹️  Training cancelled before start!\n", "warning"))
                return
            
            # Start training
            self.console_queue.put(("🎯 Initializing training...\n", "epoch"))
            
            # Wrap training call with periodic stop checks
            model, results = None, None
            try:
                model, results = self.trainer.train_yolov8s(epochs=epochs)
            except SystemExit:
                # Handle forced termination
                self.console_queue.put(("\n🛑 Training terminated by system!\n", "error"))
                return
            except KeyboardInterrupt:
                # Handle Ctrl+C or manual interruption
                self.console_queue.put(("\n🛑 Training interrupted!\n", "warning"))
                return
            
            # Check final result
            if self.stop_requested:
                self.console_queue.put(("\n⏹️  Training stopped by user!\n", "warning"))
                self.root.after(0, lambda: self.status_label.config(text="Training Stopped", foreground="orange"))
            elif model and results:
                self.console_queue.put(("\n🎉 Training completed successfully!\n", "success"))
                self.root.after(0, lambda: self.status_label.config(text="Training Complete ✓", foreground="green"))
            else:
                self.console_queue.put(("\n❌ Training failed!\n", "error"))
                self.root.after(0, lambda: self.status_label.config(text="Training Failed ✗", foreground="red"))
                
        except Exception as e:
            if not self.stop_requested:  # Only show error if not stopped by user
                self.console_queue.put((f"\n❌ Training error: {e}\n", "error"))
                self.root.after(0, lambda: self.status_label.config(text="Training Error ✗", foreground="red"))
        finally:
            # Restore stdout/stderr
            sys.stdout = self.original_stdout
            sys.stderr = self.original_stderr
            
            # Update UI
            self.is_training = False
            self.root.after(0, self.training_finished)
            
    def training_finished(self):
        """Called when training is finished"""
        self.start_btn.config(state='normal')
        self.stop_btn.config(state='disabled')
        if not self.stop_requested:
            self.progress_var.set(100)
        
    def stop_training(self):
        """Stop training process immediately"""
        if not self.is_training:
            return
            
        result = messagebox.askyesno("Stop Training", 
                                   "Are you sure you want to stop training?\n\n"
                                   "Training will be terminated immediately.")
        
        if result:
            self.stop_requested = True
            self.append_to_console("\n🛑 Stopping training immediately...\n", "warning")
            self.status_label.config(text="Stopping training...", foreground="orange")
            
            # Disable stop button to prevent multiple clicks
            self.stop_btn.config(state='disabled')
            
            # Force terminate the training thread immediately
            self.force_stop_training()
    
    def force_stop_training(self):
        """Force stop training immediately"""
        try:
            # Set stop flag
            self.stop_requested = True
            
            # Try to terminate trainer if it exists
            if self.trainer and hasattr(self.trainer, 'stop'):
                self.trainer.stop()
            
            # Force kill the training thread if it exists and is alive
            if self.training_thread and self.training_thread.is_alive():
                try:
                    import ctypes
                    # Get thread ID and force terminate (Windows only)
                    if hasattr(self.training_thread, '_thread_id'):
                        thread_id = self.training_thread._thread_id
                        if sys.platform == "win32":
                            ctypes.pythonapi.PyThreadState_SetAsyncExc(
                                ctypes.c_long(thread_id), 
                                ctypes.py_object(SystemExit)
                            )
                except:
                    pass  # If force termination fails, continue with normal cleanup
            
            # Immediately update UI
            self.is_training = False
            self.append_to_console("\n🔴 Training forcefully stopped!\n", "error")
            self.status_label.config(text="Training Stopped", foreground="red")
            self.training_finished()
            
        except Exception as e:
            self.append_to_console(f"\n❌ Error during force stop: {e}\n", "error")
            # Ensure UI is reset even if error occurs
            self.is_training = False
            self.training_finished()
        
    def clear_console(self):
        """Clear console output"""
        self.console_text.config(state='normal')
        self.console_text.delete(1.0, 'end')
        self.console_text.config(state='disabled')
        
        # Add welcome message
        self.append_to_console("🖥️  Console cleared. Ready for new session.\n\n", "normal")
        
    def save_console_log(self):
        """Save console log to file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        default_filename = f"training_log_{timestamp}.txt"
        
        filename = filedialog.asksaveasfilename(
            defaultextension=".txt",
            initialvalue=default_filename,
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")],
            title="Save Console Log"
        )
        
        if filename:
            try:
                with open(filename, 'w', encoding='utf-8') as f:
                    content = self.console_text.get(1.0, 'end')
                    # Add header to log file
                    f.write(f"YOLO Training Log\n")
                    f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                    f.write(f"Dataset: {self.dataset_path_var.get()}\n")
                    f.write(f"Model Path: {self.model_path_var.get()}\n")
                    f.write(f"Epochs: {self.epochs_var.get()}\n")
                    f.write(f"Device: {self.device_var.get()}\n")
                    f.write(f"{'='*60}\n\n")
                    f.write(content)
                messagebox.showinfo("Save Log", f"Console log saved to {filename}")
            except Exception as e:
                messagebox.showerror("Save Error", f"Failed to save log: {e}")
                
    def open_model_folder(self):
        """Open the model save folder in file explorer"""
        model_path = self.model_path_var.get()
        if os.path.exists(model_path):
            try:
                # Windows
                if sys.platform == "win32":
                    os.startfile(model_path)
                # macOS
                elif sys.platform == "darwin":
                    os.system(f"open '{model_path}'")
                # Linux
                else:
                    os.system(f"xdg-open '{model_path}'")
            except Exception as e:
                messagebox.showerror("Error", f"Could not open folder: {e}")
        else:
            messagebox.showwarning("Path Not Found", "Model save path does not exist!")

def main():
    """Main function to run the GUI"""
    # Configure ttk style
    root = tk.Tk()
    
    # Set style
    style = ttk.Style()
    
    # Try to use a modern theme
    try:
        style.theme_use('clam')  # or 'alt', 'default', 'classic'
    except:
        pass  # Use default theme if not available
        
    # Configure custom styles
    style.configure('Accent.TButton', foreground='white', background='#0078d4')
    
    # Create and run the application
    app = YOLOTrainingGUI(root)
    
    # Handle window closing
    def on_closing():
        """Handle window closing"""
        if app.is_training:
            result = messagebox.askyesno("Exit", 
                                       "Training is in progress. Do you want to:\n\n"
                                       "• Click 'Yes' to force stop training and exit\n"
                                       "• Click 'No' to cancel exit")
            if result:
                # Force stop training before closing
                app.stop_requested = True
                app.force_stop_training()
                root.after(1000, root.destroy)  # Wait 1 second then close
                return
            else:
                return  # Don't close
        root.destroy()
    
    root.protocol("WM_DELETE_WINDOW", on_closing)
    root.mainloop()

if __name__ == "__main__":
    main()