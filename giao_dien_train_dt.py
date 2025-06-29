import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import threading
import os
import sys
from pathlib import Path
import subprocess
import json
from datetime import datetime
import queue
import time

# Import your training module
try:
    from traindata_dienthoaiYOLO import PhoneDetectionTrainer
except ImportError:
    print("Warning: traindata_dienthoaiYOLO module not found. Some features may not work.")

class YOLOTrainingGUI:    
    def __init__(self, root):
        self.root = root
        self.root.title("YOLO Phone Detection Training Interface")
        self.root.geometry("1000x700")
        self.root.configure(bg='#f0f0f0')
        
        # Variables
        self.data_path = tk.StringVar(value="data")
        self.output_path = tk.StringVar(value="runs/detect")
        self.epochs = tk.IntVar(value=100)
        self.batch_size = tk.IntVar(value=32)
        self.img_size = tk.IntVar(value=640)
        self.use_gpu = tk.BooleanVar(value=True)
        self.is_training = False
        self.is_stopping = False
        self.trainer = None
        
        # Initialize logging
        self.log_text = None
        self.log_queue = queue.Queue()
        self.auto_scroll = tk.BooleanVar(value=True)
        
        # Style configuration
        self.setup_styles()
        
        # Create GUI
        self.create_widgets()
        
        # Start log queue checker
        self.check_log_queue()
        
    def setup_styles(self):
        """Configure custom styles for the GUI"""
        style = ttk.Style()
        style.theme_use('clam')
        
        # Configure button styles
        style.configure('Action.TButton', 
                       font=('Arial', 10, 'bold'),
                       padding=(10, 5))
        
        # Configure frame styles
        style.configure('Card.TFrame',
                       background='white',
                       relief='raised',
                       borderwidth=1)
        
        # Configure label styles
        style.configure('Title.TLabel',
                       font=('Arial', 14, 'bold'),
                       background='white')
        
        style.configure('Header.TLabel',
                       font=('Arial', 12, 'bold'),
                       background='#f0f0f0')
    
    def create_widgets(self):
        """Create and arrange GUI widgets"""
        # Main container
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        
        # Title
        title_frame = ttk.Frame(main_frame)
        title_frame.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 20))
        
        title_label = ttk.Label(title_frame, 
                               text="🚀 YOLO Phone Detection Training Interface",
                               font=('Arial', 16, 'bold'),
                               foreground='#2c3e50')
        title_label.pack()
        
        # Left panel - Configuration
        config_frame = ttk.LabelFrame(main_frame, text="Training Configuration", 
                                     padding="15", style='Card.TFrame')
        config_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), 
                         padx=(0, 10), pady=(0, 10))
        
        self.create_config_widgets(config_frame)
        
        # Right panel - Controls and Status
        control_frame = ttk.Frame(main_frame)
        control_frame.grid(row=1, column=1, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Control buttons
        button_frame = ttk.LabelFrame(control_frame, text="Training Controls", 
                                     padding="15", style='Card.TFrame')
        button_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        
        self.create_control_widgets(button_frame)
        
        # Progress and status
        status_frame = ttk.LabelFrame(control_frame, text="Training Status", 
                                     padding="10", style='Card.TFrame')
        status_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        control_frame.rowconfigure(1, weight=1)
        
        self.create_status_widgets(status_frame)
        
        # Bottom panel - Log output
        log_frame = ttk.LabelFrame(main_frame, text="Training Log", 
                                  padding="10", style='Card.TFrame')
        log_frame.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), 
                      pady=(10, 0))
        main_frame.rowconfigure(2, weight=1)
        
        self.create_log_widgets(log_frame)
    
    def create_config_widgets(self, parent):
        """Create configuration input widgets"""
        # Dataset Path
        ttk.Label(parent, text="Dataset Path:", font=('Arial', 10, 'bold')).grid(
            row=0, column=0, sticky=tk.W, pady=(0, 5))
        
        path_frame = ttk.Frame(parent)
        path_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(0, 15))
        path_frame.columnconfigure(0, weight=1)
        
        ttk.Entry(path_frame, textvariable=self.data_path, width=30).grid(
            row=0, column=0, sticky=(tk.W, tk.E), padx=(0, 5))
        ttk.Button(path_frame, text="Browse", 
                  command=self.browse_dataset).grid(row=0, column=1)
        
        # Training Parameters
        params_frame = ttk.LabelFrame(parent, text="Training Parameters", padding="10")
        params_frame.grid(row=2, column=0, sticky=(tk.W, tk.E), pady=(0, 15))
        
        # Epochs
        ttk.Label(params_frame, text="Epochs:").grid(row=0, column=0, sticky=tk.W, pady=2)
        epochs_spin = ttk.Spinbox(params_frame, from_=1, to=1000, 
                                 textvariable=self.epochs, width=10)
        epochs_spin.grid(row=0, column=1, sticky=tk.W, padx=(5, 0), pady=2)
        
        # Batch Size
        ttk.Label(params_frame, text="Batch Size:").grid(row=1, column=0, sticky=tk.W, pady=2)
        batch_spin = ttk.Spinbox(params_frame, from_=1, to=128, 
                                textvariable=self.batch_size, width=10)
        batch_spin.grid(row=1, column=1, sticky=tk.W, padx=(5, 0), pady=2)
        
        # Image Size
        ttk.Label(params_frame, text="Image Size:").grid(row=2, column=0, sticky=tk.W, pady=2)
        size_combo = ttk.Combobox(params_frame, values=[320, 416, 512, 640, 832], 
                                 textvariable=self.img_size, width=8, state="readonly")
        size_combo.grid(row=2, column=1, sticky=tk.W, padx=(5, 0), pady=2)
        
        # GPU Usage
        ttk.Checkbutton(params_frame, text="Use GPU (CUDA)", 
                       variable=self.use_gpu).grid(row=3, column=0, columnspan=2, 
                                                  sticky=tk.W, pady=(10, 5))
        
        # Dataset Info
        info_frame = ttk.LabelFrame(parent, text="Dataset Information", padding="10")
        info_frame.grid(row=3, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        
        self.dataset_info = ttk.Label(info_frame, text="Click 'Verify Dataset' to check data structure",
                                     wraplength=250, foreground='#7f8c8d')
        self.dataset_info.pack(anchor=tk.W)
        
        # Verify button
        ttk.Button(parent, text="🔍 Verify Dataset", 
                  command=self.verify_dataset,
                  style='Action.TButton').grid(row=4, column=0, pady=(0, 10))
    
    def create_control_widgets(self, parent):
        """Create control button widgets"""
        # Start Training Button
        self.start_btn = ttk.Button(parent, text="🚀 Start Training", 
                                   command=self.start_training,
                                   style='Action.TButton')
        self.start_btn.grid(row=0, column=0, pady=5, padx=5, sticky=(tk.W, tk.E))
        
        # Stop Training Button
        self.stop_btn = ttk.Button(parent, text="⏹️ Stop Training", 
                                  command=self.stop_training,
                                  style='Action.TButton',
                                  state='disabled')
        self.stop_btn.grid(row=0, column=1, pady=5, padx=5, sticky=(tk.W, tk.E))
          
        # Configure grid weights
        parent.columnconfigure(0, weight=1)
        parent.columnconfigure(1, weight=1)
    
    def create_status_widgets(self, parent):
        """Create status display widgets"""
        # Progress bar
        ttk.Label(parent, text="Training Progress:", font=('Arial', 10, 'bold')).pack(anchor=tk.W)
        
        self.progress_var = tk.StringVar(value="Ready to start training")
        self.progress_label = ttk.Label(parent, textvariable=self.progress_var,
                                       foreground='#27ae60')
        self.progress_label.pack(anchor=tk.W, pady=(0, 10))
        
        self.progress_bar = ttk.Progressbar(parent, mode='indeterminate')
        self.progress_bar.pack(fill=tk.X, pady=(0, 15))
        
        # System info
        system_frame = ttk.LabelFrame(parent, text="System Information", padding="10")
        system_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.system_info = ttk.Label(system_frame, text=self.get_system_info(),
                                    font=('Courier', 9))
        self.system_info.pack(anchor=tk.W)
        
        # Model info
        model_frame = ttk.LabelFrame(parent, text="Model Information", padding="10")
        model_frame.pack(fill=tk.X)
        
        self.model_info = ttk.Label(model_frame, text="No model trained yet",
                                   foreground='#95a5a6')
        self.model_info.pack(anchor=tk.W)
    def create_log_widgets(self, parent):
        """Create log display widgets"""
        # Log text area
        self.log_text = scrolledtext.ScrolledText(parent, height=15, 
                                                 font=('Courier', 9),
                                                 background='#2c3e50',
                                                 foreground='#ecf0f1',
                                                 insertbackground='white')
        self.log_text.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        # Log controls
        log_controls = ttk.Frame(parent)
        log_controls.pack(fill=tk.X)
        
        # Add clear log button
        self.clear_log_btn = ttk.Button(log_controls, text="Clear Log", 
                                       command=lambda: self.clear_log())
        self.clear_log_btn.pack(side=tk.LEFT, padx=(0, 10))
        
        # Add save log button
        self.save_log_btn = ttk.Button(log_controls, text="Save Log", 
                                      command=lambda: self.save_log())
        self.save_log_btn.pack(side=tk.LEFT)
        
        # Auto-scroll checkbox
        self.auto_scroll = tk.BooleanVar(value=True)
        ttk.Checkbutton(log_controls, text="Auto Scroll", 
                       variable=self.auto_scroll).pack(side=tk.RIGHT)
    
    def get_system_info(self):
        """Get system information"""
        try:
            import torch
            cuda_available = torch.cuda.is_available()
            cuda_device = torch.cuda.get_device_name(0) if cuda_available else "N/A"
            cuda_count = torch.cuda.device_count() if cuda_available else 0
            
            info = f"GPU: {'Available' if cuda_available else 'Not Available'}\n"
            if cuda_available:
                info += f"Device: {cuda_device}\n"
                info += f"Count: {cuda_count}\n"
            info += f"PyTorch: {torch.__version__}"
            
        except ImportError:
            info = "PyTorch not installed\nGPU information unavailable"
        
        return info
    
    def browse_dataset(self):
        """Browse for dataset folder"""
        folder = filedialog.askdirectory(title="Select Dataset Folder")
        if folder:
            self.data_path.set(folder)
            self.log_message(f"Dataset path set to: {folder}")
    
    def verify_dataset(self):
        """Verify dataset structure"""
        try:
            data_folder = Path(self.data_path.get())
            
            if not data_folder.exists():
                self.dataset_info.config(text="❌ Dataset folder does not exist", 
                                       foreground='red')
                return
            
            # Check required folders
            required_dirs = ["train/images", "train/labels", "val/images", "val/labels"]
            status = []
            
            for dir_path in required_dirs:
                full_path = data_folder / dir_path
                if full_path.exists():
                    count = len([f for f in full_path.glob("*") if f.is_file()])
                    status.append(f"✅ {dir_path}: {count} files")
                else:
                    status.append(f"❌ {dir_path}: Missing")
            
            info_text = "\n".join(status)
            self.dataset_info.config(text=info_text, foreground='black')
            self.log_message("Dataset structure verified")
            
        except Exception as e:
            self.dataset_info.config(text=f"❌ Error: {str(e)}", foreground='red')
            self.log_message(f"Dataset verification failed: {e}")
    
    def start_training(self):
        """Start training in separate thread"""
        if self.is_training:
            return
        
        # Validate inputs
        if not Path(self.data_path.get()).exists():
            messagebox.showerror("Error", "Dataset path does not exist!")
            return
        
        self.is_training = True
        self.start_btn.config(state='disabled')
        self.stop_btn.config(state='normal')
        self.progress_bar.start()
        self.progress_var.set("Initializing training...")
        
        # Start training thread
        training_thread = threading.Thread(target=self.training_worker)
        training_thread.daemon = True
        training_thread.start()
        
        self.log_message("🚀 Training started!")
    def training_worker(self):
        """Training worker thread"""
        try:
            # Initialize trainer
            self.trainer = PhoneDetectionTrainer(
                data_path=self.data_path.get(),
                output_path=self.output_path.get(),
                use_gpu=self.use_gpu.get()
            )
            
            self.log_queue.put("Trainer initialized successfully")
            self.is_stopping = False
            
            # Start training
            results = self.trainer.train_model(
                epochs=self.epochs.get(),
                img_size=self.img_size.get(),
                batch_size=self.batch_size.get()
            )
            
            if results:
                if self.is_stopping:
                    self.log_queue.put("⏹️ Training stopped by user")
                    self.log_queue.put("TRAINING_STOPPED")
                else:
                    self.log_queue.put("✅ Training completed successfully!")
                    self.log_queue.put("TRAINING_COMPLETE")
            else:
                if self.is_stopping:
                    self.log_queue.put("⏹️ Training stopped by user")
                    self.log_queue.put("TRAINING_STOPPED")
                else:
                    self.log_queue.put("❌ Training failed!")
                    self.log_queue.put("TRAINING_FAILED")
                
        except KeyboardInterrupt:
            self.log_queue.put("⏹️ Training stopped by user")
            self.log_queue.put("TRAINING_STOPPED")
        except Exception as e:
            self.log_queue.put(f"❌ Training error: {str(e)}")
            self.log_queue.put("TRAINING_FAILED")
        finally:
            # Reset UI state
            self.is_training = False
            self.is_stopping = False
            self.root.after(0, lambda: self.start_btn.config(state='normal'))
            self.root.after(0, lambda: self.stop_btn.config(state='disabled'))
            self.root.after(0, lambda: self.progress_bar.stop())
            if hasattr(self.trainer, 'stop_training') and self.trainer.stop_training:
                self.root.after(0, lambda: self.progress_var.set("Training stopped by user"))
            self.trainer = None  # Clear trainer reference
    
    def stop_training(self):
        """Handle stop training button click"""
        if self.trainer and self.is_training:
            self.is_stopping = True
            success = self.trainer.stop()
            if success:
                self.log_message("⏹️ Stopping training... Please wait for the current epoch to finish.")
                self.stop_btn.config(state='disabled')
                self.progress_var.set("Stopping training...")
            else:
                self.log_message("❌ Could not stop training. No active training session found.")
  
    def log_message(self, message):
        """Add message to log and update display"""
        if not hasattr(self, 'log_text') or not self.log_text:
            # If log widget isn't ready yet, just queue the message
            self.log_queue.put(message)
            return
            
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_entry = f"[{timestamp}] {message}\n"
        
        # Use after() to ensure thread safety
        self.root.after(0, lambda: self._update_log_display(log_entry))
        
    def _update_log_display(self, message):
        """Update log display (should be called from main thread)"""
        if self.log_text:
            self.log_text.insert(tk.END, message)
            if self.auto_scroll.get():
                self.log_text.see(tk.END)
        
    def clear_log(self):
        """Clear the log text area"""
        if self.log_text:
            self.log_text.delete('1.0', tk.END)
            self.log_message("Log cleared")
    def save_log(self):
        """Save log contents to a file"""
        if not self.log_text:
            messagebox.showerror("Error", "Log not initialized yet")
            return
            
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = filedialog.asksaveasfilename(
                defaultextension=".txt",
                initialfile=f"training_log_{timestamp}.txt",
                title="Save Log File",
                filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
            )
            
            if filename:
                with open(filename, 'w', encoding='utf-8') as f:
                    log_content = self.log_text.get('1.0', tk.END)
                    f.write(log_content)
                self.log_message(f"✅ Log saved to: {filename}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save log: {str(e)}")
            self.log_message(f"❌ Error saving log: {str(e)}")
    
    def check_log_queue(self):
        """Check for log messages from worker thread"""
        try:
            while True:
                message = self.log_queue.get_nowait()
                
                # Handle special status messages
                if message == "TRAINING_COMPLETE":
                    self.progress_var.set("Training completed successfully")
                    self.progress_bar.stop()
                    self.start_btn.config(state='normal')
                    self.stop_btn.config(state='disabled')
                    self.is_training = False
                elif message == "TRAINING_FAILED":
                    self.progress_var.set("Training failed")
                    self.progress_bar.stop()
                    self.start_btn.config(state='normal')
                    self.stop_btn.config(state='disabled')
                    self.is_training = False
                elif message == "TRAINING_STOPPED":
                    self.progress_var.set("Training stopped by user")
                    self.progress_bar.stop()
                    self.start_btn.config(state='normal')
                    self.stop_btn.config(state='disabled')
                    self.is_training = False
                    self.is_stopping = False
                else:
                    # Regular log message
                    self.log_message(message)
                    
        except queue.Empty:
            pass
        finally:
            # Schedule next check
            self.root.after(100, self.check_log_queue)
    
    def update_model_info(self):
        """Update model information display"""
        try:
            model_path = Path("model/phone_detection5/weights/best.pt")
            if model_path.exists():
                size = model_path.stat().st_size / (1024 * 1024)  # MB
                modified = datetime.fromtimestamp(model_path.stat().st_mtime)
                
                info = f"✅ Model Available\n"
                info += f"Size: {size:.1f} MB\n"
                info += f"Created: {modified.strftime('%Y-%m-%d %H:%M:%S')}\n"
                info += f"Path: {model_path}"
                
                self.model_info.config(text=info, foreground='#27ae60')
            else:
                self.model_info.config(text="No model available", foreground='#95a5a6')
        except Exception as e:
            self.model_info.config(text=f"Error: {str(e)}", foreground='red')

def main():
    """Main function to run the GUI"""
    root = tk.Tk()
    app = YOLOTrainingGUI(root)
    
    # Handle window closing
    def on_closing():
        if app.is_training:
            if messagebox.askokcancel("Quit", "Training is in progress. Do you want to quit?"):
                app.stop_training()
                root.quit()
        else:
            root.quit()
    
    root.protocol("WM_DELETE_WINDOW", on_closing)
    
    # Center window
    root.update_idletasks()
    x = (root.winfo_screenwidth() // 2) - (root.winfo_width() // 2)
    y = (root.winfo_screenheight() // 2) - (root.winfo_height() // 2)
    root.geometry(f"+{x}+{y}")
    
    root.mainloop()

if __name__ == "__main__":
    main()