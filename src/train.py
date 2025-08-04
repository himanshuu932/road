# train.py (for the combined 'All_Countries' dataset)

import os
import torch
from ultralytics import YOLO

def main():
    """
    Main function to run the YOLOv8 training process on the combined dataset.
    """
    print("--- Starting YOLOv8 Training on Combined Dataset ---")

    # --- Device Configuration ---
    if torch.cuda.is_available():
        device = 'cuda:0'
        print(f"✅ Found compatible GPU. Using device: {device}")
        print(f"   GPU Name: {torch.cuda.get_device_name(0)}")
    else:
        device = 'cpu'
        print("⚠️ WARNING: No compatible NVIDIA GPU found. Training will use the CPU.")

    # --- Dataset and Model Configuration ---
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Path to your new dataset configuration file in the project root
    data_config_path = os.path.join(script_dir, '..', 'rdd_all.yaml')
    
    # Using the more powerful 'small' model for better accuracy
    model_name = 'yolov8s.pt'
    
    # Training parameters
    num_epochs = 100
    # Batch size is lowered because yolov8s uses more GPU memory.
    # If you get an 'out of memory' error, try reducing this to 4.
    batch_size = 8
    
    image_size = 640

    # --- Load Model ---
    try:
        model = YOLO(model_name)
        print(f"Successfully loaded model: {model_name}")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # --- Start Training ---
    print("\nStarting model training...")
    print(f"  Model: {model_name}")
    print(f"  Dataset: {data_config_path}")
    print(f"  Epochs: {num_epochs}")
    print(f"  Batch Size: {batch_size}")
    
    try:
        results = model.train(
            data=data_config_path,
            epochs=num_epochs,
            batch=batch_size,
            imgsz=image_size,
            device=device,
            name='yolov8s_all_countries_custom' # A new name for the output folder
        )
        
        best_model_path = os.path.join(results.save_dir, 'weights', 'best.pt')
        print(f"\n✅ Training complete!")
        print(f"The best model has been saved to: {best_model_path}")

    except Exception as e:
        print(f"\nAn error occurred during training: {e}")
        print("If this is an 'out of memory' error, try reducing the 'batch_size' in this script.")

if __name__ == '__main__':
    main()
