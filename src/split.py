# split_data.py (for all countries)

import os
import random
import shutil
import glob

# --- CONFIGURATION ---
# List of all country folders you want to include
country_folders = ['Czech', 'India', 'Japan', 'Norway', 'United_States']

# Path to the parent directory containing all the country folders
source_base_path = os.path.join('..', 'data')

# Path where the new combined dataset will be created
# We will create a new top-level folder to keep things clean
unified_dataset_path = os.path.join('..', 'data', 'All_Countries')

# The percentage of data to use for validation
val_split_ratio = 0.20

print("--- Starting Data Unification and Splitting ---")

# --- SCRIPT ---

# 1. Create the new unified directory structure
train_img_path = os.path.join(unified_dataset_path, 'images', 'train')
val_img_path = os.path.join(unified_dataset_path, 'images', 'val')
train_xml_path = os.path.join(unified_dataset_path, 'annotations', 'train')
val_xml_path = os.path.join(unified_dataset_path, 'annotations', 'val')

os.makedirs(train_img_path, exist_ok=True)
os.makedirs(val_img_path, exist_ok=True)
os.makedirs(train_xml_path, exist_ok=True)
os.makedirs(val_xml_path, exist_ok=True)

# 2. Loop through each country folder
for country in country_folders:
    print(f"\nProcessing country: {country}")
    
    # --- CORRECTED LINE ---
    # The path no longer has the country name repeated.
    country_train_path = os.path.join(source_base_path, country,country, 'train')
    original_img_dir = os.path.join(country_train_path, 'images')
    original_xml_dir = os.path.join(country_train_path, 'annotations', 'xmls')

    if not os.path.exists(original_img_dir):
        print(f"Warning: Could not find image directory for {country} at {original_img_dir}. Skipping.")
        continue

    # Get a list of all image files for the current country
    all_images = [os.path.basename(p) for p in glob.glob(os.path.join(original_img_dir, '*.jpg'))]
    random.shuffle(all_images)

    # Calculate the split index
    split_index = int(len(all_images) * val_split_ratio)
    val_images = all_images[:split_index]
    train_images = all_images[split_index:]

    print(f"  Found {len(all_images)} images. Splitting into {len(train_images)} train and {len(val_images)} val.")

    # Function to copy files to the new unified directories
    def copy_files(file_list, dest_img_path, dest_xml_path):
        for img_filename in file_list:
            base_name = os.path.splitext(img_filename)[0]
            xml_filename = f"{base_name}.xml"

            src_img = os.path.join(original_img_dir, img_filename)
            src_xml = os.path.join(original_xml_dir, xml_filename)

            if os.path.exists(src_img) and os.path.exists(src_xml):
                shutil.copy(src_img, os.path.join(dest_img_path, img_filename))
                shutil.copy(src_xml, os.path.join(dest_xml_path, xml_filename))

    # Copy the files for the current country
    copy_files(train_images, train_img_path, train_xml_path)
    copy_files(val_images, val_img_path, val_xml_path)

print("\n\n✅✅✅ Unification and splitting complete!")
print(f"Your new combined dataset is ready at: {unified_dataset_path}")
