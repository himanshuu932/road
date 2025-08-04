# convert.py (for all countries)

import xml.etree.ElementTree as ET
import os
import glob

# --- CONFIGURATION ---
# The complete list of all damage classes from all countries in the dataset
class_names = [
    'D00', 'D01', 'D10', 'D11', 'D20', 'D40', 'D43', 'D44', 'D50'
]

# --- ROBUST PATHS ---
# Get the directory where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))
# Build the path to the 'data' folder, which is one level up from 'src'
base_path = os.path.join(script_dir, '..', 'data', 'All_Countries')

# --- SCRIPT ---
def convert_voc_to_yolo(xml_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    xml_files = glob.glob(os.path.join(xml_dir, '*.xml'))
    print(f"Found {len(xml_files)} XML files in '{xml_dir}'.")
    error_count = 0

    for xml_file in xml_files:
        try:
            tree = ET.parse(xml_file)
            root = tree.getroot()

            size = root.find('size')
            if size is None:
                continue
                
            img_width = int(size.find('width').text)
            img_height = int(size.find('height').text)

            yolo_lines = []
            for obj in root.findall('object'):
                class_name = obj.find('name').text
                if class_name not in class_names:
                    continue
                
                bndbox = obj.find('bndbox')
                if bndbox is None:
                    continue

                # --- CORRECTED LINE ---
                # Some XML files have float values for coordinates. Convert to float first, then to int.
                xmin, ymin, xmax, ymax = [int(float(b.text)) for b in bndbox]
                
                class_id = class_names.index(class_name)

                x_center = (xmin + xmax) / 2.0 / img_width
                y_center = (ymin + ymax) / 2.0 / img_height
                width = (xmax - xmin) / img_width
                height = (ymax - ymin) / img_height

                yolo_lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")

            base_filename = os.path.splitext(os.path.basename(xml_file))[0]
            output_filepath = os.path.join(output_dir, f"{base_filename}.txt")
            
            with open(output_filepath, 'w') as f:
                f.write('\n'.join(yolo_lines))
        except Exception as e:
            # If an error occurs, print it and move to the next file
            # print(f"Error processing file {os.path.basename(xml_file)}: {e}")
            error_count += 1
            continue
            
    print(f"Conversion complete. Skipped {error_count} files due to errors.")
    print(f"TXT files saved in '{output_dir}'.\n")

# --- Define Paths and Run Conversion for the Unified Dataset ---
print("--- Converting Unified Training Set ---")
train_xml_dir = os.path.join(base_path, 'annotations', 'train')
train_output_dir = os.path.join(base_path, 'labels', 'train')
convert_voc_to_yolo(train_xml_dir, train_output_dir)

print("--- Converting Unified Validation Set ---")
val_xml_dir = os.path.join(base_path, 'annotations', 'val')
val_output_dir = os.path.join(base_path, 'labels', 'val')
convert_voc_to_yolo(val_xml_dir, val_output_dir)

print("\n✅ All annotations have been converted to YOLO format.")
