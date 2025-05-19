import os
import shutil
import re

replacement_dict = {
    "q0.664062": "6",
    "q0.667969": "6",
    "q0.332031": "12",
    "q0.335938": "12",
    "q0.285156": "14",
    "q0.289062": "14",
    "q0.199219": "20",
    "q0.203125": "20",
    "q0.164062": "24",
    "q0.167969": "24",
    "q0.140625": "28",
    "q0.144531": "28",
    "q0.097656": "40",
    "q0.101562": "40",
    "q0.082031": "48",
    "q0.085938": "48",
    "q0.070312": "56",
    "q0.074219": "56",
    "q0.042969": "96"
}

source_folder = "my-datasets/MTD-custom"
destination_folder = "my-datasets/MTD-custom-noq"

if os.path.exists(destination_folder):
    shutil.rmtree(destination_folder)
os.makedirs(destination_folder)

triplet_count = 0 
checked_files = 0  

for filename in os.listdir(source_folder):
    if filename.endswith(".krn"):
        checked_files += 1
        krn_path = os.path.join(source_folder, filename)
        
        with open(krn_path, "r", encoding="utf-8") as file:
            content = file.read()
            
            q0_matches = re.findall(r"q0\.\d+", content)
            
            multiple_dots_matches = re.findall(r"\d+\.\.\.+", content)
            if multiple_dots_matches:
                print(f"{filename} has digits with multiple dots: {set(multiple_dots_matches)}")
            
            if not q0_matches:
                shutil.copy(krn_path, destination_folder)

                png_filename = filename.replace(".krn", ".png")
                png_path = os.path.join(source_folder, png_filename)
                if os.path.exists(png_path):
                    shutil.copy(png_path, destination_folder)
                else:
                    print(f"Copied {filename}, but {png_filename} not found")
                continue
            
            if any(match not in replacement_dict for match in q0_matches):
                triplet_count += 1
                print(f"{filename} has unsupported triplets: {set(q0_matches) - set(replacement_dict.keys())}")
                continue

            for match in q0_matches:
                content = content.replace(match, replacement_dict[match])
            
            fixed_krn_path = os.path.join(destination_folder, filename)
            with open(fixed_krn_path, "w", encoding="utf-8") as fixed_file:
                fixed_file.write(content)
            
            png_filename = filename.replace(".krn", ".png")
            png_path = os.path.join(source_folder, png_filename)
            if os.path.exists(png_path):
                shutil.copy(png_path, destination_folder)
            else:
                print(f"Copied {filename}, but {png_filename} not found")

print(f"\nProcess completed. Checked {checked_files} files. {triplet_count} files contained unsupported triplets.")
