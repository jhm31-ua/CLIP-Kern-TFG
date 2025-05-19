#!/bin/bash

input_folder="my-datasets/MTD"
output_folder="my-datasets/MTD-custom"

if [ ! -d "$input_folder" ]; then
  echo "Input folder does not exist: $input_folder"
  exit 1
fi

if [ ! -d "$output_folder" ]; then
  echo "Output folder does not exist: $output_folder"
  exit 1
fi

echo "Cleaning $output_folder"
rm -f "$output_folder"/*.krn

for xml_file in "$input_folder"/*.xml; do
  if [ -f "$xml_file" ]; then
    base_name=$(basename "$xml_file" .xml)

    xml2hum "$xml_file" > "$output_folder/$base_name.krn"
    echo "Converted $xml_file to $output_folder/$base_name.krn"
  fi
done

echo "Conversion complete"
