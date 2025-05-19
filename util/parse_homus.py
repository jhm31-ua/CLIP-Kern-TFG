import os
from PIL import Image, ImageDraw

CANVAS_SIZE = (128, 128) 

def parse_file(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()

    class_name = lines[0].strip()  
    strokes = []

    for line in lines[1:]:  
        points = []
        for point_text in line.strip().split(';'):
            if point_text:
                x, y = map(int, point_text.split(','))
                points.append((x, y))
        if points:
            strokes.append(points)  

    return class_name, strokes

def get_bounding_box(strokes):
    all_x = [x for stroke in strokes for x, y in stroke]
    all_y = [y for stroke in strokes for x, y in stroke]
    return min(all_x), min(all_y), max(all_x), max(all_y)

def center_strokes(strokes, canvas_size):
    min_x, min_y, max_x, max_y = get_bounding_box(strokes)

    stroke_center_x = (min_x + max_x) // 2
    stroke_center_y = (min_y + max_y) // 2

    canvas_center_x, canvas_center_y = canvas_size[0] // 2, canvas_size[1] // 2

    offset_x = canvas_center_x - stroke_center_x
    offset_y = canvas_center_y - stroke_center_y

    centered_strokes = [
        [(x + offset_x, y + offset_y) for x, y in stroke] 
        for stroke in strokes
    ]

    return centered_strokes

def draw_strokes(strokes, img_size, line_color=(0, 0, 0), line_width=2):
    image = Image.new('RGB', img_size, "white") 
    draw = ImageDraw.Draw(image)

    for stroke in strokes:
        if len(stroke) > 1:
            draw.line(stroke, fill=line_color, width=line_width)
        elif len(stroke) == 1: 
            x, y = stroke[0]
            draw.ellipse([x - 1, y - 1, x + 1, y + 1], fill=line_color)

    return image

def process_folder(input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)

    file_counter = 0  
    for root, _, files in os.walk(input_folder):  
        for file_name in sorted(files):  
            if file_name.endswith(".txt"): 
                file_path = os.path.join(root, file_name)

                class_name, strokes = parse_file(file_path)

                centered_strokes = center_strokes(strokes, CANVAS_SIZE)

                class_output_path = os.path.join(output_folder, f"{file_counter}.txt")
                with open(class_output_path, 'w') as class_file:
                    class_file.write(class_name)

                image = draw_strokes(centered_strokes, CANVAS_SIZE)
                image_output_path = os.path.join(output_folder, f"{file_counter}.png")
                image.save(image_output_path)

                print(f"Processed {file_path} -> {file_counter}.txt, {file_counter}.png")

                file_counter += 1 

input_folder = "my-datasets/HOMUS"
output_folder = "my-datasets/HOMUS-parsed"
process_folder(input_folder, output_folder)