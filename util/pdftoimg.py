from pdf2image import convert_from_path
import os

def convert_pdf_to_png(input_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir) 

    for filename in os.listdir(input_dir):
        if filename.endswith('.pdf'):
            pdf_path = os.path.join(input_dir, filename)  
            output_path = os.path.join(output_dir, f'{os.path.splitext(filename)[0]}.png')  

            try:
                images = convert_from_path(pdf_path)
                for i, img in enumerate(images):
                    img_path = f'{output_path[:-4]}_{i}.png' if i > 0 else output_path  
                    img.save(img_path, 'PNG')

            except Exception as e:
                print(f'Error converting {pdf_path}: {e}')

            else:
                print(f'Successfully converted {pdf_path} to {output_path}')
           
input_dir="my-datasets/data_SCORE_IMG"
output_dir="my-datasets/MTD-custom"
convert_pdf_to_png(input_dir, output_dir)
