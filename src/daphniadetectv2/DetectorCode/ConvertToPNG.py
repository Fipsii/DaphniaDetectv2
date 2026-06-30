import os
import shutil
from PIL import Image

def convert_to_png(source_dir, output_parent):
    if not os.path.exists(source_dir):
        print(f"Error: {source_dir} not found.")
        return

    save_folder = os.path.join(output_parent, "PNG")
    if os.path.exists(save_folder):
        shutil.rmtree(save_folder)
    os.makedirs(save_folder)

    for filename in os.listdir(source_dir):
        file_path = os.path.join(source_dir, filename)
        if not os.path.isfile(file_path):
            continue

        save_path = os.path.join(save_folder, os.path.splitext(filename)[0] + ".png")

        try:
            with Image.open(file_path) as im:
                # Handle transparency
                if im.mode in ("RGBA", "P"):
                    im = im.convert("RGBA")
                    bg = Image.new("RGB", im.size, (255, 255, 255))
                    bg.paste(im, mask=im.split()[3])
                    im = bg
                else:
                    im = im.convert("RGB")
                
                # STRIP METADATA: This removes the ICC profile and other junk data
                im.info = {} 
                
                # Now save
                im.save(save_path, "PNG", optimize=True)
        except (IOError, SyntaxError, Image.DecompressionBombError):
            continue

    print(f"Conversion complete. Files located in: {save_folder}")
    return save_folder