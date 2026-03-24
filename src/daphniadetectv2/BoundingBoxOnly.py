import os
import sys
import time
import pandas as pd
from contextlib import contextmanager
from tqdm import tqdm

from DetectorCode import (
    NMS_detect_Rezoom, SegmentYOLODeploy, YOLODeploy, 
    DataDict, ScaleDetect, LengthMeasure, ConvertToJPG, SaveData, SpinaBaseRefine
)

@contextmanager
def suppress_stdout():
    """Context manager to route standard output to devnull, silencing internal module prints."""
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout

def log_step(step_num, total_steps, description):
    """Standardized terminal output for pipeline steps."""
    print(f"[{step_num}/{total_steps}] {description:<30} ", end="", flush=True)

def log_done():
    print("[Complete]")

def main():
    start_time = time.time()
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    bbox_model = os.path.join(script_dir, "Model/detect/weights/best.pt")
    segment_model = os.path.join(script_dir, "Model/segment/daphnia_body/weights/NonObjectSeg.pt")
    classify_model = os.path.join(script_dir, "Model/classify/weights/best.pt")
    spina_model = os.path.join(script_dir, "Model/segment/spina_base/weights/SpinaBase.pt")
    classify_species_flag = True

    # 1. Resolve Directories
    image_dir = input("Enter ImageDir path: ").strip()
    while not os.path.exists(image_dir):
        image_dir = input("Invalid. Enter ImageDir path: ").strip()
    image_dir = os.path.normpath(image_dir)

    custom_output = input("Enter OutputDir (or Enter for default): ").strip()
    output_dir = custom_output if custom_output else f"{image_dir}_Results"
    os.makedirs(output_dir, exist_ok=True)

    print("\n" + "="*50)
    print(f"[INIT] Input:  {image_dir}")
    print(f"[INIT] Output: {output_dir}")
    print("="*50 + "\n")

    TOTAL_STEPS = 3

    # STEP 1: CONVERT TO JPG
    log_step(1, TOTAL_STEPS, "Converting to JPG...")
    jpg_dir = os.path.join(image_dir, "JPG")
    with suppress_stdout():
        ConvertToJPG.convert_to_jpeg(image_dir, image_dir)
    image_dir = jpg_dir
    log_done()

    # STEP 2: DETECT ORGANS
    log_step(2, TOTAL_STEPS, "Detecting Organs...")
    with suppress_stdout():
        _, confidence_data = NMS_detect_Rezoom.DetectOrgans(
            image_dir, output_dir, 
            vis=True, NMS=True, refineTip=False,
            organs=["Heart", "Daphnia", "Eye", "Spina tip", "Spina base"], 
            ModelPath=bbox_model, SpinaModelPath=bbox_model, use_sahi=False
        )
    labels_dir = os.path.join(output_dir, "Detection", "labels")
    SpinaBaseRefine.Refine_spine_base(image_dir, labels_dir, spina_model)
    log_done()

    # STEP 3: PROCESS BBOX DATA
    log_step(3, TOTAL_STEPS, "Parsing Annotations...")
    with suppress_stdout():
        BoundingBoxAnnotations = SaveData.read_yolo_folder(labels_dir, image_dir)
        BoundingBoxAnnotationsPixel = SaveData.convert_yolo_to_pixel(BoundingBoxAnnotations)
        
    # Save as JSON
    # Save the flat pixel annotations as JSON
    output_json_path = os.path.join(output_dir, "data.json")
    measurements_df = SaveData.flatten_and_merge_data(BoundingBoxAnnotationsPixel, confidence_data)
    SaveData.export_to_nested_json(measurements_df, output_json_path)
    log_done()
    print(f"\n[PIPELINE COMPLETE] Time elapsed: {time.time() - start_time:.2f} seconds")

if __name__ == "__main__":
    main()