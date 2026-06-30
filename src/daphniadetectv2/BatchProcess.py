import os
import sys
import time
import pandas as pd
from contextlib import contextmanager

from DetectorCode import (
    NMS_detect_Rezoom, SegmentYOLODeploy, YOLODeploy, 
    DataDict, ScaleDetect, LengthMeasure, ConvertToPNG, 
    SaveData, SpinaBaseRefine)

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

def process_folder(image_dir, output_dir, models):
    bbox_model, segment_model, classify_model, spina_model, classify_species_flag = models
    
    print("\n" + "="*50)
    print(f"[INIT] Input:  {image_dir}")
    print(f"[INIT] Output: {output_dir}")
    print("="*50 + "\n")

    TOTAL_STEPS = 8

    # STEP 1: CONVERT TO PNG
    log_step(1, TOTAL_STEPS, "Converting to PNG...")
    
    # Passing image_dir as the second argument leverages ConvertToPNG's internal 
    # logic to generate exactly image_dir/PNG without recursion.
    with suppress_stdout():
        ConvertToPNG.convert_to_png(image_dir, image_dir)
    
    current_image_dir = os.path.join(image_dir, "PNG")
    log_done()

    # STEP 2: DETECT ORGANS
    log_step(2, TOTAL_STEPS, "Detecting Organs...")
    with suppress_stdout():
        _, confidence_data = NMS_detect_Rezoom.DetectOrgans(
            current_image_dir, output_dir, 
            vis=True, NMS=True, refineTip=False,
            organs=["Heart", "Daphnia", "Eye", "Spina tip", "Spina base"], 
            ModelPath=bbox_model, SpinaModelPath=bbox_model, use_sahi=False
        )
    labels_dir = os.path.join(output_dir, "Detection", "labels")
    SpinaBaseRefine.Refine_spine_base(current_image_dir, labels_dir, spina_model)
    log_done()
    log_step(3, TOTAL_STEPS, "Segmenting Body...")
    with suppress_stdout():
        SegmentYOLODeploy.Segment_Exp(image_dir, output_dir, ModelPath=segment_model, Vis=True)
    log_done()

def main():
    global_start_time = time.time()
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    bbox_model = os.path.join(script_dir, "Model/detect/weights/best.pt")
    segment_model = os.path.join(script_dir, "Model/segment/daphnia_body/weights/NonObjectSeg.pt")
    classify_model = os.path.join(script_dir, "Model/classify/weights/best.pt")
    spina_model = os.path.join(script_dir, "Model/segment/spina_base/weights/SpinaBase.pt")
    classify_species_flag = True

    models = (bbox_model, segment_model, classify_model, spina_model, classify_species_flag)

    parent_dir = input("Enter Parent Directory containing image folders: ").strip()
    while not os.path.exists(parent_dir):
        parent_dir = input("Invalid. Enter Parent Directory path: ").strip()
    parent_dir = os.path.normpath(parent_dir)

    # Centralized superfolder for all batch outputs
    super_output_dir = f"{parent_dir}_Results"
    os.makedirs(super_output_dir, exist_ok=True)

    # Extract valid subdirectories
    subdirs = [
        os.path.join(parent_dir, d) for d in os.listdir(parent_dir) 
        if os.path.isdir(os.path.join(parent_dir, d)) 
        and not d.endswith("_Results")
        and d.upper() != "PNG"
    ]

    print(f"\nFound {len(subdirs)} folders to process.")

    for folder in subdirs:
        folder_name = os.path.basename(folder)
        # Route specific subfolder output into the superfolder
        output_dir = os.path.join(super_output_dir, folder_name)
        os.makedirs(output_dir, exist_ok=True)
        
        try:
            process_folder(folder, output_dir, models)
        except Exception as e:
            print(f"\n[ERROR] Pipeline failed on folder {folder}. Exception: {e}")
            continue

    global_end_time = time.time()
    print("\n" + "="*50)
    print(f"[DONE] Batch Pipeline completed in {global_end_time - global_start_time:.2f} seconds.")

if __name__ == "__main__":
    main()