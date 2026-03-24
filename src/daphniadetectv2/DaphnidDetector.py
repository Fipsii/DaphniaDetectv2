import os
import sys
import time
import pandas as pd
from contextlib import contextmanager
from tqdm import tqdm

# Import required modules from the CollectedCode package
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

    TOTAL_STEPS = 8

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

    # STEP 3: SEGMENTATION
    log_step(3, TOTAL_STEPS, "Segmenting Body...")
    with suppress_stdout():
        SegmentYOLODeploy.Segment_Exp(image_dir, output_dir, ModelPath=segment_model, Vis=True)
    log_done()

    # STEP 4: CLASSIFY SPECIES
    species = {}
    if classify_species_flag:
        log_step(4, TOTAL_STEPS, "Classifying Species...")
        daphnia_crop_dir = os.path.join(output_dir, "Detection", "crops", "Daphnia")
        with suppress_stdout():
            species = YOLODeploy.Classify_Species(daphnia_crop_dir, classify_model)
        log_done()
    else:
        log_step(4, TOTAL_STEPS, "Classifying Species [SKIPPED]")
        print()

    # STEP 5: MEASUREMENTS
    log_step(5, TOTAL_STEPS, "Measuring Body Width...")
    
    # Three Methods implemented Imhof, Rabus and Sperfeld
    Method = "Imhof"
    with suppress_stdout():
        measurements_dict = DataDict.BodyWidthMeasure(image_dir, output_dir, Method=Method)
    print(f"[Method: {Method}]")
    
    # STEP 6: GET SCALE VALUES
    log_step(6, TOTAL_STEPS, "Detecting Scales...")
    scale_detector_mode = 2
    with suppress_stdout():
        scales_df = ScaleDetect.Scale_Measure(measurements_dict, scale_detector_mode, Conv_factor=2.139)
    scales_dict = scales_df.set_index("image_name").to_dict(orient="index")
    
    combined_dict = {}
    for key in set(scales_dict) | set(measurements_dict):
        combined_dict[key] = {**scales_dict.get(key, {}), **measurements_dict.get(key, {})}
    log_done()
    # STEP 7: VISUALIZE RESULTS
    log_step(7, TOTAL_STEPS, "Exporting Visualizations...")
    vis_dir = os.path.join(output_dir, "visualization")
    with suppress_stdout():
        DataDict.visualize_and_save(combined_dict, vis_dir, scale_detector_mode)
    log_done()
    
    # STEP 8: MEASURE LENGTH & FORMAT DATA
    log_step(8, TOTAL_STEPS, "Formatting & Saving Arrays...")
    with suppress_stdout():
        final_measurements = LengthMeasure.MeasureLength(combined_dict)
        measurements_df = pd.DataFrame.from_dict(final_measurements, orient='index').reset_index()
        
        if 'image_name' in measurements_df.columns:
            measurements_df = measurements_df.rename(columns={'image_name': 'image_name_no_ext'})
        measurements_df.columns.values[0] = 'image_name'

        if species:
            species_df = pd.DataFrame(list(species.items()), columns=['image_name', 'species'])
            measurements_df = measurements_df.merge(species_df, on='image_name', how='left')
     
        measurements_df = SaveData.flatten_and_merge_data(measurements_df, confidence_data)
        SaveData.export_to_nested_json(measurements_df, output_dir + "/data.json")
        scaled_data = measurements_df.apply(LengthMeasure.scale_values, axis=1).apply(pd.Series)
        scaled_data.to_csv(os.path.join(output_dir, "scaled_measurements.csv"), index=False)
    log_done()     

    end_time = time.time()
    print("\n" + "="*50)
    print(f"[DONE] Pipeline completed in {end_time - start_time:.2f} seconds.")

if __name__ == "__main__":
    main()
