import pandas as pd
import os
from flux.cli_kontext import main as flux_kontext_main

# --- CONFIGURATION ---
CSV_FILE = "/data/local/nemmler/data/ddi/ddi_metadata.csv"
GLOBAL_PROMPT = ""
OUTPUT_FOLDER = "output/flux/test"
DEVICE = "cuda:0"
# ---------------------

def run_filtered_batch(file_name):
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    # 1. Load the CSV with Pandas
    df = pd.read_csv(CSV_FILE)

    # 2. FILTERING LOGIC
    # Example A: Only images where a 'status' column is 'active'
    filtered_df = df[df['skin_tone'] == 56]

    print(f"Found {len(df)} rows. Filtered down to {len(filtered_df)} images.")
    filtered_df = filtered_df.rename(columns={'file_name':'image_path'})
    # 3. Process the filtered list
    for i, row in filtered_df.iterrows():
        input_path = os.path.join("data/ddi", row['image_path'])
        print(f"\n>>> Processing image {i+1}/{len(filtered_df)}: {input_path}")
        try:
            flux_kontext_main(
                prompt="",
                img_cond_path=input_path,
                device=DEVICE,
                offload=True,
                output_dir="output",
                loop=False, 
                seed=1234
            )
        except Exception as e:
            print(f"Error on {input_path}: {e}")

def run():
    flux_kontext_main(
        prompt="",
        img_cond_path="/data/local/nemmler/data/ddi/000001.png",
        #device="cuda",
        offload=True,
        output_dir="output",
        loop=False, 
        seed=1234
    )
if __name__ == "__main__":
    #run_filtered_batch('DDI_file')
    run()
