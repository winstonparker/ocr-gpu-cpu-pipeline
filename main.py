#!/usr/bin/env python3
import os
import logging
import gc
from pathlib import Path
from multiprocessing import set_start_method
from cli import parse_cli_args, update_config_from_args
from config import CONFIG
from utils import load_completed
from pdf_processor import stage1
from ocr_processor import stage2
import cv2

def setup_logging():
    """
    Setup logging configuration.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler("ocr_progress.log"), logging.StreamHandler()]
    )

def main():
    """
    Main function to run the OCR Pipeline.
    Parses CLI arguments, updates configuration, and orchestrates Stage 1 and Stage 2.
    """
    args = parse_cli_args()
    update_config_from_args(args)

    # Optional Config, if provided
    tessdata_prefix = CONFIG.get("tessdata_prefix", "").strip()
    if len(tessdata_prefix) > 0:
        try:
            os.environ["TESSDATA_PREFIX"] = tessdata_prefix
        except Exception as e:
            logging.error(f"Error setting TESSDATA_PREFIX: {e}")

    setup_logging()

    # Create necessary directories
    Path(CONFIG["output_dir"]).mkdir(parents=True, exist_ok=True)
    Path(CONFIG["temp_dir"]).mkdir(parents=True, exist_ok=True)

    # Load completed PDF lists from previous runs if available
    completed_stage1 = load_completed("completed_stage1.json")
    completed_stage2 = load_completed("completed_stage2.json")

    if cv2.cuda.getCudaEnabledDeviceCount() > 0:
        logging.info(f"GPU Found, using GPU for Stage 1")
    else:
        logging.info(f"No GPU Found, using CPU only. If GPU support requested, please consult the README.")


    logging.info(f"Loaded Completed List of finished PDFs. Stage 1: {len(completed_stage1)} PDFs, Stage 2: {len(completed_stage2)} PDFs.")
    logging.info(f"Starting OCR on all unprocessed PDFs in directory: {CONFIG["pdf_dir"]}")

    pdf_files = [str(p) for p in Path(CONFIG["pdf_dir"]).glob("*.pdf")]
    logging.info(f"Total PDFs to process: {len(pdf_files)}")
    
    total_pages_target = 83000
    remaining_pdfs = len(pdf_files) - len(completed_stage2)
    
    while remaining_pdfs > 0:
        loop_start = os.times()[4]  # using system time in seconds
        completed_stage1 = stage1(pdf_files, completed_stage1, CONFIG)
        completed_stage2, loop_pages = stage2(completed_stage1, completed_stage2, CONFIG)
        remaining_pdfs = len(pdf_files) - len(completed_stage2)
        logging.info(f"Progress - Stage 1: {len(completed_stage1)} PDFs, Stage 2: {len(completed_stage2)} PDFs, Remaining: {remaining_pdfs} PDFs")
        elapsed = os.times()[4] - loop_start

        logging.info(f"Round Progress: {loop_pages} pages completed in {elapsed:.2f} seconds.")
        pages_done = sum(txt_file.read_text().count("---Page") - 1
                         for txt_file in Path(CONFIG["output_dir"]).glob("*.txt") if txt_file.exists())
        est_rem_pg = total_pages_target - pages_done
        est_rem_time_min = ((elapsed / loop_pages) * est_rem_pg) / 60 if loop_pages > 0 else 0
        logging.info(f"Overall Progress: {pages_done}/{total_pages_target} pages. Est: {est_rem_time_min:.2f} minutes remaining.")
        gc.collect()
    
    logging.info("All PDFs processed!")

if __name__ == "__main__":
    set_start_method('spawn', force=True)
    main()