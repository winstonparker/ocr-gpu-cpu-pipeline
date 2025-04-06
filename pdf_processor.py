import os
import time
import gc
import logging
import shutil
from pathlib import Path
import fitz  # PyMuPDF
import numpy as np
import cv2
from PIL import Image
from preprocessing import preprocess_image_batch
from utils import load_completed, save_completed
from functools import partial
from multiprocessing import Pool
from tqdm import tqdm

def process_pdf(pdf_path, config):
    """
    Convert a PDF into preprocessed images, one per page.
    Saves intermediate images as numpy arrays in a temporary directory.
    
    Args:
        pdf_path (str): Path to the PDF file.
        config (dict): Configuration dictionary.
        
    Returns:
        tuple: (pdf_name, number of pages processed, average conversion time per page in ms)
    """
    pdf_name = Path(pdf_path).stem
    temp_dir = Path(config["temp_dir"]) / pdf_name
    temp_dir.mkdir(exist_ok=True)
    
    conv_start = time.time()
    doc = None
    try:
        doc = fitz.open(pdf_path)
        if doc.page_count == 0:
            logging.warning(f"Empty PDF: {pdf_path}")
            return pdf_name, 0, 0
        logging.info(f"Processing {pdf_name} with {doc.page_count} pages")
        
        preproc_durations = []
        for i in range(0, doc.page_count, config["preprocess_batch_size"]):
            batch = []
            for page_num in range(i, min(i + config["preprocess_batch_size"], doc.page_count)):
                page = doc[page_num]
                pix = page.get_pixmap(dpi=config["image_dpi"])
                mode = "RGB" if pix.n < 4 else "RGBA"
                img = Image.frombytes(mode, [pix.width, pix.height], pix.samples)
                batch.append(img)
            processed_batch, duration = preprocess_image_batch(batch)
            preproc_durations.append(duration)
            for j, img in enumerate(processed_batch):
                page_index = i + j + 1
                if config["preview_image"]:
                    cv2.imwrite(str(temp_dir / f"page_{page_index:03d}.png"), img)
                else:
                    np.save(str(temp_dir / f"page_{page_index:03d}.npy"), img)
                del img
            gc.collect()
        
        conv_duration = (time.time() - conv_start) * 1000 / doc.page_count if doc.page_count else 0
        avg_preproc = sum(preproc_durations) / len(preproc_durations) if preproc_durations else 0
        total_stage1 = conv_duration + avg_preproc
        speed = 1000 / total_stage1 if total_stage1 > 0 else 0
        logging.info(f"Processed {pdf_name}: {doc.page_count} pages, Conv: {conv_duration:.2f} ms/page, "
                     f"Preproc: {avg_preproc:.2f} ms/page, Speed: {speed:.2f} pages/sec")
        return pdf_name, doc.page_count, total_stage1
    except Exception as e:
        logging.error(f"Failed to process PDF {pdf_path}: {e}")
        return pdf_name, 0, 0
    finally:
        if doc is not None:
            doc.close()

def stage1(pdf_files, completed_stage1, config):
    """
    Stage 1: Preprocess PDFs into images.
    Processes a batch of PDFs, updates the completed list, and logs performance.

    Args:
        pdf_files (list): List of PDF file paths.
        completed_stage1 (list): List of already processed PDF names.
        config (dict): Configuration dictionary.

    Returns:
        list: Updated list of completed PDF names for Stage 1.
    """
    stage1_start = time.time()
    to_process = [f for f in pdf_files if Path(f).stem not in completed_stage1][:config["max_pdfs_per_batch"]]
    if not to_process:
        logging.info("No PDFs to process in Stage 1.")
        return completed_stage1

    logging.info(f"Stage 1: Processing {len(to_process)} PDFs: {', '.join(Path(f).stem for f in to_process)}")
    
    # Use partial to bind the 'config' argument to process_pdf
    worker = partial(process_pdf, config=config)
    with Pool(config["num_producers"]) as pool:
        results = list(tqdm(pool.imap_unordered(worker, to_process),
                            total=len(to_process),
                            desc="Stage 1: PDF to Preprocessed Images",
                            unit="PDF"))
    
    total_pages = sum(page_count for _, page_count, _ in results)
    elapsed = time.time() - stage1_start
    elapsed_ms = elapsed * 1000
    avg_stage1_per_page = elapsed_ms / total_pages if total_pages > 0 else 0
    logging.info(f"Stage 1 Completed: {total_pages} pages, Avg Time: {avg_stage1_per_page:.2f} ms/page, "
                 f"Total Stage 1 Time: {elapsed:.2f} sec.")
    
    for pdf_name, _, _ in results:
        if pdf_name not in completed_stage1:
            completed_stage1.append(pdf_name)
    save_completed("completed_stage1.json", completed_stage1)
    return completed_stage1