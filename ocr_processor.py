import time
import logging
import numpy as np
from PIL import Image
import tesserocr
import os
import contextlib
from pathlib import Path
import shutil
import asyncio
from tqdm import tqdm
from utils import write_txt_async, save_completed
from multiprocessing import Pool

def process_image(args):
    """
    Perform OCR on a single image saved as a numpy array.
    
    Args:
        args (tuple): Contains (pdf_name, img_path, page_num).
    
    Returns:
        tuple: (pdf_name, page number, recognized text, OCR duration in ms)
    """
    pdf_name, img_path, page_num = args
    start_time = time.time()
    img = np.load(str(img_path))
    if img is None:
        logging.error(f"Failed to load image: {img_path}")
        return pdf_name, page_num, "", 0
    pil_img = Image.fromarray(img)
    # Suppress Tesseract warnings by redirecting stderr to devnull temporarily
    with tesserocr.PyTessBaseAPI(psm=tesserocr.PSM.AUTO, oem=tesserocr.OEM.LSTM_ONLY) as api:
        api.SetImage(pil_img)
        with open(os.devnull, "w") as devnull:
            with contextlib.redirect_stderr(devnull):
                text = api.GetUTF8Text()
    duration = (time.time() - start_time) * 1000
    logging.debug(f"OCR duration for {img_path}: {duration:.2f} ms")
    del img, pil_img
    return pdf_name, page_num, text, duration

def stage2(completed_stage1, completed_stage2, config):
    """
    Stage 2: Perform OCR on preprocessed images for PDFs not yet completed.
    Writes final text output to disk and cleans up temporary files.
    
    Args:
        completed_stage1 (list): List of PDFs processed in Stage 1.
        completed_stage2 (list): List of PDFs already processed in Stage 2.
        config (dict): Configuration dictionary.
    
    Returns:
        tuple: (Updated completed_stage2 list, total number of pages processed)
    """
    stage2_start = time.time()
    to_process = [pdf_name for pdf_name in completed_stage1 if pdf_name not in completed_stage2]
    if not to_process:
        logging.info("No PDFs to process in Stage 2.")
        return completed_stage2, 0

    logging.info(f"Stage 2: OCR on {len(to_process)} PDFs: {to_process}")
    image_tasks = []
    for pdf_name in to_process:
        temp_dir = Path(config["temp_dir"]) / pdf_name
        for img_path in sorted(temp_dir.glob("page_*.npy")):
            page_num = int(img_path.stem.split("_")[1])
            image_tasks.append((pdf_name, img_path, page_num))
    
    with Pool(config["num_consumers"]) as pool:
        results = list(tqdm(pool.imap_unordered(process_image, image_tasks),
                            total=len(image_tasks), desc="Stage 2: OCR", unit="page"))
    
    pdf_texts = {}
    for pdf_name, page_num, text, _ in results:
        pdf_texts.setdefault(pdf_name, []).append((page_num, text))
    
    async def write_all():
        tasks = []
        for pdf_name, pages in pdf_texts.items():
            pages.sort()
            full_text = "\n".join(f"---Page {num}---\n{txt}" for num, txt in pages)
            txt_path = Path(config["output_dir"]) / f"{pdf_name}.txt"
            tasks.append(write_txt_async(txt_path, full_text))
            temp_dir = Path(config["temp_dir"]) / pdf_name
            if temp_dir.exists():
                shutil.rmtree(temp_dir)
        await asyncio.gather(*tasks)
    
    asyncio.run(write_all())
    
    total_pages = sum(len(pages) for pages in pdf_texts.values())
    elapsed = time.time() - stage2_start  # seconds
    avg_ocr_per_page = ((elapsed * 1000) / total_pages) if total_pages > 0 else 0 
    logging.info(f"Stage 2 Completed: {total_pages} pages, Avg OCR: {avg_ocr_per_page:.2f} ms/page, "
                 f"Total OCR Time: {elapsed:.2f} sec")
    
    for pdf_name in pdf_texts:
        if pdf_name not in completed_stage2:
            completed_stage2.append(pdf_name)
    save_completed("completed_stage2.json", completed_stage2)
    return completed_stage2, total_pages