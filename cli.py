import argparse
from config import CONFIG, VERSION

def parse_cli_args():
    """
    Parse command-line arguments to override default configuration.

    Returns:
        argparse.Namespace: Parsed arguments with potential config overrides.
    """
    parser = argparse.ArgumentParser(
        description="OCR Pipeline - Convert PDFs to text using GPU/CPU processing.\n"
                    "Adjust configuration parameters based on available memory and CPU cores. The pipeline will save a record of the progress, allowing you to stop between stages or batches.",
        epilog="Examples:\n  python main.py --pdf_dir ./input --num_consumers 4\n  python main.py --pdf_dir ./input --image_dpi 300 --max_pdfs_per_batch 20 --preprocess_batch_size 200 --num_producers 12 --num_consumers 30 \npython main.py --version",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--pdf_dir", type=str, help="Directory containing input PDF files.")
    parser.add_argument("--output_dir", type=str, help="Directory to save output text files.")
    parser.add_argument("--temp_dir", type=str, help="Temporary directory for intermediate images. The dir is cleared after stage 2.")
    parser.add_argument("--max_pdfs_per_batch", type=int, help="Maximum number of PDFs to process per batch.")
    parser.add_argument("--preprocess_batch_size", type=int, help="Number of pages to process in each preprocessing batch.")
    parser.add_argument("--num_producers", type=int, help="Number of producer processes for Stage 1.")
    parser.add_argument("--num_consumers", type=int, help="Number of consumer processes for Stage 2 (must be less than total CPU cores minus one).")
    parser.add_argument("--image_dpi", type=int, help="DPI to use when converting PDF pages to images.")
    parser.add_argument("--tessdata_prefix", type=str, help="Path to Tesseract tessdata directory. Use only if the default is not working/setup. Example: /usr/share/tesseract-ocr/5/tessdata")
    parser.add_argument("--preview_image", type=bool, help="Enable to save temp images in viewable PNG format during Stage 1. For debugging. Images are erased after stage 2. You can stop the pipeline during Stage 2 if you wish to inspect the images.")
    parser.add_argument("-v", "--version", action="version", version=f"OCR Pipeline Version {VERSION}",
                        help="Display version information and exit.")
    return parser.parse_args()

def update_config_from_args(args):
    """
    Update the global CONFIG dictionary based on parsed CLI arguments.

    Args:
        args (argparse.Namespace): Parsed CLI arguments.
    """
    for key in CONFIG.keys():
        arg_val = getattr(args, key, None)
        if arg_val is not None:
            CONFIG[key] = arg_val