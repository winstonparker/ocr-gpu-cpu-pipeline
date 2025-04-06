# OCR GPU CPU Pipeline

## Overview

The OCR GPU CPU Pipeline is a modular system to process PDF documents using a combination CPU and GPU (Optional) design. The system batch converts each page into a preprocessed image (GPU) and subsequently extracts text using Tesseract OCR (CPU). During the preprocessing phase, the system leverages GPU acceleration, when available, to efficiently perform border expansion, grayscale conversion, and Gaussian filtering. If GPU processing is unavailable or encounters errors, the pipeline defaults to the CPU. The OCR stage is executed on the CPU; for high-core CPU systems, lower-memory GPUs tend to underperform CPU based OCR (GPU latency tested with PaddleOCR and Doctr). 

The CLI options allow for changing: DPI settings, batch sizes, CPU, and the number of concurrent processes to best align with the system’s memory.

The pipeline will save a record of the progress, allowing you to stop between stages or batches.

## Setup

## Initial Steps 
1. **Clone the Repository:**
   ```bash
   https://github.com/winstonparker/ocr-gpu-cpu-pipeline.git
   cd ocr_pipeline
   ```

2. **Setup venv (Suggested)** If you do not already have a virtual environment in your project directory, you can set one up using the following:
    ```bash
    python -m venv venv
    source venv/bin/activate
    ```

    Windows (Repo Untested)
    ```
    python -m venv venv
    venv\Scripts\activate
    ```


### Using CPU Only
1. **Install Dependencies:**
   Ensure you have Python 3 installed. Install the required packages using pip:
    ```bash
    pip install -r requirements.txt
    ```

2. **Configure Tesseract (Suggested):**
   If you wish to use a specific Tesseract tessdata path, you can change the `tessdata_prefix` in the configs or through the CLI. This may not be needed if your default paths are working. 

   Example: `/usr/share/tesseract-ocr/5/tessdata`

### Adding GPU Support (Optional, faster)

The python opencv library does not provide CUDA support. We will need to build OpenCV from its source.

Prerequisites:
- Ensure you have an NVIDIA GPU with supported CUDA compute capability.
- Install the latest NVIDIA drivers.
- Install the CUDA Toolkit (e.g., CUDA 12.x) and cuDNN libraries.
- Install other dependencies required for building OpenCV (e.g., CMake, build-essential, libgtk, etc.).

Build Steps:

1. Install the required packages using pip:
    ```bash
    pip install -r requirements_gpu.txt
    ```

2. Follow [this guide to add cv2 cuda support to python](https://medium.com/@amosstaileyyoung/build-opencv-with-dnn-and-cuda-for-gpu-accelerated-face-detection-27a3cdc7e9ce). More [install scripts can be found here](https://github.com/Qengineering/Install-OpenCV-Jetson-Nano), additional links must be made to link opencv and python.

    Note: This can take 15-30+ minutes.


3. Environment Variables: Ensure your system library paths include CUDA libraries. Update to your cuda version/path. For example:
    ```bash
    export LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64:/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH
    export PATH=/usr/local/cuda-12.8/bin:$PATH
    ```

4. Verify support with this command:
    ```bash
    python -c "import cv2; print(cv2.__version__); print(cv2.cuda.getCudaEnabledDeviceCount())"
    ```
    A value of 1 or higher means your GPU is found. If 0, CPU only, ensure you don't have the base python library still installed.

5. **Configure Tesseract (Suggested):**
You may need to use a specific Tesseract tessdata path, you can change the `tessdata_prefix` in the configs or through the CLI. This might not be needed if your default paths are working. 

    Example: `/usr/share/tesseract-ocr/5/tessdata`


## Usage

Run the pipeline with default settings:
```bash
python main.py
```

### CLI Options
You can override configuration options via CLI arguments:
- `--pdf_dir`: Directory containing input PDF files.
- `--output_dir`: Directory to save output text files.
- `--temp_dir`: Temporary directory for intermediate images. The dir is  cleared after stage 2.
- `--max_pdfs_per_batch`: Maximum number of PDFs to process per batch.
- `--preprocess_batch_size`: Number of pages to process in each preprocessing batch.
- `--num_producers`: Number of producer processes for Stage 1.
- `--num_consumers`: Number of consumer processes for Stage 2 (must be less than total CPU cores minus one).
- `--image_dpi`: DPI to use when converting PDF pages to images.
- `--preview_image`: Enable to save temp images in viewable PNG format during Stage 1. For debugging. Images are erased after stage 2. You can stop the pipeline during Stage 2 if you wish to inspect the images. 
- `--tessdata_prefix`: Change the path to the Tesseract tessdata directory. Use only if the default is not working/setup.
- `-v` or `--version`: Display version information and exit.

#### Default CPU Options
By default, the pipeline uses the following CPU configuration parameters:
- **Number of Producers:** 6
- **Number of Consumers:** 10 (must be less than your total CPU cores minus one)
- **Preprocess Batch Size:** 50 pages per batch
- **Max PDFs Per Batch:** 10

**Note**: The defaults are setup for a low-mid tier system. For better performance, increase the params to meet your system spec. 
Defaults can be changed via the CLI or the `config.py` file.

#### Example:
```bash
python main.py --pdf_dir ./input --num_consumers 8
```

### Example Performance Configs

The following configuration achieved a total OCR throughput (Preprocessing + OCR) of approximately **5–10 pages per second** on a system with:
- **CPU:** 32 Cores (2.1 GHz)
- **Memory:** 80 GiB
- **GPU:** Tesla T4 (16 GB GDDR6)
- **OS:** Ubuntu Server 24.10

```bash
python main.py --pdf_dir ./input --image_dpi 300 --max_pdfs_per_batch 20 --preprocess_batch_size 200 --num_producers 12 --num_consumers 30
```

*Note: If you have limited disk space, consider using a lower DPI, fewer producers, and a smaller maximum batch size. I recommend allocating at least 1 GiB of disk space per producer for temporary image storage (using a fast SSD or NVMe drive is ideal, though not required). All temporary files are removed after each processing round, so space is freed periodically.*

## How It Works

The OCR Pipeline operates in two main stages:

- **Stage 1: PDF Processing & Preprocessing**  
  Each PDF is split into individual pages, which are then converted to images (npy files). These images are preprocessed to enhance OCR accuracy. The system first attempts to leverage GPU acceleration for preprocessing using CUDA. However, if GPU processing fails or is unavailable, the system falls back to CPU-based processing. This dual approach ensures that even systems with lower memory GPUs can still process images effectively.

- **Stage 2: OCR Processing**  
  Preprocessed images are then fed into Tesseract OCR (CPU). For high-core CPU systems, lower-memory GPUs tend to underperform CPU based OCR (GPU latency was higher when tested with PaddleOCR and Doctr). The OCR results are collated and saved as text files, with temporary image files cleaned up after each round to free up disk space.

The pipeline will save a record of the progress in two files: `compled_stage1.json` and `compled_stage2.json`. This allows you to stop the pipeline and not redo work done in a prior batch or stage. You can redo OCR by removing these files or specific file names from each.

The pipeline is tailored for systems with lower memory GPUs and high core CPUs, but can work across most machines (assuming configs are adjusted).

## Help

The GPU supported setup requires OpenCV be build from source. When the pipeline starts, a log will state if the GPU is being used.
### Common GPU Issues
1. Ensure your venv does not contain the python version of opencv. 
2. Ensure your NVIDIA drivers and CUDA libraries are setup. 
3. `No module named 'cv2'`: Make sure your cv2 .so is linked or copied to your `venv/lib/python3.XX/site-packages/` directory.

### Tesseract Issues
1. `RuntimeError: Failed to init API, possibly an invalid tessdata path: ...`: your default path is not pointing to your tessdata path. You can add the correct path via the CLI or in `config.py` (Example: `/usr/share/tesseract-ocr/5/tessdata`)



## Attribution
See the `ATTRIBUTIONS.md` file for third-party dependency attributions.

## Legal Disclaimer
This software is intended solely for lawful research, academic, and educational purposes.

**Before using this software, you are solely responsible for ensuring that your intended use complies with all applicable local, state, national, and international laws, regulations, and agreements.** Do not use this software in any jurisdiction where its use would be in violation of the law.

This project may include or rely upon third-party libraries and dependencies. The author makes **no warranties or representations** regarding the legality, safety, functionality, or licensing compliance of any third-party code included in or used by this project. Users are responsible for reviewing and complying with the licenses and terms of all dependencies.

The author:
- Does **not** encourage or support illegal activity,
- Provides this software **as-is**, without warranty or guarantee of legal fitness,
- Assumes **no responsibility** for how it is used or the consequences of its use.

This is a personal project. It is not affiliated with, endorsed by, or representative of the views, opinions, or interests of the author’s employer, past or present.
