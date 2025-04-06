import json
from pathlib import Path
import psutil
import logging
import asyncio
import aiofiles

def worker_init():
    """
    Initialize a worker process by pinning it to a specific CPU core.
    """
    p = psutil.Process()
    total_cores = psutil.cpu_count(logical=True)
    core = p.pid % total_cores
    p.cpu_affinity([core])
    logging.info(f"Process {p.pid} pinned to core {core}")

def load_completed(file_path):
    """
    Load a JSON list from a file containing completed PDF names.
    
    Args:
        file_path (str): Path to the JSON file.
        
    Returns:
        list: List of completed PDF names.
    """
    path = Path(file_path)
    return json.loads(path.read_text()) if path.exists() else []

def save_completed(file_path, completed_list):
    """
    Save a list of completed PDF names to a JSON file.
    
    Args:
        file_path (str): Path to the JSON file.
        completed_list (list): List of completed PDF names.
    """
    with open(file_path, "w") as f:
        json.dump(completed_list, f)

async def write_txt_async(file_path, text):
    """
    Asynchronously write text to a file.
    
    Args:
        file_path (Path): Path object to the output text file.
        text (str): Text content to write.
    """
    async with aiofiles.open(file_path, "w") as f:
        await f.write(text)