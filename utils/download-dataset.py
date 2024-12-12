import os
import sys
import requests
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import threading
import signal
from datetime import datetime
import urllib.parse
from rich.console import Console
from rich.progress import Progress, BarColumn, DownloadColumn, TextColumn, TimeRemainingColumn
from rich.live import Live
from rich.layout import Layout
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich import box

# ============================
# Configuration and Setup
# ============================

# Base URL and target directory
base_url = 'https://files.ifi.uzh.ch/ddis/teaching/ATAI2024/dataset/'
target_directory = os.path.join(os.path.dirname(__file__), 'dataset')

#!! THESE WILL BE SKIPPED DURING THE DOWNLOAD, uncomment to download everything
# relative to the base url
SKIP_DIRECTORIES = [
    'movienet/images/',
    "movienet/image_features/",
    "movienet/frames/",
    "movienet/images/"
]

# Initialize Console
console = Console()

# Statistics (protected by a lock for thread safety)
stats_lock = threading.Lock()
stats = {
    "downloaded": 0,
    "skipped": 0,
    "failed": 0
}

# Active downloads dictionary with thread-safe access
active_downloads_lock = threading.Lock()
active_downloads = {}

# Log messages queue
log_lock = threading.Lock()
logs = []

# Event to signal shutdown
shutdown_event = threading.Event()

# ThreadPoolExecutor
executor = ThreadPoolExecutor(max_workers=100)

# ============================
# UI Layout Setup
# ============================

def create_layout():
    layout = Layout()

    layout.split(
        Layout(name="header", size=3),
        Layout(name="body", ratio=1)
    )

    layout["body"].split_row(
        Layout(name="downloads", ratio=2),
        Layout(name="messages", ratio=1)
    )

    return layout

def render_header():
    table = Table.grid(expand=True)
    table.add_column(justify="center")
    with stats_lock:
        table.add_row(
            Text(f"Downloaded: {stats['downloaded']} | Skipped: {stats['skipped']} | Failed: {stats['failed']}", style="bold green")
        )
    return Panel(table, style="cyan", box=box.ROUNDED)

def render_downloads():
    table = Table(title="Active Downloads", box=box.ROUNDED, expand=True)
    table.add_column("File", style="cyan", no_wrap=True)
    table.add_column("Progress", style="magenta")

    with active_downloads_lock:
        for file, progress in active_downloads.items():
            progress_bar = f"{progress['current']}/{progress['total']} [{progress['percentage']:.2f}%]"
            table.add_row(file, progress_bar)
    
    return Panel(table, style="yellow", box=box.ROUNDED)

def render_messages():
    with log_lock:
        if not logs:
            messages = "No messages yet."
        else:
            messages = "\n".join(logs[-100:])  # Show last 100 messages
    return Panel(Text(messages, style="white"), title="Messages", box=box.ROUNDED)

def update_logs(message):
    with log_lock:
        logs.append(message)
        if len(logs) > 1000:
            logs.pop(0)  # Keep the log size manageable

# ============================
# Download Function
# ============================


def should_skip_directory(directory_url):
    """
    Determines if the given directory URL should be skipped based on the SKIP_DIRECTORIES list.
    
    Args:
        directory_url (str): The URL of the directory to check.
    
    Returns:
        bool: True if the directory should be skipped, False otherwise.
    """
    base_parsed = urllib.parse.urlparse(base_url)
    directory_parsed = urllib.parse.urlparse(directory_url)
    
    # Extract the path relative to the base_url
    base_path = base_parsed.path
    directory_path = directory_parsed.path
    
    if not directory_path.startswith(base_path):
        return False  # Not under base_url, process normally
    
    relative_path = directory_path[len(base_path):]  # Remove base_path from directory_path
    
    for skip_dir in SKIP_DIRECTORIES:
        if relative_path.startswith(skip_dir):
            return True
    return False

def download_file(file_url, local_file_path):
    global stats
    if shutdown_event.is_set():
        return
    try:
        # Log queuing
        # update_logs(f"Queueing file: {file_url}")
        
        # Send HEAD request to get the file size
        head = requests.head(file_url, timeout=5)
        remote_file_size = int(head.headers.get('content-length', 0))

        # Check if file exists and size matches
        if os.path.exists(local_file_path):
            local_file_size = os.path.getsize(local_file_path)
            if local_file_size == remote_file_size:
                with stats_lock:
                    stats['skipped'] += 1
                # update_logs(f"Skipping (already exists and is the same size): {local_file_path}")
                return

        # Log entering download
        # update_logs(f"Starting download: {local_file_path}")

        # Initialize progress
        with active_downloads_lock:
            active_downloads[local_file_path] = {
                "current": 0,
                "total": remote_file_size,
                "percentage": 0.0
            }

        # Download with stream
        with requests.get(file_url, stream=True, timeout=10) as response:
            response.raise_for_status()
            chunk_size = 1024
            with open(local_file_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=chunk_size):
                    if shutdown_event.is_set():
                        raise Exception("Download interrupted due to shutdown.")
                    if chunk:
                        f.write(chunk)
                        with active_downloads_lock:
                            active_downloads[local_file_path]["current"] += len(chunk)

                            if active_downloads[local_file_path]["total"] > 0:
                                active_downloads[local_file_path]["percentage"] = (
                                    active_downloads[local_file_path]["current"] / active_downloads[local_file_path]["total"]
                                ) * 100
                            else:
                                active_downloads[local_file_path]["percentage"] = 100.0  # Consider 100% if the file size is unknown
                            # active_downloads[local_file_path]["percentage"] = (active_downloads[local_file_path]["current"] / active_downloads[local_file_path]["total"]) * 100

        with stats_lock:
            stats['downloaded'] += 1
        # update_logs(f"Completed download: {local_file_path}")
    

    except Exception as e:
        if not shutdown_event.is_set():
            with stats_lock:
                stats['failed'] += 1
            update_logs(f"Failed to download {file_url}: {e}")
        # Delete partial file
        try:
            if os.path.exists(local_file_path):
                os.remove(local_file_path)
                update_logs(f"Partially downloaded file deleted: {local_file_path}")
        except Exception as del_e:
            update_logs(f"Error deleting file {local_file_path}: {del_e}")
    finally:
        with active_downloads_lock:
            if local_file_path in active_downloads:
                del active_downloads[local_file_path]

# ============================
# Directory Traversal
# ============================

def traverse_directories():
    dirs_to_process = [(base_url, target_directory)]

    while dirs_to_process and not shutdown_event.is_set():
        current_dir_url, current_local_dir = dirs_to_process.pop()
        
        # Check if the current directory should be skipped
        if should_skip_directory(current_dir_url):
            update_logs(f"Skipping directory: {current_dir_url}")
            continue  # Skip processing this directory
        
        try:
            response = requests.get(current_dir_url, timeout=5)
            response.raise_for_status()
            html = response.text
        except Exception as e:
            update_logs(f"Error accessing {current_dir_url}: {e}")
            continue

        # Log entering directory
        update_logs(f"Entering directory: {current_dir_url}")

        for line in html.splitlines():
            if shutdown_event.is_set():
                break
            if 'href="' in line and 'Parent Directory' not in line:
                # Extract the href value
                try:
                    href_start = line.index('href="') + len('href="')
                    href_end = line.index('"', href_start)
                    file_name = line[href_start:href_end]
                except ValueError:
                    continue  # Malformed line, skip

                if '?' in file_name:
                    continue  # Skip links with query parameters

                file_url = urllib.parse.urljoin(current_dir_url, file_name)
                if file_name.endswith('/'):
                    # It's a directory, add to stack
                    new_local_dir = os.path.join(current_local_dir, file_name)
                    os.makedirs(new_local_dir, exist_ok=True)
                    dirs_to_process.append((file_url, new_local_dir))
                else:
                    # It's a file, submit to executor
                    local_file_path = os.path.join(current_local_dir, file_name)
                    if shutdown_event.is_set():
                        break
                    executor.submit(download_file, file_url, local_file_path)


# ============================
# Graceful Shutdown Handling
# ============================

def handle_shutdown():
    shutdown_event.set()
    update_logs("Shutdown initiated. Cleaning up...")

    # Shutdown the executor to prevent new tasks
    executor.shutdown(wait=False)

    # Remove all incomplete files
    with active_downloads_lock:
        for file_path in list(active_downloads.keys()):
            if os.path.exists(file_path):
                try:
                    os.remove(file_path)
                    update_logs(f"Removed incomplete file: {file_path}")
                except Exception as e:
                    update_logs(f"Error deleting file {file_path}: {e}")
            del active_downloads[file_path]

# ============================
# UI Update Function
# ============================

def update_ui(layout):
    layout["header"].update(render_header())
    layout["body"]["downloads"].update(render_downloads())
    layout["body"]["messages"].update(render_messages())

# ============================
# Signal Handler Registration
# ============================

def register_signal_handler(live):
    def signal_handler_inner(sig, frame):
        handle_shutdown()
    signal.signal(signal.SIGINT, signal_handler_inner)

# ============================
# Main Function
# ============================

def main():
    layout = create_layout()
    with Live(layout, refresh_per_second=2, screen=True):
        register_signal_handler(live=None)  # Already handled via signal.signal
        traversal_thread = threading.Thread(target=traverse_directories, daemon=True)
        traversal_thread.start()

        while traversal_thread.is_alive() or active_downloads:
            update_ui(layout)
            if shutdown_event.is_set():
                break
            threading.Event().wait(0.5)  # Wait for 0.5 seconds before next update

        # Wait for all tasks to finish
        executor.shutdown(wait=True)

        # Final UI update
        update_ui(layout)

    # Show summary
    console.print("\n[bold green]Download Summary:[/bold green]")
    with stats_lock:
        console.print(f"Downloaded files: {stats['downloaded']}")
        console.print(f"Skipped files: {stats['skipped']}")
        console.print(f"Failed files: {stats['failed']}")

if __name__ == "__main__":
    main()
