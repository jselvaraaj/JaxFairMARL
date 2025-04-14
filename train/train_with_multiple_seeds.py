import argparse
import os
import subprocess
import sys
import time
from datetime import datetime

from rich.console import Console
from rich.live import Live
from rich.table import Table

from config.mappo_config import MAPPOConfig

# Add the project root to Python path to fix module imports
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, ".."))
sys.path.insert(0, project_root)


# Parse command line arguments for this script
parser = argparse.ArgumentParser(
    description="Run multiple training processes with different seeds"
)
parser.add_argument(
    "--full-logs", action="store_true", help="Show full logs in the console"
)
parser.add_argument(
    "--max-lines",
    type=int,
    default=20,
    help="Maximum number of log lines to show per process",
)
parser.add_argument(
    "--keep-logs",
    action="store_true",
    help="Keep log files even if all processes succeed",
)
args = parser.parse_args()

timestamp = datetime.now().strftime("%b-%d-%Y_%H-%M-%S")
config: MAPPOConfig = MAPPOConfig.create()
assert (
    config.training_config.num_envs > 1
), "Number of environments must be greater than 1 for training"
if config.training_config.seed is None:
    seeds = list(range(config.training_config.num_seeds))
else:
    seeds = list(config.training_config.seed)

console = Console()
console.print(f"[bold green]Starting {len(seeds)} training processes...[/bold green]")

# Create logs directory if it doesn't exist
log_dir = f"logs_{timestamp}"
os.makedirs(log_dir, exist_ok=True)
console.print(f"Saving full logs to directory: [bold cyan]{log_dir}[/bold cyan]")

processes = []
process_outputs = {seed: [] for seed in seeds}
log_files = {}

# Start all processes
for seed in seeds:
    # Create Python script with proper imports as a string
    python_code = f"""
import os
import sys

# Set path to project root
sys.path.insert(0, "{project_root}")

# Import our modules now that path is set
from train.train_with_single_seed import main

# Run the experiment
main(seed={seed}, timestamp="{timestamp}")
"""

    # Create log file for this process
    log_path = os.path.join(log_dir, f"seed_{seed}.log")
    log_file = open(log_path, "w")
    log_files[seed] = log_file

    # Start process with output capture
    process = subprocess.Popen(
        [sys.executable, "-c", python_code],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    processes.append((seed, process))
    console.print(f"Started process for seed [bold cyan]{seed}[/bold cyan]")


def generate_table():
    table = Table(title="Training Processes")
    table.add_column("Seed")
    table.add_column("Status")
    table.add_column("Output")

    for seed, process in processes:
        status = "[green]Running" if process.poll() is None else "[red]Finished"

        if args.full_logs:
            output_lines = process_outputs[seed]
        else:
            output_lines = (
                process_outputs[seed][-args.max_lines :]
                if process_outputs[seed]
                else ["Waiting for output..."]
            )

        output_text = "\n".join(output_lines)
        table.add_row(f"{seed}", status, output_text)

    return table


# Process monitoring loop
with Live(generate_table(), refresh_per_second=2) as live:
    running = True
    while running:
        running = False
        for seed, process in processes:
            if process.poll() is None:
                running = True

            # Check for new output
            while True:
                line = process.stdout.readline() if process.stdout else ""
                if not line:
                    break
                process_outputs[seed].append(line.rstrip())
                # Also write to log file
                log_files[seed].write(line)
                log_files[seed].flush()

        live.update(generate_table())
        time.sleep(0.25)

# Close all log files
for log_file in log_files.values():
    log_file.close()

# Check if all processes completed successfully
all_succeeded = True
for seed, process in processes:
    if process.returncode != 0:
        all_succeeded = False
        console.print(
            f"[bold red]Process for seed {seed} failed with exit code {process.returncode}[/bold red]"
        )

# Delete log files if all processes succeeded and --keep-logs is not set
if all_succeeded and not args.keep_logs:
    console.print(
        "[bold yellow]All processes succeeded, deleting log files...[/bold yellow]"
    )
    # Delete all log files
    for seed in seeds:
        log_path = os.path.join(log_dir, f"seed_{seed}.log")
        try:
            os.remove(log_path)
        except Exception as e:
            console.print(
                f"[bold red]Failed to delete log file {log_path}: {e}[/bold red]"
            )

    # Delete log directory
    try:
        os.rmdir(log_dir)
        console.print(
            f"[bold green]Successfully deleted log directory: {log_dir}[/bold green]"
        )
    except Exception as e:
        console.print(
            f"[bold red]Failed to delete log directory {log_dir}: {e}[/bold red]"
        )
else:
    console.print("[bold green]All training processes completed![/bold green]")
    console.print(f"Full logs available in directory: [bold cyan]{log_dir}[/bold cyan]")
