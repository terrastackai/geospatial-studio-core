# © Copyright IBM Corporation 2025
# SPDX-License-Identifier: Apache-2.0


import glob
import os
import subprocess

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

console = Console()


def load_seed_data(db_url):
    """
    Load all SQL files from seed-data/ into the database, ignoring errors in each file.
    """
    location = "gfmstudio/db_migrations/seed_data"
    sql_files = sorted(glob.glob(f"{location}/*.sql"))

    if not sql_files:
        console.print(f"[bold yellow]⚠️  No SQL files found in {location}[/bold yellow]")
        return

    completed = []
    skipped = []

    console.print(
        f"[bold cyan]🔎 Found {len(sql_files)} SQL file(s) to load[/bold cyan]"
    )

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        transient=True,
        console=console,
    ) as progress:
        task = progress.add_task("Loading SQL files...\n", total=len(sql_files))

        for sql_file in sql_files:
            progress.update(task, description=f"Running [green]{sql_file}[/green]")
            try:
                # Use psql via subprocess
                output = subprocess.run(
                    ["psql", "-d", db_url, "-f", sql_file], capture_output=False
                )
                # rprint(f"✅ Successfully ran: {sql_file}")
            except subprocess.CalledProcessError:
                progress.update(task, description=f"Skipping [red]{sql_file}[/red]")
                skipped.append(sql_file)
            else:
                if "ERROR" in str(output):
                    progress.update(task, description=f"Skipping [red]{sql_file}[/red]")
                    skipped.append(sql_file)
                else:
                    completed.append(sql_file)

    # Show results in a nice table
    table = Table(title="Seed Data Load Summary", show_lines=True)
    table.add_column("Status", style="bold")
    table.add_column("File", style="dim")

    for f in completed:
        table.add_row("[green]✅ Loaded[/green]", f)

    for f in skipped:
        table.add_row("[red]⚠️  Skipped[/red]", f)

    console.print(table)


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()
    db_url = os.getenv("DATABASE_URI")
    if db_url:
        db_url = db_url.replace("+pg8000", "")
    # rprint(db_url)
    load_seed_data(db_url=db_url)
