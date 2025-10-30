"""Main entry point for Charisma application"""

import argparse
import sys
from pathlib import Path

try:
    from rich.console import Console
except ImportError as e:
    raise ImportError(
        f"Required package not installed: {e}. "
        "Please run 'uv sync' or 'pip install -e .' to install dependencies."
    )

from charisma import __version__
from charisma.ui import CharismaApp
from charisma.utils.logger import setup_logger, get_logger

setup_logger()
logger = get_logger()
# Configure console for Windows compatibility - use ASCII characters only
console = Console(legacy_windows=False, force_terminal=True, no_color=False)


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description="Charisma - Personal Memory Clone using Unsloth LLMs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Launch locally
  charisma

  # Launch with public URL (for Google Colab)
  charisma --live

  # Launch on specific port
  charisma --port 7860

  # Custom config file
  charisma --config my_config.toml

Privacy Notice:
  All processing happens locally on your machine.
  No data is collected on our servers.
  Your Notion data and personal information stay private.
        """,
    )

    parser.add_argument(
        "--live",
        action="store_true",
        help="Create a public Gradio URL (useful for Google Colab)",
    )

    parser.add_argument(
        "--port",
        type=int,
        default=7860,
        help="Port to run the server on (default: 7860)",
    )

    parser.add_argument(
        "--config",
        type=str,
        default="charisma.toml",
        help="Path to configuration file (default: charisma.toml)",
    )

    parser.add_argument(
        "--server-name",
        type=str,
        default="127.0.0.1",
        help="Server name/IP to bind to (default: 127.0.0.1)",
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode",
    )

    args = parser.parse_args()

    # Ensure config file exists
    config_path = Path(args.config)
    if not config_path.exists():
        logger.warning(f"Config file not found: {args.config}")
        logger.info("A default config file will be created on first save in Settings tab")

    # Print banner with Rich
    console.print()
    console.print("-" * 80)
    console.print("[bold cyan]Charisma - Personal Memory Clone[/bold cyan]".center(80))
    console.print("-" * 80)
    console.print()
    
    # Create info table
    from rich.table import Table
    info_table = Table(show_header=False, box=None, padding=(0, 2))
    info_table.add_column(style="bold yellow", no_wrap=True)
    info_table.add_column(style="white")
    
    info_table.add_row("Version:", f"[bold green]{__version__}[/bold green]")
    info_table.add_row("Config:", f"[cyan]{args.config}[/cyan]")
    info_table.add_row("Port:", f"[magenta]{args.port}[/magenta]")
    info_table.add_row("Server:", f"[blue]{args.server_name}[/blue]")
    info_table.add_row("URL:", f"[bold blue]http://{args.server_name}:{args.port}/[/bold blue]")
    info_table.add_row("Public URL:", f"[{'green' if args.live else 'red'}]{'Yes' if args.live else 'No'}[/{'green' if args.live else 'red'}]")
    info_table.add_row("Debug:", f"[{'yellow' if args.debug else 'dim'}]{'Yes' if args.debug else 'No'}[/{'yellow' if args.debug else 'dim'}]")
    
    console.print(info_table)
    console.print()
    
    # Privacy notice
    from rich.panel import Panel
    privacy_notice = Panel(
        "[yellow]PRIVACY NOTICE:[/yellow]\n\n"
        "[green]*[/green] All processing happens [bold green]locally on your machine[/bold green]\n"
        "[green]*[/green] No data is collected on our servers\n"
        "[green]*[/green] Your data [bold cyan]never leaves your computer[/bold cyan]\n"
        "[green]*[/green] Notion data stays [bold magenta]completely private[/bold magenta]",
        title="[bold red]Privacy & Security[/bold red]",
        border_style="yellow",
    )
    console.print(privacy_notice)
    console.print()
    console.print("-" * 80)
    console.print()

    logger.info("Starting Charisma application")
    logger.info(f"Arguments: {vars(args)}")

    try:
        # Create and launch app
        app = CharismaApp(config_path=args.config)

        launch_kwargs = {
            "share": args.live,
            "server_port": args.port,
            "server_name": args.server_name,
            "show_error": True,
            "quiet": not args.debug,
        }

        logger.info("Launching Charisma UI")
        logger.info(f"Access URL: http://{args.server_name}:{args.port}/")
        
        # Display access URL with Rich
        console.print("[bold green]>> Starting server...[/bold green]")
        console.print(f"[bold cyan]Access at:[/bold cyan] [bold blue underline]http://{args.server_name}:{args.port}/[/bold blue underline]")
        if args.live:
            console.print("[yellow]Generating public URL...[/yellow]")
        console.print()
        
        app.launch(**launch_kwargs)

    except KeyboardInterrupt:
        logger.info("Application stopped by user")
        console.print("\n")
        console.print("[bold green]Goodbye![/bold green] Your data remains [bold cyan]safe and local[/bold cyan].")
        console.print()
        logger.complete()  # Flush all logs
        sys.exit(0)

    except Exception as e:
        logger.error(f"Application error: {str(e)}", exc_info=True)
        console.print("\n")
        console.print(f"[bold red]Error:[/bold red] {str(e)}")
        console.print("\n[yellow]Please check the logs for more details.[/yellow]\n")
        logger.complete()  # Flush all logs
        sys.exit(1)


if __name__ == "__main__":
    main()
