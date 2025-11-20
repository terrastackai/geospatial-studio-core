import typer
import uvicorn

cli = typer.Typer(help="GFM Studio command-line interface")


@cli.command(help="Run the FastAPI server in development mode.")
def dev(
    host: str = "0.0.0.0",
    port: int = 4400,
    log_level: str = "debug",
):
    uvicorn.run(
        "gfmstudio.main:app",
        host=host,
        port=port,
        reload=True,
        log_level=log_level,
        loop="asyncio",
    )


@cli.command(help="Run the FastAPI server in production mode.")
def serve(
    host: str = "0.0.0.0",
    port: int = 4400,
    workers: int = 4,
    log_level: str = "info",
):
    uvicorn.run(
        "gfmstudio.main:app",
        host=host,
        port=port,
        reload=False,
        workers=workers,
        log_level=log_level,
        loop="asyncio",
    )


def main():
    cli()


if __name__ == "__main__":
    main()
