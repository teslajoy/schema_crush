"""cli for schema_crush."""

import click
from schema_crush import __version__


@click.group()
@click.version_option(version=__version__)
def cli():
    """schema_crush - multi-agent schema mapping system."""
    pass


@cli.command()
@click.argument('source')
@click.argument('target')
def map_schema(source, target):
    """map source schema to target schema."""
    click.echo(f"mapping {source} to {target}...")
    # TODO: implement mapping logic


@cli.command()
@click.argument('csv_path')
def load_csv(csv_path):
    """load and analyze csv file."""
    click.echo(f"loading {csv_path}...")
    # TODO: implement csv loading


if __name__ == "__main__":
    cli()