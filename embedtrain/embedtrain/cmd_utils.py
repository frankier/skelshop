import click

image_base_option = click.option(
    "--image-base", envvar="IMAGE_BASE", required=True, type=click.Path(exists=True)
)
