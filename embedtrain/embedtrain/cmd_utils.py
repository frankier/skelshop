import click

image_base_option = click.option(
    "--image-base", envvar="IMAGE_BASE", required=True, type=click.Path(exists=True)
)
body_labels_option = click.option(
    "--body-labels", envvar="BODY_LABELS", type=click.Path(exists=True)
)
