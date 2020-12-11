from pathlib import Path
from typing import List, TextIO

import click

from skelshop.utils.click import PathPath


def read_entities(listin: TextIO) -> List[str]:
    entities: List[str] = []
    for entity in listin:
        entities.append(entity.split()[0])
    return entities


def get_source_info(source, oauth_creds):
    from SPARQLWrapper import SPARQLWrapper

    from skelshop.iden.buildrefs import (
        COMMONS_ENDPOINT,
        COMMONS_QUERY,
        WIKIDATA_ENDPOINT,
        WIKIDATA_QUERY,
        OAuth1SPARQLWrapper,
        oauth_client,
    )

    if source == "wikidata":
        return SPARQLWrapper(WIKIDATA_ENDPOINT), WIKIDATA_QUERY
    else:
        client = oauth_client(oauth_creds)
        return OAuth1SPARQLWrapper(COMMONS_ENDPOINT, client=client), COMMONS_QUERY


@click.command()
@click.argument("type", type=click.Choice(["entities", "categories"]))
@click.argument("listin", type=click.File("r"))
@click.argument("refout", type=PathPath(exists=False))
@click.argument("source", type=click.Choice(["wikidata", "commons"]), default="commons")
@click.option("--oauth-creds", type=click.File("r"))
def buildrefs(
    type: str, listin: TextIO, refout: Path, source: str, oauth_creds: TextIO
):
    """
    Build a reference face library from a list of entities or categories.
    """
    from skelshop.iden.buildrefs import (
        build_in_category_snippet,
        build_in_list_snippet,
        download_refs,
    )

    if source == "commons" and oauth_creds is None:
        raise click.BadOptionUsage(
            "--oauth-creds", "--oauth-creds must be provided when source is 'commons'"
        )
    entities = read_entities(listin)
    if type == "entities":
        builder = build_in_list_snippet
    else:
        builder = build_in_category_snippet
    where_clause = builder(entities)
    endpoint_wrapper, query_tmpl = get_source_info(source, oauth_creds)
    query = query_tmpl.substitute(where_clause=where_clause)
    download_refs(endpoint_wrapper, query, refout)
