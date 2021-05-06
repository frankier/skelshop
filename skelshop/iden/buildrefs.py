import shutil
import urllib.request
from pathlib import Path
from string import Template

import certifi
from SPARQLWrapper import JSON, SPARQLWrapper

WIKIDATA_ENDPOINT = "https://query.wikidata.org/sparql"
COMMONS_ENDPOINT = "https://wcqs-beta.wmflabs.org/sparql"


WIKIDATA_QUERY = Template(
    """
SELECT ?item ?pic
WHERE
{
    $where_clause
    .
    ?item wdt:P18 ?pic
}
"""
)


COMMONS_QUERY = Template(
    """
SELECT ?item ?pic WHERE {
    $where_clause
    .
    ?file wdt:P180 ?item
    .
    ?file schema:contentUrl ?url
    .
    bind(iri(concat("http://commons.wikimedia.org/wiki/Special:FilePath/", wikibase:decodeUri(substr(str(?url),53)))) AS ?pic)
}
"""
)


def build_in_category_snippet(categories):
    snippets = []
    for category in categories:
        snippets.append("{?item wdt:P106/wdt:P279* wd:" + category + "}")
    return "\nUNION\n".join(snippets)


def build_in_list_snippet(items):
    return "FILTER (?item IN (" + ",".join(("wd:" + item for item in items)) + "))"


def monkeypatch_sparqlwrapper():
    from functools import partial

    from SPARQLWrapper import Wrapper

    if not hasattr(Wrapper.urlopener, "monkeypatched"):
        Wrapper.urlopener = partial(Wrapper.urlopener, cafile=certifi.where())
        setattr(Wrapper.urlopener, "monkeypatched", True)


def oauth_client(auth_file):
    from oauthlib.oauth1 import Client

    creds = []
    for idx, line in enumerate(auth_file):
        if idx % 2 == 0:
            continue
        creds.append(line.strip())
    return Client(*creds)


class OAuth1SPARQLWrapper(SPARQLWrapper):
    def __init__(self, *args, **kwargs):
        self.client = kwargs.pop("client")
        super().__init__(*args, **kwargs)

    def _createRequest(self):
        request = super()._createRequest()
        uri = request.get_full_url()
        method = request.get_method()
        body = request.data
        headers = request.headers
        new_uri, new_headers, new_body = self.client.sign(uri, method, body, headers)
        request.full_url = new_uri
        request.headers = new_headers
        request.data = new_body
        return request


def download_refs(sparql: SPARQLWrapper, query: str, outdir: Path):
    monkeypatch_sparqlwrapper()
    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)
    results = sparql.query().convert()
    for row in results["results"]["bindings"]:
        entity_id = row["item"]["value"].rsplit("/", 1)[-1]
        pic = row["pic"]["value"]
        entity_dir = outdir / entity_id
        entity_dir.mkdir(parents=True, exist_ok=True)
        orig_image_path = image_path = entity_dir / pic.split("/")[-1]
        idx = 2
        while image_path.exists():
            image_path = orig_image_path.with_name(
                f"{orig_image_path.stem}_{idx}{orig_image_path.suffix}"
            )
            idx += 1
        with urllib.request.urlopen(pic, cafile=certifi.where()) as response, open(
            image_path, "wb"
        ) as img_file:
            shutil.copyfileobj(response, img_file)
