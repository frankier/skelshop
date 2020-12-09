from csv import DictReader
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

from skelshop.shotseg.io import open_grouper


def read_corpus_desc(
    corpus_desc: Iterator[str], corpus_base: Optional[Path]
) -> List[Tuple[Path, str, Path, Path]]:
    next(corpus_desc)
    result: List[Tuple[Path, str, Path, Path]] = []
    for line in corpus_desc:
        row = line.strip().split(",")
        group = Path(row[0])
        group_typ = row[1]
        faces = Path(row[2])
        segsout = Path(row[3])
        if corpus_base is not None:
            group = corpus_base.joinpath(group)
            faces = corpus_base.joinpath(faces)
            segsout = corpus_base.joinpath(segsout)
        result.append((group, group_typ, faces, segsout))
    return result


class CorpusReader:
    PATH_PROPERTIES = {
        "faces",
        "segsout",
        "video",
        "skels_untracked",
        "skels_tracked",
        "bestcands",
    }

    def __init__(self, inf: Path, corpus_base: Optional[Path] = None):
        self.file = open(inf)
        self.corpus_base = corpus_base

    def _proc_path(self, str_path: str) -> Path:
        val = Path(str_path)
        if self.corpus_base is not None:
            val = self.corpus_base.joinpath(val)
        return val

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        self.file.seek(0)
        reader = DictReader(self.file)
        for row in reader:
            res: Dict[str, Any] = {}
            if "group" in row:
                if "group_typ" not in row:
                    raise ValueError("Column 'group' implies 'grouptyp'")
                group = self._proc_path(row.pop("group"))
                group_typ = row.pop("group_typ")
                res["group"] = lambda: open_grouper(group, group_typ)
            for k, str_val in row.items():
                val: Any
                if k in self.PATH_PROPERTIES:
                    res[k] = self._proc_path(str_val)
                    res["raw_" + k] = str_val
                else:
                    res[k] = str_val
            yield res

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.file.close()


def index_corpus_desc(corpus_desc, corpus_base):
    with CorpusReader(corpus_desc, corpus_base) as corpus:
        return list(corpus)
