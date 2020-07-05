import logging
import pickle
from os.path import join as pjoin
from typing import Any, Dict

import click
from pytorch_lightning import Trainer, loggers

from embedtrain.cmd_utils import body_labels_option
from embedtrain.litmod import MetGcnLit

logger = logging.getLogger(__name__)


@click.command()
# Files
@click.argument("h5fn")
@click.argument("skel")
@click.argument("outdir")
@click.option("--pre-vocab", type=click.File("rb"))
@body_labels_option
# Config
@click.option("--run-type", type=click.Choice(["prod", "eval"]), default="eval")
@click.option("--embed-size", type=int, default=64)
@click.option("--loss", type=click.Choice(["nsm", "msl"]), default="nsm")
@click.option("--no-aug", is_flag=True)
@click.option("--include-score", is_flag=True)
# Diagnostic modes
@click.option("--fastdev", is_flag=True)
@click.option("--overfit", is_flag=True)
@click.option("--lr-find-plot", is_flag=True)
@click.option("--test", is_flag=True)
def train(
    h5fn,
    skel,
    outdir,
    pre_vocab,
    body_labels,
    run_type,
    embed_size,
    loss,
    no_aug,
    include_score,
    fastdev,
    overfit,
    lr_find_plot,
    test,
):
    if skel != "HAND":
        assert body_labels is not None, "--body-labels needed for BODY skels"
    vocab = pickle.load(pre_vocab) if pre_vocab else None
    model = MetGcnLit(
        h5fn, run_type, graph=skel, vocab=vocab, loss=loss, body_labels=body_labels
    )
    tb_logger = loggers.TensorBoardLogger(pjoin(outdir, "logs/"))
    kwargs: Dict[str, Any] = {}
    if fastdev:
        kwargs["fast_dev_run"] = True
    elif overfit:
        kwargs["overfit_batches"] = 0.01
    trainer = Trainer(
        default_root_dir=outdir,
        logger=tb_logger,
        gpus=-1,
        early_stop_callback=True,
        **kwargs
    )
    if lr_find_plot:
        model.setup(None)
        lr_finder = trainer.lr_find(model)
        fig = lr_finder.plot(suggest=True)
        fig.show()
        input()
        return
    if not fastdev and not overfit:
        trainer.scale_batch_size(model, init_val=4096)
    trainer.fit(model)
    if test:
        trainer.test(ckpt_path=None)


if __name__ == "__main__":
    train()
