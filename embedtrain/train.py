import logging
import pickle
from collections import ChainMap
from os.path import join as pjoin

from pytorch_lightning import Trainer, loggers

from embedtrain.litmod import MetGcnLit

logger = logging.getLogger(__name__)


def parse_args():
    from argparse import ArgumentParser, FileType

    parser = ArgumentParser()

    # Files
    parser.add_argument("h5fn")
    parser.add_argument("skel")
    parser.add_argument("outdir")
    parser.add_argument("--pre-vocab", type=FileType("rb"))
    parser.add_argument("--body-labels")

    # Model specific args
    parser = MetGcnLit.add_model_specific_args(parser)

    # Diagnostic modes
    parser.add_argument("--lr-find-plot", action="store_true")
    parser.add_argument("--test", action="store_true")

    # All the available trainer options
    parser = Trainer.add_argparse_args(parser)

    return parser.parse_args()


def train():
    args = parse_args()
    if args.skel != "HAND":
        assert args.body_labels is not None, "--body-labels needed for BODY skels"
    vocab = pickle.load(args.pre_vocab) if args.pre_vocab else None
    model = MetGcnLit(
        **ChainMap(
            vars(args),
            dict(graph=args.skel, body_labels=args.body_labels, vocab=vocab,),
        )
    )
    tb_logger = loggers.TensorBoardLogger(pjoin(args.outdir, "logs/"))
    trainer = Trainer.from_argparse_args(
        args, default_root_dir=args.outdir, logger=tb_logger, early_stop_callback=True,
    )
    if args.lr_find_plot:
        model.setup(None)
        lr_finder = trainer.lr_find(model)
        fig = lr_finder.plot(suggest=True)
        fig.show()
        input()
        return
    if not args.fast_dev_run and not args.overfit_batches:
        trainer.scale_batch_size(model, init_val=4096)
    trainer.fit(model)
    if args.test:
        trainer.test(ckpt_path=None)


if __name__ == "__main__":
    train()
