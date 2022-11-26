# This file illustrates how you might experiment with the HMM interface at the prompt.
# You can also run it directly.
import os
import logging
import math
from pathlib import Path
from typing import Callable

from corpus import TaggedCorpus
from eval import eval_tagging, model_cross_entropy, model_error_rate
from crf import CRFModel
from lexicon import build_lexicon
import torch

# Set up logging
logging.basicConfig(format="%(levelname)s : %(message)s", level=logging.INFO)  # could change INFO to DEBUG
# torch.autograd.set_detect_anomaly(True)    # uncomment to improve error messages from .backward(), but slows down

# Switch working directory to the directory where the data live.  You may want to edit this line.
os.chdir("../data")

# Get the corpora
icsup = TaggedCorpus(Path("icsup"), add_oov=False)
logging.info(f"Ice cream vocabulary: {list(icsup.vocab)}")
logging.info(f"Ice cream tagset: {list(icsup.tagset)}")

# Initialize a CRF
# normal
lexicon = build_lexicon(icsup, one_hot=True)  # works better with more attributes!
crf = CRFModel(icsup.tagset, icsup.vocab, lexicon)  # not changing the name for convenience
logging.info("Running on CRF Model")

# Let's initialize with supervised training to approximately maximize the
# regularized log-likelihood.  If you want to speed this up, you can increase
# the tolerance of training (using the `tolerance` argument), since we don't
# really have to train to convergence.
loss_sup = lambda model: model_cross_entropy(model, eval_corpus=icsup)
crf.train(corpus=icsup, loss=loss_sup, minibatch_size=10, evalbatch_size=500, lr=0.0001, reg=1,
          save_path=Path('ic_crf.pkl'))
logging.info(f"dev error rate is: {model_error_rate(crf, eval_corpus=icsup)}")
