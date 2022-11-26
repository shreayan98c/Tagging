# This file illustrates how you might experiment with the HMM interface at the prompt.
# You can also run it directly.
import os
import logging
import math
from pathlib import Path
from typing import Callable

from corpus import TaggedCorpus, desupervise, sentence_str
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
entrain = TaggedCorpus(Path("ensup"), Path("enraw"))  # all training
ensup = TaggedCorpus(Path("ensup"), tagset=entrain.tagset, vocab=entrain.vocab)  # supervised training
endev = TaggedCorpus(Path("endev"), tagset=entrain.tagset, vocab=entrain.vocab)  # evaluation
logging.info(f"Tagset: f{list(entrain.tagset)}")
known_vocab = TaggedCorpus(Path("ensup")).vocab  # words seen with supervised tags; used in evaluation

# Initialize an HMM
# normal
lexicon = build_lexicon(entrain, embeddings_file=Path('../lexicons/words-50.txt'))  # works better with more attributes!
crf = CRFModel(ensup.tagset, ensup.vocab, lexicon, birnn=True)  # not changing the name for convenience
logging.info("Running on CRF Model")

# Let's initialize with supervised training to approximately maximize the
# regularized log-likelihood.  If you want to speed this up, you can increase
# the tolerance of training (using the `tolerance` argument), since we don't
# really have to train to convergence.
loss_sup = lambda model: model_cross_entropy(model, eval_corpus=ensup)
crf.train(corpus=ensup, loss=loss_sup, minibatch_size=32, evalbatch_size=10000, lr=0.00015, reg=2,
          save_path=Path('en_crf_birnn.pkl'))
logging.info(f"sup error rate is: {model_error_rate(crf, eval_corpus=ensup, known_vocab=known_vocab)}")

# More detailed look at the first 10 sentences in the held-out corpus,
# including Viterbi tagging.
for m, sentence in enumerate(endev):
    if m >= 10: break
    viterbi = crf.viterbi_tagging(desupervise(sentence), endev)
    counts = eval_tagging(predicted=viterbi, gold=sentence,
                          known_vocab=known_vocab)
    num = counts['NUM', 'ALL']
    denom = counts['DENOM', 'ALL']

    logging.info(f"Gold:    {sentence_str(sentence)}")
    logging.info(f"Viterbi: {sentence_str(viterbi)}")
    logging.info(f"Loss:    {denom - num}/{denom}")
    logging.info(f"Prob:    {math.exp(crf.log_prob(sentence, endev))}")
