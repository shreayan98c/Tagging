#!/usr/bin/env python3

# CS465 at Johns Hopkins University.
# Implementation of Hidden Markov Models.

from __future__ import annotations
import logging
from math import inf, log, exp, sqrt
from pathlib import Path
from typing import Callable, List, Optional, Tuple, cast

import torch
from torch import Tensor as Tensor
from torch import tensor as tensor
from torch import optim as optim
from torch import nn as nn
from torch import cuda as cuda
from torch.nn import functional as F
from torch.nn.parameter import Parameter as Parameter
from torchtyping import TensorType, patch_typeguard
from typeguard import typechecked
from tqdm import tqdm # type: ignore

from logsumexp_safe import *
from corpus import (BOS_TAG, BOS_WORD, EOS_TAG, EOS_WORD, Sentence, Tag, TaggedCorpus, Word)
from integerize import Integerizer

logger = logging.getLogger(Path(__file__).stem)  # For usage, see findsim.py in earlier assignment.
# Note: We use the name "logger" this time rather than "log" since we
# are already using "log" for the mathematical log!

# Set the seed for random numbers in torch, for replicability
torch.manual_seed(1337)
cuda.manual_seed(69_420)  # No-op if CUDA isn't available

patch_typeguard()   # makes @typechecked work with torchtyping

###
# HMM tagger
###


class HiddenMarkovModel(nn.Module):
    """An implementation of an HMM, whose emission probabilities are
    parameterized using the word embeddings in the lexicon.
    
    We'll refer to the HMM states as "tags" and the HMM observations 
    as "words."
    """

    def __init__(self,
                 tagset: Integerizer[Tag],
                 vocab: Integerizer[Word],
                 lexicon: Tensor,
                 unigram: bool = False,
                 awesome: bool = False,
                 affixes: bool = False):
        """Construct an HMM with initially random parameters, with the
        given tagset, vocabulary, and lexical features.
        
        Normally this is an ordinary first-order (bigram) HMM.  The unigram
        flag says to fall back to a zeroth-order HMM, in which the different
        positions are generated independently.  (The code could be extended
        to support higher-order HMMs: trigram HMMs used to be popular.)"""

        super().__init__() # type: ignore # pytorch nn.Module does not have type annotations

        # We'll use the variable names that we used in the reading handout, for
        # easy reference.  (It's typically good practice to use more descriptive names.)
        # As usual in Python, attributes starting with _ are intended as private;
        # in this case, they might go away if you changed the parametrization of the model.

        # We omit EOS_WORD and BOS_WORD from the vocabulary, as they can never be emitted.
        # See the reading handout section "Don't guess when you know."

        assert vocab[-2:] == [EOS_WORD, BOS_WORD]  # make sure these are the last two

        self.k = len(tagset)       # number of tag types
        self.V = len(vocab) - 2    # number of word types (not counting EOS_WORD and BOS_WORD)
        self.d = lexicon.size(1)   # dimensionality of a word's embedding in attribute space
        self.unigram = unigram     # do we fall back to a unigram model?
        self.awesome = awesome     # open ended improvements
        self.affixes = affixes     # open ended improvements

        self.tagset = tagset
        self.vocab = vocab
        self._E = lexicon[:-2]  # embedding matrix; omits rows for EOS_WORD and BOS_WORD

        # Useful constants that are invoked in the methods
        self.bos_t: Optional[int] = tagset.index(BOS_TAG)
        self.eos_t: Optional[int] = tagset.index(EOS_TAG)
        assert self.bos_t is not None    # we need this to exist
        assert self.eos_t is not None    # we need this to exist
        self.eye: Tensor = torch.eye(self.k)  # identity matrix, used as a collection of one-hot tag vectors

        self.init_params()     # create and initialize params


    @property
    def device(self) -> torch.device:
        """Get the GPU (or CPU) our code is running on."""
        # Why the hell isn't this already in PyTorch?
        return next(self.parameters()).device

    def _integerize_sentence(self, sentence: Sentence, corpus: TaggedCorpus) -> List[Tuple[int,Optional[int]]]:
        """Integerize the words and tags of the given sentence, which came from the given corpus."""

        # Make sure that the sentence comes from a corpus that this HMM knows
        # how to handle.
        if corpus.tagset != self.tagset or corpus.vocab != self.vocab:
            raise TypeError("The corpus that this sentence came from uses a different tagset or vocab")

        # If so, go ahead and integerize it.
        return corpus.integerize_sentence(sentence)

    def init_params(self) -> None:
        """Initialize params to small random values (which breaks ties in the fully unsupervised case).
        However, we initialize the BOS_TAG column of _WA to -inf, to ensure that
        we have 0 probability of transitioning to BOS_TAG (see "Don't guess when you know").
        See the "Parametrization" section of the reading handout."""

        # See the reading handout section "Parametrization.""

        ThetaB = 0.01*torch.rand(self.k, self.d)
        self._ThetaB = Parameter(ThetaB)    # params used to construct emission matrix

        WA = 0.01*torch.rand(1 if self.unigram # just one row if unigram model
                             else self.k,      # but one row per tag s if bigram model
                             self.k)           # one column per tag t
        WA[:, self.bos_t] = -inf               # correct the BOS_TAG column
        self._WA = Parameter(WA)               # params used to construct transition matrix


    @typechecked
    def params_L2(self) -> TensorType[()]:
        """What's the L2 norm of the current parameter vector?
        We consider only the finite parameters."""
        l2 = tensor(0.0)
        for x in self.parameters():
            x_finite = x[x.isfinite()]
            l2 = l2 + x_finite @ x_finite   # add ||x_finite||^2
        return l2

    def updateAB(self) -> None:
        """Set the transition and emission matrices A and B, based on the current parameters.
        See the "Parametrization" section of the reading handout."""

        A = F.softmax(self._WA, dim=1)       # run softmax on params to get transition distributions
                                             # note that the BOS_TAG column will be 0, but each row will sum to 1
        if self.unigram:
            # A is a row vector giving unigram probabilities p(t).
            # We'll just set the bigram matrix to use these as p(t | s)
            # for every row s.  This lets us simply use the bigram
            # code for unigram experiments, although unfortunately that
            # preserves the O(nk^2) runtime instead of letting us speed 
            # up to O(nk).
            self.A = A.repeat(self.k, 1)
        else:
            # A is already a full matrix giving p(t | s).
            self.A = A

        WB = self._ThetaB @ self._E.t()  # inner products of tag weights and word embeddings
        B = F.softmax(WB, dim=1)         # run softmax on those inner products to get emission distributions
        self.B = B.clone()
        self.B[self.eos_t, :] = 0  # but don't guess: EOS_TAG can't emit any column's word (only EOS_WORD)
        self.B[self.bos_t, :] = 0  # same for BOS_TAG (although BOS_TAG will already be ruled out by other factors)
        self.A = self.A + 1e-45
        self.B = self.B + 1e-45

    def printAB(self) -> None:
        """Print the A and B matrices in a more human-readable format (tab-separated)."""
        print("Transition matrix A:")
        col_headers = [""] + [str(self.tagset[t]) for t in range(self.A.size(1))]
        print("\t".join(col_headers))
        for s in range(self.A.size(0)):   # rows
            row = [str(self.tagset[s])] + [f"{self.A[s,t]:.3f}" for t in range(self.A.size(1))]
            print("\t".join(row))
        print("\nEmission matrix B:")
        col_headers = [""] + [str(self.vocab[w]) for w in range(self.B.size(1))]
        print("\t".join(col_headers))
        for t in range(self.A.size(0)):   # rows
            row = [str(self.tagset[t])] + [f"{self.B[t,w]:.3f}" for w in range(self.B.size(1))]
            print("\t".join(row))
        print("\n")


    @typechecked
    def log_prob(self, sentence: Sentence, corpus: TaggedCorpus) -> TensorType[()]:
        """Compute the log probability of a single sentence under the current
        model parameters.  If the sentence is not fully tagged, the probability
        will marginalize over all possible tags.  

        When the logging level is set to DEBUG, the alpha and beta vectors and posterior counts
        are logged.  You can check this against the ice cream spreadsheet."""
        return self.log_forward(sentence, corpus)


    @typechecked
    def log_forward(self, sentence: Sentence, corpus: TaggedCorpus) -> TensorType[()]:
        """Run the forward algorithm from the handout on a tagged, untagged, 
        or partially tagged sentence.  Return log Z (the log of the forward 
        probability).

        The corpus from which this sentence was drawn is also passed in as an
        argument, to help with integerization and check that we're 
        integerizing correctly."""

        sent = self._integerize_sentence(sentence, corpus)

        # you fill this in!
        # The "nice" way to construct alpha is by appending to a List[Tensor] at each
        # step.  But to better match the notation in the handout, we'll instead preallocate
        # a list of length n+2 so that we can assign directly to alpha[j].
        # alpha = [torch.empty(self.k) for _ in sent]  # 0
        alpha = [-float("Inf") * torch.ones(self.k) for _ in sent]  # -ve infinity

        n = len(sent)

        # print([self.vocab[i] for (i,j) in sent])
        # print([self.tagset[j] for (i,j) in sent])

        # for i in range(self.k):
        #     for j in range(self.k):
        #         print((i, j), self.tagset[i]," to ", self.tagset[j])
        # for i in range(self.k):
        #     for j in range(self.V):
        #         print((i, j), self.tagset[i], " to ", self.vocab[j])

        assert sent[0][1] == self.bos_t  # ensure that the sent starts with <BOS> tag
        assert sent[-1][1] == self.eos_t  # ensure that the sent ends with <EOS> tag
        alpha[0][self.bos_t] = 0

        # Z = A . B
        # log Z = log A + log B

        for j in range(1, n):
            (prev_word, prev_tag) = sent[j-1]
            (curr_word, curr_tag) = sent[j]
            alpha_transition = alpha[j-1].reshape(-1, 1) + torch.log(self.A)

            if prev_tag is None:
                # unsupervised training or partially supervised training
                if curr_tag == self.eos_t:
                    alpha[j] = logsumexp_new(alpha_transition, dim=0, safe_inf=True)
                else:
                    alpha[j] = logsumexp_new(alpha_transition, dim=0, safe_inf=True) + torch.log(self.B[:, curr_word])
            else:
                # supervised training if the tag is present
                if curr_tag == self.eos_t:
                    alpha[j] = alpha_transition[prev_tag, :]
                else:
                    alpha[j] = alpha_transition[prev_tag] + torch.log(self.B[:, curr_word])

        # Z = alpha[n-1][self.eos_t]
        # return Z
        log_Z = alpha[n-1][self.eos_t]
        return log_Z

    def viterbi_tagging(self, sentence: Sentence, corpus: TaggedCorpus) -> Sentence:
        """Find the most probable tagging for the given sentence, according to the
        current model."""

        # Note: This code is mainly copied from the forward algorithm.
        # We just switch to using max, and follow backpointers.
        # I've continued to call the vector alpha rather than mu.

        sent = self._integerize_sentence(sentence, corpus)

        # you fill this in!
        n = len(sent)

        mu = [-float("Inf") * torch.ones(self.k) for _ in sent]  # setting mu to -inf
        backpointers = [torch.empty(self.k) for _ in sent]  # dictionary storing index to the tag

        assert sent[0][1] == self.bos_t  # ensure that the sent starts with <BOS> tag
        assert sent[-1][1] == self.eos_t  # ensure that the sent ends with <EOS> tag

        # support for bos tag
        mu[0][self.bos_t] = 0

        for j in range(1, n-1):
            (curr_word, curr_tag) = sent[j]
            max_prob = torch.max((mu[j-1].reshape(-1, 1) + torch.log(self.A)) + torch.log(self.B[:, curr_word]), 0)
            # mu[j], backpointers[j] = max_prob
            mu[j] = max_prob.values
            backpointers[j] = max_prob.indices

        # support for eos tag
        # sent[-1] is the <EOS>
        # sent[-2] is the last word before <EOS>
        max_prob = torch.max(mu[n-2].reshape(-1, 1) + torch.log(self.A), 0)
        # backpointer from last word to the <EOS> tag
        # mu[n-1], backpointers[n-1] = max_prob
        mu[n-1] = max_prob.values
        backpointers[n-1] = max_prob.indices

        # follow backpointers to find the best sequence
        previous_tag = self.eos_t
        words = []
        for i in range(n-1, -1, -1):
            curr_word, curr_tag = sent[i]
            word = self.vocab[curr_word]
            tag = self.tagset[previous_tag]
            previous_tag = backpointers[i][previous_tag]
            words.append((word, tag))
            # print(f"Current: {tag}, Prev: {previous_tag}")
        words.reverse()
        sent = Sentence(words)
        return sent

    def train(self,
              corpus: TaggedCorpus,
              loss: Callable[[HiddenMarkovModel], float],
              tolerance: float =0.001,
              minibatch_size: int = 1,
              evalbatch_size: int = 500,
              lr: float = 1.0,
              reg: float = 0.0,
              save_path: Path = Path("my_hmm.pkl")) -> None:
        """Train the HMM on the given training corpus, starting at the current parameters.
        The minibatch size controls how often we do an update.
        (Recommended to be larger than 1 for speed; can be inf for the whole training corpus.)
        The evalbatch size controls how often we evaluate (e.g., on a development corpus).
        We will stop when evaluation loss is not better than the last evalbatch by at least the
        tolerance; in particular, we will stop if we evaluation loss is getting worse (overfitting).
        lr is the learning rate, and reg is an L2 batch regularization coefficient."""

        # This is relatively generic training code.  Notice however that the
        # updateAB step before each minibatch produces A, B matrices that are
        # then shared by all sentences in the minibatch.

        # All of the sentences in a minibatch could be treated in parallel,
        # since they use the same parameters.  The code below treats them
        # in series, but if you were using a GPU, you could get speedups
        # by writing the forward algorithm using higher-dimensional tensor 
        # operations that update alpha[j-1] to alpha[j] for all the sentences
        # in the minibatch at once, and then PyTorch could actually take
        # better advantage of hardware parallelism.

        assert minibatch_size > 0
        if minibatch_size > len(corpus):
            minibatch_size = len(corpus)  # no point in having a minibatch larger than the corpus
        assert reg >= 0

        old_dev_loss: Optional[float] = None    # we'll keep track of the dev loss here

        optimizer = optim.SGD(self.parameters(), lr=lr)  # optimizer knows what the params are
        self.updateAB()                                        # compute A and B matrices from current params
        log_likelihood = tensor(0.0, device=self.device)       # accumulator for minibatch log_likelihood
        for m, sentence in tqdm(enumerate(corpus.draw_sentences_forever())):
            # Before we process the new sentence, we'll take stock of the preceding
            # examples.  (It would feel more natural to do this at the end of each
            # iteration instead of the start of the next one.  However, we'd also like
            # to do it at the start of the first time through the loop, to print out
            # the dev loss on the initial parameters before the first example.)

            # m is the number of examples we've seen so far.
            # If we're at the end of a minibatch, do an update.
            if m % minibatch_size == 0 and m > 0:
                logger.debug(f"Training log-likelihood per example: {log_likelihood.item()/minibatch_size:.3f} nats")
                optimizer.zero_grad()          # backward pass will add to existing gradient, so zero it
                objective = -log_likelihood + (minibatch_size/corpus.num_tokens()) * reg * self.params_L2()
                objective.backward()           # type: ignore # compute gradient of regularized negative log-likelihod
                length = sqrt(sum((x.grad*x.grad).sum().item() for x in self.parameters()))
                logger.debug(f"Size of gradient vector: {length}")  # should approach 0 for large minibatch at local min
                optimizer.step()               # SGD step
                self.updateAB()                # update A and B matrices from new params
                log_likelihood = tensor(0.0, device=self.device)    # reset accumulator for next minibatch

            # If we're at the end of an eval batch, or at the start of training, evaluate.
            if m % evalbatch_size == 0:
                with torch.no_grad():       # type: ignore # don't retain gradients during evaluation
                    dev_loss = loss(self)   # this will print its own log messages
                if old_dev_loss is not None and dev_loss >= old_dev_loss * (1-tolerance):
                    # we haven't gotten much better, so stop
                    self.save(save_path)  # Store this model, in case we'd like to restore it later.
                    break
                old_dev_loss = dev_loss            # remember for next eval batch

            # Finally, add likelihood of sentence m to the minibatch objective.
            log_likelihood = log_likelihood + self.log_prob(sentence, corpus)

    def save(self, destination: Path) -> None:
        import pickle
        with open(destination, mode="wb") as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info(f"Saved model to {destination}")


    @classmethod
    def load(cls, source: Path) -> HiddenMarkovModel:
        import pickle  # for loading/saving Python objects
        logger.info(f"Loading model from {source}")
        with open(source, mode="rb") as f:
            result = pickle.load(f)
            if not isinstance(result, cls):
                raise ValueError(f"Type Error: expected object of type {cls} but got {type(result)} from pickled file.")
            logger.info(f"Loaded model from {source}")
            return result
