import math
import statistics
import warnings

import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import KFold
from asl_utils import combine_sequences


class ModelSelector(object):
    '''
    base class for model selection (strategy design pattern)
    '''

    def __init__(self, all_word_sequences: dict, all_word_Xlengths: dict, this_word: str,
                 n_constant=3,
                 min_n_components=2, max_n_components=10,
                 random_state=14, verbose=False):
        self.words = all_word_sequences
        self.hwords = all_word_Xlengths
        self.sequences = all_word_sequences[this_word]
        self.X, self.lengths = all_word_Xlengths[this_word]
        self.this_word = this_word
        self.n_constant = n_constant
        self.min_n_components = min_n_components
        self.max_n_components = max_n_components
        self.random_state = random_state
        self.verbose = verbose

    def select(self):
        raise NotImplementedError

    def base_model(self, num_states):
        # with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # warnings.filterwarnings("ignore", category=RuntimeWarning)
        try:
            hmm_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
            if self.verbose:
                print("model created for {} with {} states".format(self.this_word, num_states))
            return hmm_model
        except:
            if self.verbose:
                print("failure on {} with {} states".format(self.this_word, num_states))
            return None


class SelectorConstant(ModelSelector):
    """ select the model with value self.n_constant

    """

    def select(self):
        """ select based on n_constant value

        :return: GaussianHMM object
        """
        best_num_components = self.n_constant
        return self.base_model(best_num_components)


class SelectorBIC(ModelSelector):
    """ select the model with the lowest Bayesian Information Criterion(BIC) score

    http://www2.imm.dtu.dk/courses/02433/doc/ch6_slides.pdf
    Bayesian information criteria: BIC = -2 * logL + p * logN
    """

    def select(self):
        """ select the best model for self.this_word based on
        BIC score for n between self.min_n_components and self.max_n_components

        :return: GaussianHMM object
        """
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # TODO implement model selection based on BIC scores
        return min(
            [self.base_model(num_components) for num_components in range(self.min_n_components, self.max_n_components + 1)],
            key=self.calculate_score
        )

    def calculate_score(self, model):
        try:
            p = math.pow(model.n_components, 2) + 2 * len(self.X[0]) * model.n_components - 1
            logL = model.score(self.X, self.lengths)
            N = np.sum(self.lengths)
            score = -2 * logL + p * np.log(N)
            return score
        except:
            return float('Inf')


class SelectorDIC(ModelSelector):
    ''' select best model based on Discriminative Information Criterion

    Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
    Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
    https://pdfs.semanticscholar.org/ed3d/7c4a5f607201f3848d4c02dd9ba17c791fc2.pdf
    DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # TODO implement model selection based on DIC scores
        return min(
            [self.base_model(num_components) for num_components in range(self.min_n_components, self.max_n_components + 1)],
            key=self.calculate_score
        )

    def calculate_score(self, model):
        other_score = 0
        try:
            score = model.score(self.X, self.lengths) 
            other_words = filter(lambda word: word != self.this_word, self.words)
            for word in other_words:
                otherX, otherlength = self.hwords[word]
                try:
                    other_score += model.score(otherX, otherlength)
                except:
                    pass
            other_score = other_score / len(other_words)
            return score - other_score
        except:
            return float('Inf')


class SelectorCV(ModelSelector):
    ''' select best model based on average log Likelihood of cross-validation folds

    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # TODO implement model selection using CV
        def calculate_score(num_components):
            total_score = 0
            index = 0

            try:
                split_method = KFold(n_splits=min(3, len(self.sequences)))
            except:
                return float('Inf')

            for train_idx, test_idx in split_method.split(self.sequences):
                self.X, self.lengths = combine_sequences(train_idx, self.sequences)
                test_x, test_lengths = combine_sequences(test_idx, self.sequences)

                model = self.base_model(num_components)
                try:
                    total_score += model.score(test_x, test_lengths)
                    index = index + 1

                    return total_score / index
                except:
                    return float('Inf')

        best_num = min(
            range(self.min_n_components, self.max_n_components + 1),
            key=calculate_score
        )

        return self.base_model(best_num)
