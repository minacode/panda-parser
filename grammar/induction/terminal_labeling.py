from abc import ABCMeta, abstractmethod
from collections import defaultdict
from hybridtree.monadic_tokens import MonadicToken
from discodop.lexicon import getunknownwordmodel, unknownword4, replaceraretestwords, YEARRE, NUMBERRE, UNK
import string


class TerminalLabeling:
    __metaclass__ = ABCMeta

    @abstractmethod
    def token_label(self, token, _loc=None):
        """
        :type token: MonadicToken
        :type _loc: int
        """
        pass

    def token_tree_label(self, token):
        if token.type() == "CONLL-X":
            return self.token_label(token) + " : " + token.deprel()
        elif token.type() == "CONSTITUENT-CATEGORY":
            return token.category() + " : " + token.edge()
        else:
            return self.token_label(token) + " : " + token.edge()

    def prepare_parser_input(self, tokens):
        return [self.token_label(token, _loc) for _loc, token in enumerate(tokens)]

# class ConstituentTerminalLabeling(TerminalLabeling):
#     def token_label(self, token):
#         if isinstance(token, ConstituentTerminal):
#             return token.pos()
#         elif isinstance(token, ConstituentCategory):
#             return token.category()
#         else:
#             assert False


class FeatureTerminals(TerminalLabeling):
    def __init__(self, token_to_features, feature_filter):
        self.token_to_features = token_to_features
        self.feature_filter = feature_filter

    def token_label(self, token, _loc=None):
        isleaf = token.type() == "CONSTITUENT-TERMINAL"
        feat_list = self.token_to_features(token, isleaf)
        features = self.feature_filter([feat_list])
        return "[" + (",".join([str(key) + ':' + str(val) for key, val in features]))\
            .translate(str.maketrans('', '', ' ')) + "]"

    def __str__(self):
        return "feature-terminals"


class FrequencyBiasedTerminalLabeling(TerminalLabeling):
    def __init__(self, fine_labeling, fall_back, corpus, threshold):
        self.fine_labeling = fine_labeling
        self.fall_back = fall_back
        self.fine_label_count = defaultdict(lambda: 0)
        self.threshold = threshold
        self.backoff_mode = False
        for tree in corpus:
            for token in tree.token_yield():
                label = self.fine_labeling.token_label(token)
                self.fine_label_count[label] += 1
        self.fine_label_count = {label for label in self.fine_label_count if self.fine_label_count[label] >= threshold}

    def token_label(self, token, _loc=None):
        fine_label = self.fine_labeling.token_label(token)
        if not self.backoff_mode and fine_label in self.fine_label_count:
            return fine_label
        else:
            return self.fall_back.token_label(token)

    def __str__(self):
        return "frequency-biased["\
               + str(self.threshold) \
               + '|' + str(self.fine_labeling) \
               + '|' + str(self.fall_back) + "]"


class FormTerminals(TerminalLabeling):
    def token_label(self, token, _loc=None):
        return token.form()

    def __str__(self):
        return 'form'


class CPosTerminals(TerminalLabeling):
    def token_label(self, token, _loc=None):
        return token.cpos()

    def __str__(self):
        return 'cpos'


class PosTerminals(TerminalLabeling):
    def token_label(self, token, _loc=None):
        return token.pos()

    def __str__(self):
        return 'pos'


class CPOS_KON_APPR(TerminalLabeling):
    def token_label(self, token, _loc=None):
        cpos = token.pos()
        if cpos in ['KON', 'APPR']:
            return cpos + token.form().lower()
        else:
            return cpos

    def __str__(self):
        return 'cpos-KON-APPR'


class FormTerminalsUnk(TerminalLabeling):
    def __init__(self, trees, threshold, UNK="UNKNOWN", filter=[]):
        """
        :param trees: corpus of trees
        :param threshold: UNK words below the threshold
        :type threshold: int
        :param UNK: representation string of UNK in grammar
        :type UNK: str
        :param filter: a list of POS tags which are always UNKed
        :type filter: list[str]
        """
        self.__terminal_counts = {}
        self.__UNK = UNK
        self.__threshold = threshold
        for tree in trees:
            for token in tree.token_yield():
                if token.pos() not in filter:
                    key = token.form().lower()
                    if key in self.__terminal_counts:
                        self.__terminal_counts[key] += 1
                    else:
                        self.__terminal_counts[key] = 1

    def __str__(self):
        return 'form-unk-' + str(self.__threshold)

    def token_label(self, token, _loc=None):
        form = token.form().lower()
        if self.__terminal_counts.get(form, 0) < self.__threshold:
            form = self.__UNK
        return form


class FormTerminalsPOS(TerminalLabeling):
    def __init__(self, trees, threshold, filter=[]):
        """
        :param trees: corpus of trees
        :param threshold: UNK words below the threshold
        :type threshold: int
        :param UNK: representation string of UNK in grammar
        :type UNK: str
        :param filter: a list of POS tags which are always UNKed
        :type filter: list[str]
        """
        self.__terminal_counts = {}
        self.__threshold = threshold
        for tree in trees:
            for token in tree.token_yield():
                if token.pos() not in filter:
                    key = token.form().lower()
                    if key in self.__terminal_counts:
                        self.__terminal_counts[key] += 1
                    else:
                        self.__terminal_counts[key] = 1

    def __str__(self):
        return 'form-POS-' + str(self.__threshold)

    def token_label(self, token, _loc=None):
        form = token.form().lower()
        if self.__terminal_counts.get(form, 0) < self.__threshold:
            form = token.pos()
        return form


class FormPosTerminalsUnk(TerminalLabeling):
    def __init__(self, trees, threshold, UNK="UNKNOWN", filter=[]):
        """
        :param trees: corpus of trees
        :param threshold: UNK words below the threshold
        :type threshold: int
        :param UNK: representation string of UNK in grammar
        :type UNK: str
        :param filter: a list of POS tags which are always UNKed
        :type filter: list[str]
        """
        self.__terminal_counts = {}
        self.__UNK = UNK
        self.__threshold = threshold
        for tree in trees:
            for token in tree.token_yield():
                if token.pos() not in filter:
                    key = (token.form().lower(), token.pos())
                    if key in self.__terminal_counts:
                        self.__terminal_counts[key] += 1
                    else:
                        self.__terminal_counts[key] = 1

    def __str__(self):
        return 'form-pos-unk-' + str(self.__threshold) + '-pos'

    def token_label(self, token, _loc=None):
        pos = token.pos()
        form = token.form().lower()
        if self.__terminal_counts.get((form, pos), 0) < self.__threshold:
            form = self.__UNK
        return form + '-:-' + pos


class FormPosTerminalsUnkMorph(TerminalLabeling):
    def __init__(self, trees, threshold, UNK="UNKNOWN", filter=[], add_morph={}):
        self.__terminal_counts = defaultdict(lambda: 0)
        self.__UNK = UNK
        self.__threshold = threshold
        self.__add_morph = add_morph
        for tree in trees:
            for token in tree.token_yield():
                if token.pos() not in filter:
                    self.__terminal_counts[(token.form().lower(), token.pos())] += 1

    def __str__(self):
        return 'form-pos-unk-' + str(self.__threshold) + '-morph-pos'

    def token_label(self, token, _loc=None):
        pos = token.pos()
        form = token.form().lower()
        if self.__terminal_counts.get((form, pos)) < self.__threshold:
            form = self.__UNK
            if pos in self.__add_morph:
                feats = map(lambda x: tuple(x.split('=')), token.feats().split('|'))
                for feat in feats:
                    if feat[0] in self.__add_morph[pos]:
                        form += '#' + feat[0] + ':' + feat[1]
        return form + '-:-' + pos


class StanfordUNKing(TerminalLabeling):
    def __init__(self, trees, unknown_threshold=4, openclass_threshold=150):
        self.unknown_threshold = unknown_threshold
        self.openclass_threshold = openclass_threshold
        sentences = []
        for tree in trees:
            sentence = []
            for token in tree.token_yield():
                sentence.append((token.form(), token.pos()))
            sentences.append(sentence)
        (sigs, words, lexicon, wordsfortag, openclasstags,
            openclasswords, tags, wordtags,
            wordsig, sigtag), msg \
            = getunknownwordmodel(sentences, unknownword4, self.unknown_threshold, self.openclass_threshold)
        self.openclasswords = openclasswords
        self.sigs = sigs
        self.words = words
        self.lexicon = lexicon
        self.backoff_mode = False

    def __str__(self):
        return "stanford-unk-" + str(self.unknown_threshold) \
               + "-openclass-" + str(self.openclass_threshold)

    def token_label(self, token, _loc=None):
        word = token.form()

        # adapted from discodop
        if YEARRE.match(word):
            return '1970'
        elif NUMBERRE.match(word):
            return '000'
        if not self.backoff_mode:
            if word in self.lexicon:
                return word
            elif word.lower() in self.lexicon:
                return word.lower()
            else:
                sig = unknownword4(word, _loc, self.lexicon)
                if sig in self.sigs:
                    return sig
                else:
                    return UNK
        else:
            if word in self.lexicon and word not in self.openclasswords:
                return word
            elif word.lower() in self.lexicon and word not in self.openclasswords:
                return word.lower()
            else:
                sig = unknownword4(word, _loc, self.lexicon)
                if sig in self.sigs:
                    return sig
                else:
                    return UNK


class TerminalLabelingFactory:
    def __init__(self):
        self.__strategies = {}

    def register_strategy(self, name, strategy):
        """
        :type name: str
        :type strategy: TerminalLabeling
        """
        self.__strategies[name] = strategy

    def get_strategy(self, name):
        """
        :type name: str
        :rtype: TerminalLabeling
        """
        return self.__strategies[name]


def the_terminal_labeling_factory():
    """
    :rtype : TerminalLabelingFactory
    """
    factory = TerminalLabelingFactory()
    factory.register_strategy('form', FormTerminals())
    factory.register_strategy('pos', PosTerminals())
    factory.register_strategy('cpos', CPosTerminals())
    factory.register_strategy('cpos-KON-APPR', CPOS_KON_APPR())
    return factory