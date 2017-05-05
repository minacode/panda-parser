from libcpp.map cimport map
from libcpp.memory cimport make_shared
from cython.operator cimport dereference as deref
from parser.commons.commons cimport *
from parser.trace_manager.trace_manager cimport PyTraceManager, TraceManagerPtr
from parser.trace_manager.sm_trainer_util cimport PyGrammarInfo, GrammarInfo2, PyStorageManager
from parser.trace_manager.score_validator cimport PyCandidateScoreValidator, CandidateScoreValidator
import time
import random
import grammar.lcfrs as gl
import itertools

DEF ENCODE_NONTERMINALS = True
DEF ENCODE_TERMINALS = True

cdef extern from "Trainer/EMTrainer.h" namespace "Trainer":
    cdef cppclass EMTrainer[Nonterminal, TraceID]:
        vector[double] do_em_training[SemiRing](vector[double], vector[vector[unsigned_int]], unsigned)

cdef extern from "Trainer/SplitMergeTrainer.h" namespace "Trainer":
    cdef cppclass Splitter:
        void reset_random_seed(unsigned_int seed)
    cdef cppclass SplitMergeTrainer[Nonterminal, TraceID]:
        shared_ptr[Splitter] splitter
        LatentAnnotation split_merge_cycle(LatentAnnotation)

cdef extern from "Trainer/EMTrainerLA.h" namespace "Trainer":
    ctypedef enum TrainingMode:
        Default,
        Splitting,
        Merging,
        Smoothing

    cdef cppclass EMTrainerLA:
        void setEMepochs(unsigned epochs, TrainingMode mode)
    cdef cppclass EMTrainerLAValidation(EMTrainerLA):
        void setMaxDrops(unsigned maxDrops, TrainingMode mode)

cdef extern from "Trainer/MergePreparator.h":
    pass

cdef extern from "Trainer/TrainerBuilder.h" namespace "Trainer":
    cdef cppclass EMTrainerBuilder:
         EMTrainer[Nonterminal, TraceID] build_em_trainer[Nonterminal, TraceID](TraceManagerPtr[Nonterminal, TraceID])

    cdef cppclass SplitMergeTrainerBuilder[Nonterminal, TraceID]:
        SplitMergeTrainerBuilder(TraceManagerPtr[Nonterminal, TraceID], shared_ptr[GrammarInfo2])
        SplitMergeTrainerBuilder& set_simple_expector()
        SplitMergeTrainerBuilder& set_simple_expector(unsigned_int)
        SplitMergeTrainerBuilder& set_discriminative_expector(
                TraceManagerPtr[Nonterminal, TraceID] discriminativeTraceManager
                , double maxScale)
        SplitMergeTrainerBuilder& set_discriminative_expector(
                TraceManagerPtr[Nonterminal, TraceID] discriminativeTraceManager
                , double maxScale
                , unsigned_int threads)
        SplitMergeTrainerBuilder& set_simple_maximizer()
        SplitMergeTrainerBuilder& set_simple_maximizer(unsigned_int)
        SplitMergeTrainerBuilder& set_em_epochs(unsigned_int)
        SplitMergeTrainerBuilder& set_simple_validator(
                TraceManagerPtr[Nonterminal, TraceID] discriminativeTraceManager
                , unsigned_int maxDrops)
        SplitMergeTrainerBuilder& set_simple_validator(
                TraceManagerPtr[Nonterminal, TraceID] discriminativeTraceManager
                , unsigned_int maxDrops
                , unsigned_int threads)
        SplitMergeTrainerBuilder& set_score_validator(shared_ptr[CandidateScoreValidator[Nonterminal, TraceID]], unsigned)
        SplitMergeTrainerBuilder& set_percent_merger(double)
        SplitMergeTrainerBuilder& set_percent_merger(double, unsigned)
        SplitMergeTrainerBuilder& set_threshold_merger(double)
        SplitMergeTrainerBuilder& set_threshold_merger(double, unsigned)
        SplitMergeTrainerBuilder& set_split_randomization(double, unsigned)
        SplitMergeTrainerBuilder& set_smoothing_factor(double smoothingFactor)
        SplitMergeTrainerBuilder& set_threads(unsigned_int)
        SplitMergeTrainer[Nonterminal, TraceID] build()
        shared_ptr[EMTrainerLA] getEmTrainer()

cdef extern from "util.h":
    cdef cppclass Double
    cdef cppclass LogDouble
    cdef void output_helper(string)

# choose representation: prob / log-prob
# ctypedef LogDouble SemiRing
ctypedef Double SemiRing

cdef class PyEMTrainer:
    cdef PyTraceManager traceManager

    def __init__(self, PyTraceManager traceManager):
        self.traceManager = traceManager

    def em_training(self, grammar, n_epochs, init="rfe", tie_breaking=False, sigma=0.005, seed=0):
        random.seed(seed)
        assert isinstance(grammar, gl.LCFRS)
        normalization_groups = []
        rule_to_group = {}
        for nont in grammar.nonts():
            normalization_group = []
            for rule in grammar.lhs_nont_to_rules(nont):
                rule_idx = rule.get_idx()
                normalization_group.append(rule_idx)
                rule_to_group[rule_idx] = self.traceManager.get_nonterminal_map().object_index(nont)
            normalization_groups.append(normalization_group)
        initial_weights = [0.0] * 0
        for i in range(0, len(grammar.rule_index())):
            if init == "rfe":
                prob = grammar.rule_index(i).weight()
            elif init == "equal" or True:
                prob = 1.0 / len(normalization_groups[rule_to_group[i]])

            # this may violates properness
            # but EM makes the grammar proper again
            if tie_breaking:
                prob_new = random.gauss(prob, sigma)
                while prob_new < 0.0:
                    prob_new = random.gauss(prob, sigma)
                prob = prob_new

            initial_weights.append(prob)

        cdef EMTrainerBuilder trainerBuilder
        cdef shared_ptr[EMTrainer[NONTERMINAL, size_t]] emTrainer \
            = make_shared[EMTrainer[NONTERMINAL, size_t]](trainerBuilder.build_em_trainer[NONTERMINAL, size_t](self.traceManager.trace_manager))

        final_weights = deref(emTrainer).do_em_training[SemiRing](initial_weights, normalization_groups, n_epochs)

        for i in range(0, len(grammar.rule_index())):
            grammar.rule_index(i).set_weight(final_weights[i])


cdef class PySplitMergeTrainerBuilder:
    cdef shared_ptr[SplitMergeTrainerBuilder[NONTERMINAL, size_t]] splitMergeTrainerBuilder
    def __init__(self, PyTraceManager traceManager, PyGrammarInfo grammarInfo):
        self.splitMergeTrainerBuilder = make_shared[SplitMergeTrainerBuilder[NONTERMINAL, size_t]](traceManager.trace_manager, grammarInfo.grammarInfo)

    cpdef PySplitMergeTrainerBuilder set_threads(self, unsigned_int threads):
        deref(self.splitMergeTrainerBuilder).set_threads(threads)
        return self

    cpdef PySplitMergeTrainerBuilder set_simple_expector(self, unsigned_int threads=0):
        if threads > 0:
            deref(self.splitMergeTrainerBuilder).set_simple_expector(threads)
        else:
            deref(self.splitMergeTrainerBuilder).set_simple_expector()
        return self

    cpdef PySplitMergeTrainerBuilder set_discriminative_expector(
            self
            , PyTraceManager discriminativeTraces
            , double maxScale = float("inf")
            , unsigned_int threads=0
    ):
        if threads > 0:
            deref(self.splitMergeTrainerBuilder)\
                .set_discriminative_expector(discriminativeTraces.trace_manager, maxScale, threads)
        else:
            deref(self.splitMergeTrainerBuilder).set_discriminative_expector(discriminativeTraces.trace_manager, maxScale)
        return self

    cpdef PySplitMergeTrainerBuilder set_simple_maximizer(self, unsigned_int threads=0):
        if threads > 0:
            deref(self.splitMergeTrainerBuilder).set_simple_maximizer(threads)
        else:
            deref(self.splitMergeTrainerBuilder).set_simple_maximizer()
        return self

    cpdef PySplitMergeTrainerBuilder set_em_epochs(self, size_t epochs):
        deref(self.splitMergeTrainerBuilder).set_em_epochs(epochs)
        return self

    cpdef PySplitMergeTrainerBuilder set_simple_validator(
            self
            , PyTraceManager discriminativeTraces
            , unsigned_int maxDrops=6
            , unsigned_int threads=0
    ):
        if threads > 0:
            deref(self.splitMergeTrainerBuilder)\
                .set_simple_validator(discriminativeTraces.trace_manager, maxDrops, threads)
        else:
            deref(self.splitMergeTrainerBuilder).set_simple_validator(discriminativeTraces.trace_manager, maxDrops)
        return self

    cpdef PySplitMergeTrainerBuilder set_score_validator(
            self
            , PyCandidateScoreValidator validator
            , unsigned_int maxDrops=6
    ):
        deref(self.splitMergeTrainerBuilder).set_score_validator(validator.validator, maxDrops)
        return self

    cpdef PySplitMergeTrainerBuilder set_percent_merger(self, double percent=50.0, unsigned_int threads=0):
        if threads > 0:
            deref(self.splitMergeTrainerBuilder).set_percent_merger(percent, threads)
        else:
            deref(self.splitMergeTrainerBuilder).set_percent_merger(percent)
        return self

    cpdef PySplitMergeTrainerBuilder set_threshold_merger(self, double threshold, unsigned_int threads=0):
        if threads > 0:
            deref(self.splitMergeTrainerBuilder).set_threshold_merger(threshold, threads)
        else:
            deref(self.splitMergeTrainerBuilder).set_threshold_merger(threshold)
        return self

    cpdef PySplitMergeTrainerBuilder set_split_randomization(self, double percent=1.0, unsigned seed=0):
        deref(self.splitMergeTrainerBuilder).set_split_randomization(percent, seed)
        return self

    cpdef PySplitMergeTrainerBuilder set_smoothing_factor(self, double smoothingFactor=0.01):
        deref(self.splitMergeTrainerBuilder).set_smoothing_factor(smoothingFactor)
        return self

    cpdef PySplitMergeTrainer build(self):
        trainer = PySplitMergeTrainer()
        trainer.splitMergeTrainer = make_shared[SplitMergeTrainer[NONTERMINAL, size_t]](deref(self.splitMergeTrainerBuilder).build())
        trainer.emTrainer = (deref(self.splitMergeTrainerBuilder)).getEmTrainer()
        return trainer

cdef class PyLatentAnnotation:
    cdef set_latent_annotation(self, shared_ptr[LatentAnnotation] la):
        self.latentAnnotation = la


    def build_sm_grammar(self, grammar, PyGrammarInfo grammarInfo, rule_pruning, rule_smoothing=0.0):
        new_grammar = gl.LCFRS(grammar.start() + "[0]")
        for i in range(0, len(grammar.rule_index())):
            rule = grammar.rule_index(i)

            rule_dimensions = [deref(self.latentAnnotation).nonterminalSplits[nont]
                               for nont in deref(grammarInfo.grammarInfo).rule_to_nonterminals[i]]
            rule_dimensions_product = itertools.product(*[range(dim) for dim in rule_dimensions])

            lhs_dims = deref(self.latentAnnotation).nonterminalSplits[
                deref(grammarInfo.grammarInfo).rule_to_nonterminals[i][0]
            ]

            for la in rule_dimensions_product:
                index = list(la)

                if rule_smoothing > 0.0:
                    weight_av = sum([deref(self.latentAnnotation).get_weight(i, [lhs] + list(la)[1:])
                        for lhs in range(lhs_dims)]) / lhs_dims

                    weight = (1 - rule_smoothing) * deref(self.latentAnnotation).get_weight(i, index) \
                             + rule_smoothing * weight_av
                else:
                    weight = deref(self.latentAnnotation).get_weight(i, index)
                if weight > rule_pruning:
                    lhs_la = gl.LCFRS_lhs(rule.lhs().nont() + "[" + str(la[0]) + "]")
                    for arg in rule.lhs().args():
                        lhs_la.add_arg(arg)
                    nonts = [rhs_nont + "[" + str(la[1 + j]) + "]" for j, rhs_nont in enumerate(rule.rhs())]
                    new_grammar.add_rule(lhs_la, nonts, weight, rule.dcp())

        return new_grammar

    cpdef tuple serialize(self):
        cdef vector[size_t] splits = deref(self.latentAnnotation).nonterminalSplits
        cdef vector[double] rootWeights = deref(self.latentAnnotation).get_root_weights()
        cdef vector[vector[double]] ruleWeights = deref(self.latentAnnotation).get_rule_weights()
        return splits, rootWeights, ruleWeights

cpdef PyLatentAnnotation build_PyLatentAnnotation(vector[size_t] nonterminalSplits
                                                  , vector[double] rootWeights
                                                  , vector[vector[double]] ruleWeights
                                                  , PyGrammarInfo grammarInfo
                                                  , PyStorageManager storageManager):
    cdef PyLatentAnnotation latentAnnotation = PyLatentAnnotation()
    latentAnnotation.latentAnnotation = make_shared[LatentAnnotation](nonterminalSplits
                                                                      , rootWeights
                                                                      , ruleWeights
                                                                      , deref(grammarInfo.grammarInfo)
                                                                      , deref(storageManager.storageManager))
    return latentAnnotation

cpdef PyLatentAnnotation build_PyLatentAnnotation_initial(
        grammar
        , PyGrammarInfo grammarInfo
        , PyStorageManager storageManager):
    cdef vector[double] ruleWeights = []
    cdef size_t i
    for i in range(0, len(grammar.rule_index())):
        rule = grammar.rule_index(i)
        assert(isinstance(rule, gl.LCFRS_rule))
        ruleWeights.push_back(rule.weight())
    # output_helper(str(ruleWeights) + "\n")
    cdef PyLatentAnnotation latentAnnotation = PyLatentAnnotation()
    latentAnnotation.latentAnnotation \
        = make_shared[LatentAnnotation](ruleWeights
                                        , deref(grammarInfo.grammarInfo)
                                        , deref(storageManager.storageManager))
    return latentAnnotation

cdef class PySplitMergeTrainer:
    cdef map[string,TrainingMode] modes
    cdef shared_ptr[SplitMergeTrainer[NONTERMINAL, size_t]] splitMergeTrainer
    cdef shared_ptr[EMTrainerLA] emTrainer

    def __init__(self):
        modes_ = { "default": Default
                , "splitting": Splitting
                , "merging": Merging
                , "smoothing": Smoothing}
        self.modes = modes_

    cpdef PyLatentAnnotation split_merge_cycle (self, PyLatentAnnotation la):
        timeStart = time.time()
        cdef shared_ptr[LatentAnnotation] la_trained \
            = make_shared[LatentAnnotation](deref(self.splitMergeTrainer).split_merge_cycle(deref(la.latentAnnotation)))
        cdef PyLatentAnnotation pyLaTrained = PyLatentAnnotation()
        pyLaTrained.latentAnnotation = la_trained
        output_helper("Completed split/merge cycles in " + str(time.time() - timeStart) + " seconds")
        return pyLaTrained

    cpdef reset_random_seed(self, unsigned seed):
        deref(deref(self.splitMergeTrainer).splitter).reset_random_seed(seed)

    cpdef setEMepochs(self, unsigned epochs, mode="default"):
        deref(self.emTrainer).setEMepochs(epochs, self.modes.at(mode))

    cpdef setMaxDrops(self, unsigned maxDrops, mode="default"):
        (<EMTrainerLAValidation &> deref(self.emTrainer)).setMaxDrops(maxDrops, self.modes[mode])
