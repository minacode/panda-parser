from experiment.resources import TRAINING, VALIDATION, TESTING, TESTING_INPUT
from experiment.hg_constituent_experiment import \
    ConstituentExperiment, InductionSettings, setup_corpus_resources
from grammar.induction.recursive_partitioning import the_recursive_partitioning_factory


class SuppertaggingExperiment(ConstituentExperiment):
    def __init__(self):
        ConstituentExperiment.__init__(self)


def main():
    induction_settings = InductionSettings()
    induction_settings.recursive_partitioning = \
        the_recursive_partitioning_factory().get_partitioning('fanout-2-left-to_right')
    induction_settings.naming_scheme = 'child'

    train, dev, test, test_input = setup_corpus_resources(split='NEGRA')

    experiment = SuppertaggingExperiment()
    experiment.resources[TRAINING] = train
    experiment.resources[VALIDATION] = dev
    experiment.resources[TESTING] = test
    experiment.resources[TESTING_INPUT] = test_input

    experiment.run_experiment()
