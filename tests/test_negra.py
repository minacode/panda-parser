import unittest
from collections import defaultdict

from constituent.induction import direct_extract_lcfrs, fringe_extract_lcfrs, BasicNonterminalLabeling
from corpora.negra_parse import NEGRA_DIRECTORY, NEGRA_NONPROJECTIVE, num_to_name, sentence_names_to_hybridtrees
from dependency.induction import induce_grammar as dependency_induce_grammar
from dependency.labeling import the_labeling_factory
from grammar.induction.terminal_labeling import the_terminal_labeling_factory, FormTerminalsPOS
from supertagging.induction import \
    induce, PartitionBuilder, pretty_print_partition, sorted_even_k_split, choose_middle, spans_split, flat_wide_split, \
    monadic_split, random_split, choose_random, non_lexicalized_partition, span_child_substitution_sequence, \
    srp_to_trp, nonterminal_labeling_strict, rule_classes_created_terminals, sorted_even_2_split, sorted_even_3_split, \
    choose_min, choose_max
from tests.test_induction import hybrid_tree_1
from hybridtree.general_hybrid_tree import HybridTree

N_NEGRA_SENTENCES = 20602
NEGRA_PATH = NEGRA_DIRECTORY + NEGRA_NONPROJECTIVE


class MyTestCases(unittest.TestCase):
    @staticmethod
    def get_trees_for_single_sentence(id=37):
        return sentence_names_to_hybridtrees(
            names=[num_to_name(id)],
            path=NEGRA_PATH
        )

    def get_shortest_tree(self):
        trees = sentence_names_to_hybridtrees(
            names=[
                num_to_name(num)
                for num in range(N_NEGRA_SENTENCES)
            ],
            path=NEGRA_PATH
        )
        return min(trees, key=lambda tree: tree.n_yield_nodes())

    def get_single_tree(self, id=37, n=0):
        trees = self.get_trees_for_single_sentence(id=id)
        return trees[n]

    def test_shortest_tree(self):
        print(f'shortest tree: {self.get_shortest_tree()}')

    def test_load_tree(self):
        trees = self.get_trees_for_single_sentence()
        for tree in trees:
            print(str(tree))

    def get_whole_corpus(self, n=N_NEGRA_SENTENCES):
        return sentence_names_to_hybridtrees(
            names=[num_to_name(num) for num in range(n+1)],
            path=NEGRA_PATH
        )

    def test_id_yield(self):
        tree = self.get_single_tree()
        for token in tree.full_yield():
            print(token)

    def test_string_partitioning(self):
        tree = self.get_single_tree()
        n = tree.n_yield_nodes()
        print(f'string partitioning for n={n}')
        self.__test_partitioning(n)

    def test_tree_partitioning(self):
        tree = self.get_single_tree()
        n = tree.n_nodes()
        print(f'tree partitioning for n={n}')
        self.__test_partitioning(n)

    def test_direct_induction(self):
        tree = self.get_single_tree()
        terminal_labeling = the_terminal_labeling_factory().get_strategy('pos')
        lcfrs = direct_extract_lcfrs(
            tree=tree,
            term_labeling=terminal_labeling
        )
        print(lcfrs)

    def _test_constituent_induction(self):
        tree = self.get_single_tree()
        recursive_partition = PartitionBuilder(
            choice_function=min,
            split_function=monadic_split
        )(
            tree=tree
        )
        lcfrs = fringe_extract_lcfrs(
            tree=tree,
            fringes=recursive_partition
        )
        print(lcfrs)

    # fails
    def _test_dependency_induction(self):
        tree = self.get_single_tree()
        grammar = dependency_induce_grammar(
            trees=[tree],
            nont_labelling=the_labeling_factory().create_simple_labeling_strategy('empty','pos'),
            term_labelling=the_terminal_labeling_factory().get_strategy('form').token_label,
            recursive_partitioning=[
                lambda tree: PartitionBuilder(
                    choice_function=min,
                    split_function=monadic_split
                )(tree)
            ]
        )
        print(grammar)

    def test_own_induction(self):
        # trees = [self.get_single_tree()]
        # trees = [hybrid_tree_1()]
        trees = self.get_whole_corpus()
        print(trees[0])
        partition_builders = []
        for choice_function in [
            choose_min,
            choose_max,
            choose_middle,
            # choose_random
        ]:
            for split_function in [
                monadic_split,
                spans_split,
                # flat_wide_split,
                sorted_even_2_split,
                # sorted_even_3_split,
                # random_split
            ]:
                partition_builders.append(
                    PartitionBuilder(
                        choice_function=choice_function,
                        split_function=split_function
                    )
                )
        terminal_labelings = [
            the_terminal_labeling_factory().get_strategy('form'),
            the_terminal_labeling_factory().get_strategy('pos')
        ]
        for partition_builder in partition_builders:
            for terminal_labeling in terminal_labelings:
                print(
                    'make induction with:\n'
                    f'partition builder:  {partition_builder}\n'
                    f'terminal_labelling: {terminal_labeling}'
                )
                nonterminal_counts = {}
                grammar = induce(
                    trees=trees,
                    partition_builder=partition_builder,
                    terminal_labeling=terminal_labeling,
                    nonterminal_counts=nonterminal_counts
                )
                rules = grammar.rules()
                n_rules = len(rules)
                for i, rule in enumerate(rules):
                    # if not i % 10_000:
                    #     print(f'update rule {i} of {n_rules}')
                    weight = rule.weight()
                    nonterminal = rule.lhs().nont()
                    assert nonterminal in nonterminal_counts
                    weight /= nonterminal_counts[nonterminal]
                    rule.set_weight(weight)
                # for rule in sorted(rules, key=lambda r: r.weight(), reverse=True)[100:]:
                #     print(rule)
                # print(f'grammar:\n{grammar}')
                classes = rule_classes_created_terminals(grammar)
                class_sizes = [
                    len(classes[key])
                    for key
                    in classes
                ]
                class_sizes.sort(reverse=True)
                print(
                    f'min: {min(class_sizes)})\n'
                    f'max: {max(class_sizes)}\n'
                    f'avg: {sum(class_sizes) / len(class_sizes)}\n'
                )

    # fails. bug on my side or are rules with multiple terminals are not supported?
    def test_non_lexicalized_indcution(self):
        trees = self.get_whole_corpus(n=1)
        nonterminal_counts = {}
        grammar = induce(
            trees=trees,
            partition_builder=non_lexicalized_partition,
            terminal_labeling=the_terminal_labeling_factory().get_strategy('form'),
            nonterminal_counts=nonterminal_counts
        )
        print(grammar)

    def test_split_strategies(self):
        partition = PartitionBuilder(
            choice_function=min,
            split_function=spans_split
        ).string_partition(
            tree=self.get_single_tree()
        )
        print(pretty_print_partition(partition))

    def test_monadic_split(self):
        self.assertEqual(
            monadic_split({1,2,3,4,5}),
            [{1,2,3,4,5}]
        )

    def test_choose_middle(self):
        self.assertEqual(
            choose_middle({1,2,3,4,5}),
            3
        )
        self.assertEqual(
            choose_middle({1,2,3,4}),
            2
        )
        self.assertNotEqual(
            choose_middle({1,2,3,4}),
            3
        )

    def test_flat_wide_span(self):
        self.assertEqual(
            flat_wide_split({1,2,3,4,5,6}),
            [{1}, {2}, {3}, {4}, {5}, {6}]
        )

    def test_span_split(self):
        self.assertEqual(
            spans_split({1,2,3,5,6,8,9,11}),
            [{1,2,3}, {5,6}, {8,9}, {11}]
        )

    def test_sorted_even_k_split(self):
        self.assertEqual(
            sorted_even_k_split({1,2,3,4,5,6,7,8}, 2),
            [{1,2,3,4}, {5,6,7,8}]
        )
        self.assertEqual(
            sorted_even_k_split({1,2,3,4,5,6}, 3),
            [{1,2}, {3,4}, {5,6}]
        )
        self.assertEqual(
            sorted_even_k_split({1,2,3,4,5,6,7}, 2),
            [{1,2,3}, {4,5,6,7}]
        )
        self.assertEqual(
            sorted_even_k_split({1,2,3,4,5,6,7}, 4),
            [{1}, {2}, {3}, {4,5,6,7}]
        )

    def test_recursive_partition(self):
        self.assertEqual(
            PartitionBuilder(
                choice_function=min,
                split_function=spans_split
            ).string_partition(
                tree=HybridTree()
            ),
            ({}, [])
        )

    def test_non_lexicalized_partition(self):
        self.assertEqual(
            non_lexicalized_partition({1,2,3,4,5,6,7}),
            ({1,2}, [
                ({3,4}, [
                    ({5,6}, [
                        ({7}, [])
                    ])
                ])
            ])
        )

    def test_span_child_substitution_sequence(self):
        self.assertEqual(
            span_child_substitution_sequence(
                span=(1, 10),
                children_spans=[[(2, 4), (6, 8)], [(5, 5), (10, 10)]]
            ),
            [
                ((1, 1), None),
                ((2, 4), (0, 0)),
                ((5, 5), (1, 0)),
                ((6, 8), (0, 1)),
                ((9, 9), None),
                ((10, 10), (1, 1))
            ]
        )

    # not sure if correct
    def test_srp_to_trp(self):
        # tree = hybrid_tree_1()
        tree = self.get_single_tree()
        srp = PartitionBuilder(
            choice_function=min,
            split_function=monadic_split
        )(tree=tree)
        trp = srp_to_trp(
            tree=tree,
            recursive_string_partition=srp
        )
        print(tree)
        print(pretty_print_partition(srp))
        print(pretty_print_partition(trp))

if __name__ == '__main__':
    unittest.main()
