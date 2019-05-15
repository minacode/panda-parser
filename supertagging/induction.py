from constituent.induction import START, span_to_arg, BasicNonterminalLabeling
from grammar.induction.decomposition import join_spans
from dependency.top_bottom_max import top_max, bottom_max
from grammar.lcfrs import LCFRS, LCFRS_lhs
from grammar.dcp import *
from random import shuffle, choice, random
from collections import defaultdict


# partition core
class PartitionBuilder:
    def __init__(self, choice_function, split_function):
        self.__choice_function = choice_function
        self.__split_function = split_function

    def string_partition(self, tree):
        n = tree.n_yield_nodes()
        if not n:
            raise ValueError(f'Tree contains zero nodes.')
        return self.build_partition(elements=set(range(n)))

    def build_partition(self, elements):
        """
        :param elements: set
        :param choice_function: set(a) -> a
        :param split_function: set(a) -> list(set(a)), s.t.: list is sorted asc. and sets in list are disjoint
        :return:
        """
        chosen_ones = self.__choice_function(elements)
        remaining_elements = elements - chosen_ones
        split = self.__split_function(remaining_elements)
        return elements, [
            self.build_partition(elements=part)
            for part
            in split
            if part
        ]

    def __call__(self, tree):
        return self.string_partition(tree=tree)

    def __repr__(self):
        return f'{self.__choice_function.__name__}, {self.__split_function.__name__}'


# split functions
def monadic_split(elements):
    return [elements]


def sorted_even_k_split(elements, k):
    elements = sorted(elements)
    div, mod = divmod(len(elements), k)
    split = []
    for i in range(k):
        part = set(elements[i*div:(i+1)*div])
        split.append(part)
    if mod:
        split[-1].update(elements[-mod:])
    return split


def sorted_even_2_split(elements):
    return sorted_even_k_split(elements=elements, k=2)


def sorted_even_3_split(elements):
    return sorted_even_k_split(elements=elements, k=3)


def flat_wide_split(elements):
    return [{element} for element in elements]


def spans_split(elements):
    elements = sorted(elements)
    split = []
    part = []
    i = 0
    for element in elements:
        if not part or part[-1] == element - 1:
            part.append(element)
        else:
            split.append(set(part))
            part = [element]
    split.append(set(part))
    return split


def random_split(elements, threshold=0.2):
    elements = list(elements)
    split = []
    part = []
    for element in elements:
        part.append(element)
        if random() < threshold:
            split.append(set(part))
            part = []
    split.append(set(part))
    return split


# choice functions (besides min, max)
def choose_min(elements):
    return {min(elements)}


def choose_max(elements):
    return {max(elements)}


def choose_middle(elements):
    elements = sorted(elements)
    length = len(elements)
    div = (length - 1) // 2
    return elements[div]


def choose_random(elements):
    return choice(list(elements))


def old_choice(elements):
    return elements if len(elements) == 1 else {}


def min_common_parent(tree, positions, id=None):
    if id is None:
        id = tree.virtual_root
    for child_id in tree.children(id):
        subtree_positions = set(tree.descendants(child_id))
        subtree_positions.add(child_id)
        if not positions - subtree_positions:
            return min_common_parent(
                tree=tree,
                positions=positions,
                id=child_id
            )
    return id


# helper
def srp_to_trp(tree, positions):
    tree_positions = [
        tree.index_node(position+1)
        for position
        in positions
    ]
    for position in tree_positions:
        parent = tree.parent(position)
        if parent is not None:
            subtree_positions = tree.descendants(parent)
            # print(f'p: {parent}')
            # print(f's: {subtree_positions}')
            # print(f't: {tree_positions}')
            if parent not in tree_positions and all((
                s_pos in tree_positions
                for s_pos
                in subtree_positions
            )):
                tree_positions.append(parent)
    return tree_positions


def span_contains(span1, span2):
    low1, high1 = span1
    low2, high2 = span2
    return low1 <= low2 and high1 >= high2


def non_lexicalized_partition(tree):
    n = tree.n_yield_nodes()
    return __rec_non_lexicalized_partition(set(range(n)))


def __rec_non_lexicalized_partition(elements):
    if len(elements) <= 2:
        return elements, []
    return {elements.pop() for _ in range(2)}, [
        __rec_non_lexicalized_partition(elements)
    ]


def nonterminal_labeling_strict(tree, spans, children_spans, child_labelling_strategy=''):
    nonterminal_labeling = BasicNonterminalLabeling().label_nont
    spans_labellings = []
    for span in spans:
        span_labelling = []
        sequence = span_child_substitution_sequence(
            span=span,
            children_spans=children_spans
        )
        for (low, high), child in sequence:
            if child is None:
                for index in range(low, high + 1):
                    span_labelling.append(
                        nonterminal_labeling(
                            tree=tree,
                            id=tree.index_node(index=index + 1)
                        )
                    )
            else:
                if child_labelling_strategy == 'indexed':
                    i, j = child
                    span_labelling.append(f'<{i},{j}>')
                else:
                    span_labelling.append('$')
        spans_labellings.append(
            '_'.join(span_labelling)
        )
    symbol = ';'.join(spans_labellings)

    fanout = len(spans)
    # symbol = choice(['α', 'β', 'γ'])
    return f'{symbol}/{fanout}'


def pretty_print_partition(partition, level=0):
    elements, children = partition
    lines = ' '*level + str(elements) + '\n'
    for child in children:
        lines += pretty_print_partition(partition=child, level=level+1)
    return lines


def span_child_substitution_sequence(span, children_spans):
    low, high = span
    k = low
    sequence = []
    while k <= high:
        matched = False
        for i, child_spans in enumerate(children_spans):
            for j, child_span in enumerate(child_spans):
                child_low, child_high = child_span
                if k == child_low:
                    sequence.append(
                        ((child_low, child_high), (i, j))
                    )
                    k = child_high + 1
                    matched = True
        if not matched:
            if not sequence or sequence[-1][1] is not None:
                sequence.append(
                    ((k, k), None)
                )
            else:
                span, child = sequence[-1]
                span_low, _ = span
                sequence[-1] = ((span_low, k), None)
            k += 1
    return sequence


def create_dcp_rhs(tree, root, present_tree_positions, string_positions, id_to_pos, terminal_labelling, children_tops):
    # create var if at root a subtree begins
    if root not in present_tree_positions:
        for i, child_tops in enumerate(children_tops):
            for j, position in enumerate(child_tops):
                if position == root:
                    return DCP_var(i=i, j=j)
        raise(Exception(f'{root} not in children tops {children_tops}'))

    # create head for DCP_term
    # distinguish between string terminals and non-string terminals
    if root in string_positions:
        head = DCP_index(
            i=id_to_pos[root]
        )
    else:
        token = tree.node_token(root)
        # TODO implement a more general strategy here
        if token.type() == 'CONSTITUENT-TERMINAL':
            label = terminal_labelling.token_label(token)
        elif token.type() == 'CONSTITUENT-CATEGORY':
            label = token.category()
        else:
            raise(ValueError(
                f'Unsupported token type: {token.type()}'
            ))
        head = DCP_string(
            string=label
        )

    children = tree.children(root)
    if not children:
        return head
    arguments = [
        create_dcp_rhs(
            tree=tree,
            root=child,
            present_tree_positions=present_tree_positions,
            string_positions=string_positions,
            id_to_pos=id_to_pos,
            terminal_labelling=terminal_labelling,
            children_tops=children_tops
        )
        for child
        in children
    ]
    return DCP_term(
        head=head,
        arg=arguments
    )


# rule induction
def induce(trees, partition_builder, terminal_labeling, nonterminal_counts, start=START):
    grammar = LCFRS(start=start)
    n_trees = len(trees)
    for i, tree in enumerate(trees):
        # if not i % 1000:
        #     print(f'starting induction on tree {i} out of {n_trees}')
        # tree contains nodes
        if tree.n_yield_nodes():
            partition = partition_builder(tree=tree)
            # print(pretty_print_partition(partition=partition))
            __rec_induce(
                tree=tree,
                grammar=grammar,
                string_partition=partition,
                terminal_labeling=terminal_labeling,
                nonterminal_counts=nonterminal_counts
            )
    return grammar


def __rec_induce(tree, grammar, string_partition, terminal_labeling, nonterminal_counts):
    positions, children_partitions = string_partition
    tree_positions = srp_to_trp(
        tree=tree,
        positions=positions
    )
    top = sum(
        top_max(
            tree=tree,
            id_set=tree_positions
        ),
        []
    )

    children_positions = []
    children_nonterminals = []
    for child_partition in children_partitions:
        child_positions = child_partition[0]
        children_positions.append(child_positions)
        children_nonterminals.append(__rec_induce(
            tree=tree,
            grammar=grammar,
            string_partition=child_partition,
            terminal_labeling=terminal_labeling,
            nonterminal_counts=nonterminal_counts
        ))

    children_spans = []
    children_tops = []
    children_tree_positions = []
    for child_positions in children_positions:
        children_spans.append(join_spans(child_positions))
        children_tops.append(
            sum(
                top_max(
                    tree=tree,
                    id_set=srp_to_trp(
                        tree=tree,
                        positions=child_positions
                    )
                ),
                []
            )
        )
        children_tree_positions.append(
            srp_to_trp(
                tree=tree,
                positions=child_positions
            )
        )

    spans = join_spans(positions)
    arguments = []
    term_to_pos = {}
    for span in spans:
        arguments.append(span_to_arg(
            span=span,
            children=children_spans,
            tree=tree,
            term_to_pos=term_to_pos,
            term_labeling=terminal_labeling
        ))

    # TODO idea: since we now have the top-set,
    #  we can name nonterminals based on them
    #  - also there might be a mistake: Ausgerechnet creates S(…)
    nonterminal = nonterminal_labeling_strict(
        tree=tree,
        spans=spans,
        children_spans=children_spans
    )
    lhs = LCFRS_lhs(nonterminal)
    for argument in arguments:
        lhs.add_arg(argument)

    present_tree_positions = tree_positions[:]
    for child_tree_positions in children_tree_positions:
        for position in child_tree_positions:
            present_tree_positions.remove(position)

    # print(f'pos: {positions}\ntos: {tree_positions}\ntop: {top}')
    # print(f'pst: {present_tree_positions}')
    # print()

    id_to_pos = {
        tree.index_node(index=index+1): term_to_pos[index]
        for index
        in term_to_pos
    }
    dcp_rhs = [
        create_dcp_rhs(
            tree=tree,
            root=t_id,
            present_tree_positions=present_tree_positions,
            string_positions=tree.id_yield(),
            id_to_pos=id_to_pos,
            terminal_labelling=terminal_labeling,
            children_tops=children_tops
        )
        for t_id
        in top
    ]
    dcp = [
        DCP_rule(
            lhs=DCP_var(-1, 0),
            rhs=dcp_rhs
        )
    ]

    grammar.add_rule(lhs, children_nonterminals, 1, dcp)
    if nonterminal in nonterminal_counts:
        nonterminal_counts[nonterminal] += 1
    else:
        nonterminal_counts[nonterminal] = 1
    return nonterminal


def rule_classes_created_terminals(lcfrs):
    classes = {}
    for rule in lcfrs.rules():
        weight = rule.weight()
        lhs = rule.lhs()
        arguments = sum(lhs.args(), [])
        terminals = frozenset((
            argument
            for argument
            in arguments
            if isinstance(argument, str)
        ))
        if terminals not in classes:
            classes[terminals] = [(rule, weight)]
        else:
            # insert rule, sorted ascending by weight
            for i, key in enumerate(classes[terminals]):
                if weight > classes[terminals][i][1]:
                    classes[terminals].insert(i, (rule, weight))
                    break
    return classes


