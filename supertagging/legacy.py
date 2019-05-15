def __rec_monadic_on_range(elements, choice_function, size):
    if size <= 1:
        return elements, []
    chosen_one = choice_function(elements)
    new_set = elements - {chosen_one}
    return elements, [
        __rec_monadic_on_range(
            elements=new_set,
            choice_function=choice_function,
            size=size - 1
        )
    ]


def monadic_on_range(n, choice_function):
    elements = set(range(n))
    return __rec_monadic_on_range(
            elements=elements,
            choice_function=choice_function,
            size=n
        )


def left_to_right(tree):
    return monadic_on_range(
        n=tree.n_yield_nodes(),
        choice_function=min
    )


def right_to_left(tree):
    return monadic_on_range(
        n=tree.n_yield_nodes(),
        choice_function=max
    )


def unsorted_even_k_split(elements, k):
    split = [set() for _ in range(k)]
    for i, element in enumerate(elements):
        split[i % k].add(element)
    return split


def srp_to_trp_old(tree, positions):
    tree_positions = {
        tree.index_node(position+1)
        for position
        in positions
    }
    common_parent = min_common_parent(
        tree=tree,
        positions=tree_positions,
    )
    subtree_positions = set(tree.descendants(common_parent))
    subtree_positions.add(common_parent)
    alignment_positions = set(tree.id_yield())
    if (subtree_positions & alignment_positions) - tree_positions:
        return_positions = tree_positions
    else:
        return_positions = subtree_positions
    return return_positions


'''
    nonterminal = ';'.join(
        (   
            ' '.join(
                (
                    argument_position
                    # if isinstance(argument_position, str)
                    # else str(argument_position)
                    for argument_position in argument
                )
            )
            for argument in arguments
        )
    )
    print(nonterminal)

    nonterminal_positions = positions.copy()
    for child_positions in children_positions:
        nonterminal_positions.difference_update(child_positions)
    print(nonterminal_positions)
    nonterminal = '_'.join(
        (
            str(
                tree.node_token(
                    tree.index_node(position)
                )
            )
            for position
            in nonterminal_positions
        )
    )
    print(nonterminal)
    
    

    symbols = []
    for part in span_child_substitution_sequence(span=span):
        for position in part:
            symbols.append(
                nonterminal_labeling(
                    tree=tree,
                    id=tree.index_node(position+1)
                )
            )
    symbol = '_'.join(symbols)
    
    
    children_tops = [
        top_max(
            tree=tree,
            id_set=srp_to_trp(
                tree=tree,
                positions=child_positions
            )
        )
        for child_positions
        in children_positions
    ]
    '''
