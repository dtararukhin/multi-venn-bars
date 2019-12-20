import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
import random


def count_edges(arr):
    """
    In an numpy.array of zeros and ones, find how many value changes are in all rows.
    """
    edges = 0
    ncols = arr.shape[1]
    for col, next_col in zip(range(ncols - 1), range(1, ncols)):
        edges += arr.shape[0] - arr[:, col].dot(arr[:, next_col]) - (1 - arr[:, col]).dot((1 - arr[:, next_col]))

    edges += arr.shape[0] - arr[:, 0].dot(arr[:, -1])

    return edges


def find_column_order(arr, seeds=10, noise_prob=0.1):
    """
    For a np.array of zeros and ones, find an order of columns such that there are value changes in rows.
    :param arr: np.array of zeros and ones
    :param seeds: number of random additions to the matrix of similarities, used to avoid local minima
    :param noise_prob: probability of 1 in noise matrices
    :return:
    """
    ncols = arr.shape[1]

    # similarities matrix: similarities[i,j] = number of matching values of columns i and j of arr.
    similarities = arr.T.dot(arr) + (1 - arr).T.dot(1 - arr)
    np.fill_diagonal(similarities, 0)

    best_sequence = None
    best_edges = 1e10

    for seed in range(seeds + 1):
        if seed == 0:
            noise = np.zeros_like(similarities)
        else:
            np.random.seed(seed)
            noise = (np.random.uniform(0, 1, size=similarities.shape) > 1 - noise_prob).astype(int)

        np.fill_diagonal(noise, 0)
        noisy_similarities = similarities.copy() + noise

        # placed_flags[i] == 1 <=> i-th column of arr has already been placed
        placed_flags = np.zeros(ncols, dtype=int)

        # find two most similar columns. Initialize two lists with tham.
        first_col_index, second_col_index = np.unravel_index(noisy_similarities.argmax(), noisy_similarities.shape)
        first_tail = [first_col_index]
        second_tail = [second_col_index]
        placed_flags[first_col_index] = 1
        placed_flags[second_col_index] = 1

        finished = False
        for _ in range(ncols):
            # if all columns have been placed, finish
            if placed_flags.sum() == ncols:
                finished = True
                break

            # find column most similar to the last element of any of the two lists, append it to the corresponding list
            similarities_to_first = noisy_similarities[:, first_tail[-1]] - placed_flags * 1000000
            similarities_to_second = noisy_similarities[:, second_tail[-1]] - placed_flags * 1000000
            first_argmax = similarities_to_first.argmax()
            second_argmax = similarities_to_second.argmax()
            first_max = similarities_to_first.max()
            second_max = similarities_to_second.max()
            if first_max >= second_max:
                idx = first_argmax
                first_tail.append(idx)
            else:
                idx = second_argmax
                second_tail.append(idx)

            placed_flags[idx] = 1

        if not finished:
            raise RuntimeError('find_column_order() went into an infinite loop')

        sequence = second_tail[::-1] + first_tail

        edges = count_edges(arr[:, sequence])

        if edges < best_edges:
            best_sequence = sequence
            best_edges = edges

    return best_sequence


def venn_bars_multi(sets, labels=None, order_chunks_by='smart', ignore_chunks_smaller_than=0):
    """
    Plot a venn-diagrammesque plot of any number of sets. The idea is to break the underlying set into chunks, where such
    as every one of given sets can be represented as a disjoint union of several such chunks; and thrn to show which set
    contains each of the chunks.
    :param sets: list of sets
    :param labels: labels for each set (optional)
    :param order_chunks_by:  how order set chunks (X axis)
        - 'smart': default, approximately minimizes "raggedness" of the plot using a clever algorithm
        - 'size': bigger chunks go first
        - 'occurence': chunks that are in more sets go first
        - 'random': randomly shuffle chunks

    """
    allowed_orderings = ['smart', 'size', 'occurence', 'random']
    if order_chunks_by not in allowed_orderings:
        raise ValueError(f'order_chunks_by should be one of {allowed_orderings}')

    if labels is None:
        labels = ['set_{}'.format(i) for i in range(1, len(sets) + 1)]

    if len(labels) != len(sets):
        raise ValueError('sets and labels have different len()')

    all_elements = set.union(*sets)

    # Each chunk is characterized by its signature. The signature is a unique subset of indices of our sets.
    # E.g. chunk with signature {1, 2, 5} is exactly the set of elements such that they belong to sets 1, 2, 5, and
    # don't belong to any of the other sets.
    # Build a dict with signatures as keys (as frozensets), and lists of elements as values,
    occurences = defaultdict(set)
    for element in all_elements:
        occurence_pattern = frozenset({i for i, set_ in enumerate(sets) if element in set_})
        occurences[occurence_pattern].add(element)

    chunks = [(occurence_pattern, elements_list) for occurence_pattern, elements_list in occurences.items()
              if len(elements_list) >= ignore_chunks_smaller_than] #list(occurences.items())

    ignored = [(occurence_pattern, elements_list) for occurence_pattern, elements_list in occurences.items()
               if len(elements_list) < ignore_chunks_smaller_than]



    # Sort the chunks according to order_chunks_by
    if order_chunks_by == 'size':
        chunks.sort(key=lambda tup: (len(tup[1]), len(tup[0])), reverse=True)

    elif order_chunks_by == 'occurence':
        chunks.sort(key=lambda tup: (len(tup[0]), len(tup[1])), reverse=True)

    elif order_chunks_by == 'smart':
        mat = np.zeros((len(sets), len(chunks)), dtype=int)
        for i, (set_indices, _) in enumerate(chunks):
            mat[list(set_indices), i] = 1

        permutation = find_column_order(mat)
        chunks = [chunks[i] for i in permutation]

    elif order_chunks_by == 'random':
        random.shuffle(chunks)

    # find all partition points for x axis
    partition_points = [0]
    current = 0
    for _, elements in chunks:
        current += len(elements)
        partition_points.append(current)

    # Build a mapping chunk -> segment on x axis
    chunk_to_segment = {chunk: (segment_begin, segment_end)
                       for (chunk, _), segment_begin, segment_end in zip(chunks, partition_points[:-1],
                                                                         partition_points[1:])}

    colors = plt.rcParams['axes.prop_cycle'].by_key()['color'] * 10

    for i, set_ in enumerate(sets):
        label = labels[i]
        for chunk, elements in chunks:
            if elements & set_:
                segment = chunk_to_segment[chunk]
                plt.barh(y=i, left=segment[0], width=segment[1] - segment[0], height=1, align='edge', color=colors[i],
                         alpha=0.5, label=label)
                label = None
        label_text = '{}:  {} items'.format(labels[i], len(set_))

        if ignored:
            ignored_chunks_count = 0
            ignored_elements_count = 0
            for chunk, elements in ignored:
                if elements & set_:
                    ignored_chunks_count += 1
                    ignored_elements_count += len(elements)
            label_text += '; {} items in {} chunks ignored'.format(ignored_elements_count, ignored_chunks_count)

        plt.annotate(xy=(len(all_elements) / 2, i + 0.5), s=label_text, ha='center', va='center')

    for i, (point, next_point) in enumerate(zip(partition_points[:-1], partition_points[1:])):
        if i % 2:
            y = -0.90
            va = 'bottom'
        else:
            y = -0.1
            va = 'top'
        plt.annotate(xy=(0.5 * (point + next_point), y), s=str(next_point - point), ha='center', va=va)

    plt.xticks(partition_points, [])
    plt.ylim(-1, len(sets) + 1)
    x_margin = 0.03 * len(all_elements)
    plt.xlim(-x_margin, len(all_elements) + x_margin)
    plt.yticks(range(len(sets) + 1), [])
    plt.grid(alpha=0.5)
