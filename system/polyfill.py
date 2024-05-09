import itertools

# https://docs.python.org/3/library/itertools.html#itertools.batched
def batched(iterable, n):
    # batched('ABCDEFG', 3) -> ABC DEF G
    if n < 1:
        raise ValueError('n must be at least one')
    it = iter(iterable)
    while batch := tuple(itertools.islice(it, n)):
        yield batch
