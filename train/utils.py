def combine(terms, combinations, accum=[]):
    last = (len(terms) == 1)
    n = len(terms[0])
    for i in range(n):
        item = accum + [terms[0][i]]
        if last:
            combinations.append(item)
        else:
            combine(terms[1:], combinations, item)