import numpy as np


_INF = 0xFFFFFFFF


def indexOfClosestElementInList(e, l):

    min_distance = _INF
    index = -1

    for i in range(len(l)):

        d = np.linalg.norm(l[i] - e)

        if d < min_distance:
            min_distance = d
            index = i

    return index


def countVotes(l):

    dictionary = {}

    for element in l:

        element = tuple(element)

        if element in dictionary:
            dictionary[element] += 1
        else:
            dictionary[element] = 1

    result = None
    max_votes = -1

    for element in dictionary.keys():

        if dictionary[element] > max_votes:
            result = element
            max_votes = dictionary[element]

    return result