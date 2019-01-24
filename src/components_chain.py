from math import sqrt


class Letter(object):
    def __init__(self, comp, i):
        self.comp = comp
        self.ind = i
        self.dist = -1


class Chain(object):
    def __init__(self):
        self.letters = []
        self.hasMerged = False
        self.direction = None
        self.left_bottom = (-1, -1)
        self.left_top = (-1, -1)
        self.right_top = (-1, -1)
        self.right_bottom = (-1, -1)
        self.height = -1
        self.candidate_count = -1

    def merge(chainA, chainB):
        chainA_ind = set([let.ind for let in chainA.letters])

        for l in chainB.letters:
            if l.ind not in chainA_ind:
                chainA.letters.append(l)

        chainA.candidate_count = len(chainA.letters)

        direction = 0
        for l in chainA.letters:
            direction += l.comp.orientation

        direction /= len(chainA.letters)
        chainA.direction = direction
        chainA.set_bounding_box()

    def candidate_len(self):
        return len(self.letters)

    def orientation_similarity(chainA, chainB):
        incl_angle = abs(chainA.direction - chainB.direction)
        return incl_angle <= 90

    def location_similarity(chainA, chainB):
        chainA_ind = [let.ind for let in chainA.letters]
        chainB_ind = [let.ind for let in chainB.letters]
        if len(list(set(chainA_ind) & set(chainB_ind))):
            return True
        else:
            return False

    def is_similar(chainA, chainB):
        return chainA.orientation_similarity(chainB) and chainA.location_similarity(chainB)

    def set_bounding_box(self):
        compsX = sorted([l.comp for l in self.letters], key=get_min_x)
        compsY = sorted([l.comp for l in self.letters], key=get_min_y)

        self.left_bottom = (compsX[0].minX, compsY[0].minY)
        self.left_top = (compsX[0].minX, compsY[0].maxY)
        self.right_top = (compsX[len(compsX) - 1].maxX, compsY[len(compsY) - 1].maxY)
        self.right_bottom = (compsX[len(compsX) - 1].maxX, compsY[len(compsY) - 1].minY)


class Pair(object):
    def __init__(self, letterA, letterB):
        self.letterA = letterA
        self.letterB = letterB

    def is_valid(self):
        self.letterA.comp.characteristic_scale = self.letterA.comp.minor_axis + self.letterA.comp.major_axis
        self.letterB.comp.characteristic_scale = self.letterB.comp.minor_axis + self.letterB.comp.major_axis
        if self.letterB.comp.characteristic_scale and self.letterB.comp.mean:
            return 0.5 < self.letterA.comp.mean / self.letterB.comp.mean < 2.0 and \
                   0.4 < self.letterA.comp.characteristic_scale / self.letterB.comp.characteristic_scale < 2.5 and \
                   dist(self.letterA.comp.center_x, self.letterA.comp.center_y, self.letterB.comp.center_x,
                        self.letterB.comp.center_y) < 2 * max(
                self.letterA.comp.minor_axis, self.letterB.comp.minor_axis)
        else:
            return False

    def set_to_chain(self):
        chain = Chain()
        chain.letters.append(self.letterA)
        chain.letters.append(self.letterB)
        chain.direction = (self.letterA.comp.orientation + self.letterB.comp.orientation) / 2
        chain.set_bounding_box()
        return chain


'''min x of component'''
def get_min_x(comp):
    return comp.minX


'''min y of component'''
def get_min_y(comp):
    return comp.minY


'''return distance between two points'''
def dist(x1, y1, x2, y2):
    return sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


'''unite components to pairs'''
def find_pairs(comp):
    components = comp
    n = len(components.components)
    chains_of_pairs = []

    for i in range(n):
        comp = Letter(components.components[i], i)
        if comp.comp.pred:
            for j in range(i + 1, n):
                neigh = Letter(components.components[j], j)
                if neigh.comp.pred:
                    pair = Pair(comp, neigh)
                    if pair.is_valid():
                        chains_of_pairs.append(pair.set_to_chain())

    return chains_of_pairs


'''merge chains to get text lines'''
def find_lines(chains):
    lines = []
    for chain in chains:
        merge = False
        chain.set_bounding_box()
        for i, line in enumerate(lines):
            if line.is_similar(chain):
                merge = True
                lines[i].merge(chain)
        if not merge:
            lines.append(chain)
    return lines


def merge_chains(chains):
    lines = find_lines(chains)
    length = len(lines)

    while True:
        lines = find_lines(lines)

        if length == len(lines):
            break
        length = len(lines)

    return lines
