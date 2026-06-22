import networkx as nx

class TreeIterator:
    def __init__(self, vertices):
        self.vertices = vertices
        self.l = None
        self.currentLevelSequence = None
        self.firstTime = True
        self.p = None
        self.q = None
        self.h1 = None
        self.h2 = None
        self.c = None
        self.r = None

    def __del__(self):
        self.l = None
        self.currentLevelSequence = None

    def __str__(self):
        return f"Iterator over all trees with {self.vertices} vertices"

    def __iter__(self):
        return self

    def __next__(self):
        if not self.firstTime and not self.q:
            raise StopIteration

        if self.firstTime:
            self.firstTime = False
            if self.vertices:
                self.l = [0] * self.vertices
                self.currentLevelSequence = [0] * self.vertices
                self.generateFirstLevelSequence()
            else:
                self.q = 0
        else:
            self.generateNextLevelSequence()

        G = nx.Graph()
        G.add_nodes_from(range(self.vertices))

        for i in range(2, self.vertices + 1):
            vertex1 = i - 1
            vertex2 = self.currentLevelSequence[i - 1] - 1
            G.add_edge(vertex1, vertex2)

        return G

    def generateFirstLevelSequence(self):
        k = (self.vertices // 2) + 1

        if self.vertices == 4:
            self.p = 3
        else:
            self.p = self.vertices
        self.q = self.vertices - 1
        self.h1 = k
        self.h2 = self.vertices
        self.c = float('inf') if self.vertices % 2 else self.vertices + 1
        self.r = k

        for i in range(1, k + 1):
            self.l[i - 1] = i
        for i in range(k + 1, self.vertices + 1):
            self.l[i - 1] = i - k + 1
        for i in range(self.vertices):
            self.currentLevelSequence[i] = i
        if self.vertices > 2:
            self.currentLevelSequence[k] = 1
        if self.vertices <= 3:
            self.q = 0

    def generateNextLevelSequence(self):
        fixit = False
        needr = False
        needc = False
        needh2 = False

        n = self.vertices
        p = self.p
        q = self.q
        h1 = self.h1
        h2 = self.h2
        c = self.c
        r = self.r
        l = self.l
        w = self.currentLevelSequence

        if c == n + 1 or p == h2 and (l[h1 - 1] == l[h2 - 1] + 1 and n - h2 > r - h1 or l[h1 - 1] == l[h2 - 1] and n - h2 + 1 < r - h1):
            if l[r - 1] > 3:
                p = r
                q = w[r - 1]
                if h1 == r:
                    h1 -= 1
                fixit = True
            else:
                p = r
                r -= 1
                q = 2

        if p <= h1:
            h1 = p - 1
        if p <= r:
            needr = True
        elif p <= h2:
            needh2 = True
        elif l[h2 - 1] == l[h1 - 1] - 1 and n - h2 == r - h1:
            if p <= c:
                needc = True
        else:
            c = float('inf')

        oldp = p
        delta = q - p
        oldlq = l[q - 1]
        oldwq = w[q - 1]
        p = float('inf')

        for i in range(oldp, n + 1):
            l[i - 1] = l[i - 1 + delta]
            if l[i - 1] == 2:
                w[i - 1] = 1
            else:
                p = i
                if l[i - 1] == oldlq:
                    q = oldwq
                else:
                    q = w[i - 1 + delta] - delta
                w[i - 1] = q
            if needr and l[i - 1] == 2:
                needr = False
                needh2 = True
                r = i - 1
            if needh2 and l[i - 1] <= l[i - 2] and i > r + 1:
                needh2 = False
                h2 = i - 1
                if l[h2 - 1] == l[h1 - 1] - 1 and n - h2 == r - h1:
                    needc = True
                else:
                    c = float('inf')
            if needc:
                if l[i - 1] != l[h1 - h2 + i - 1] - 1:
                    needc = False
                    c = i
                else:
                    c = i + 1

        if fixit:
            r = n - h1 + 1
            for i in range(r + 1, n + 1):
                l[i - 1] = i - r + 1
                w[i - 1] = i - 1
            w[r] = 1
            h2 = n
            p = n
            q = p - 1
            c = float('inf')
        else:
            if p == float('inf'):
                if l[oldp - 2] != 2:
                    p = oldp - 1
                else:
                    p = oldp - 2
                q = w[p - 1]
            if needh2:
                h2 = n
                if l[h2 - 1] == l[h1 - 1] - 1 and h1 == r:
                    c = n + 1
                else:
                    c = float('inf')

        self.p = p
        self.q = q
        self.h1 = h1
        self.h2 = h2
        self.c = c
        self.r = r
        self.l = l
        self.currentLevelSequence = w


