# ===========================#
# |   Connectivity graph    |#
# |  Good for doing simple  |#
# |     topology tricks     |#
# ===========================#
import networkx as nx


class MyG(nx.Graph):
    def __init__(self):
        super().__init__()
        self.Alive = True

    def __eq__(self, other):
        # This defines whether two MyG objects are "equal" to one another.
        if not self.Alive:
            return False
        if not other.Alive:
            return False
        return nx.is_isomorphic(self, other, node_match=nodematch)

    def __hash__(self):
        """The hash function is something we can use to discard two things that are obviously not equal.  Here we neglect the hash."""
        return 1

    def L(self):
        """Return a list of the sorted atom numbers in this graph."""
        return sorted(list(self.nodes()))

    def AStr(self):
        """Return a string of atoms, which serves as a rudimentary 'fingerprint' : '99,100,103,151' ."""
        return ",".join(["%i" % i for i in self.L()])

    def e(self):
        """Return an array of the elements.  For instance ['H' 'C' 'C' 'H']."""
        elems = nx.get_node_attributes(self, "e")
        return [elems[i] for i in self.L()]

    def ef(self):
        """Create an Empirical Formula"""
        Formula = list(self.e())
        return "".join(
            [
                ("%s%i" % (k, Formula.count(k)) if Formula.count(k) > 1 else "%s" % k)
                for k in sorted(set(Formula))
            ]
        )

    def x(self):
        """Get a list of the coordinates."""
        coors = nx.get_node_attributes(self, "x")
        return numpy.array([coors[i] for i in self.L()])
