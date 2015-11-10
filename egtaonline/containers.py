"""Useful containers"""
import heapq


class priorityqueue(object):
    """Priority queue with more sensible interface"""
    def __init__(self, init_elems=()):
        self._elems = list(init_elems)
        heapq.heapify(self._elems)

    def append(self, elem):
        """Add an element to the priority queue"""
        heapq.heappush(self._elems, elem)

    def pop(self):
        """Removes the highest priority element from the queue"""
        return heapq.heappop(self._elems)

    def extend(self, iterable):
        """Extends the priority queue with the supplied iterable"""
        for elem in iterable:
            self.append(elem)

    def drain(self):
        """consuming iterable of the queue in sorted order"""
        while self:
            yield self.pop()

    def __len__(self):
        return len(self._elems)

    def __bool__(self):
        return bool(self._elems)

    def __iter__(self):
        return iter(self._elems)

    def __repr__(self):
        return "<" + repr(self._elems)[1:-1] + ">"


class setset(object):
    """Set of sets

    This container uses an inverted index of element to sets that contain that
    element to do a faster lookup at the cost of some space."""

    # This class uses an inverted index to have semi fast lookup
    def __init__(self, iterable=()):
        self._inverted_index = {}
        for added_subgame in iterable:
            self.add(added_subgame)

    def add(self, added_subgame):
        """Adds a subgame to the set

        Returns True if the set was modified"""
        # If dominated, don't add
        if added_subgame in self:
            return False

        # Otherwise, add and remove all subgames
        for elem in added_subgame:
            bucket = self._inverted_index.setdefault(elem, set())
            # Copy bucket to avoid concurrent modification
            for current_subgame in list(bucket):
                if added_subgame > current_subgame:
                    # Game in bucket is a subgame
                    bucket.remove(current_subgame)
            bucket.add(added_subgame)
        return True

    def maximal_sets(self):
        """Returns a set of the maximal sets"""
        return set.union(*self.inverted_index.values())

    def __iter__(self):
        return iter(self.maximal_sets())

    def __contains__(self, check_set):
        # Because every key in the inverted index points to every
        # set that contains that element, we only need to
        # look in the bucket of an arbitrary role or strategy.
        if not check_set:
            return True  # Empty set contained by default

        key = next(iter(check_set))
        bucket = self._inverted_index.get(key, set())
        return any(check_set <= game for game in bucket)

    def __repr__(self):
        return '{' + repr(self.maximal_sets())[5:-2] + '}'