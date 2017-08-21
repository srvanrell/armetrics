
class Event:
    def __init__(self, start=None, end=None, label=""):
        self.start = start
        self.end = end
        self.label = label

    def __repr__(self):
        return str([self.start, self.end, self.label])

    def add_label(self, label):
        if label not in self.label:
            self.label += label
            if self.label == "MF":
                self.label = "FM"

    def overlap(self, event):
        """
        Return True if given event overlaps with this one (at the beginning or the end). 
        :param event: 
        :return: 
        """
        return self.start <= event.start < self.end or self.start <= event.end < self.end


class Segment:
    def __init__(self, start=None, end=None, label=""):
        self.start = start
        self.end = end
        self.label = label

    def __repr__(self):
        return str([self.start, self.end, self.label])
