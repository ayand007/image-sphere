import numpy as np
class History:

    def __init__(self):
        self.states = []
        self.index = -1

    def push(self, state):
        state = np.ascontiguousarray(state)

        if self.index >= 0:
            if np.array_equal(self.states[self.index], state):
                return False

        self.states = self.states[:self.index + 1]

        self.states.append(state.copy())
        self.index += 1

        return True

    def undo(self):
        if self.index > 0:
            self.index -= 1

        return self.current()

    def redo(self):
        if self.index < len(self.states) - 1:
            self.index += 1

        return self.current()

    def current(self):
        if self.states:
            return self.states[self.index]

        return None