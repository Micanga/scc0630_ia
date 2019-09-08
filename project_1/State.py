import operator


class State(object):

    # state intialisation method
    def __init__(self, c_cannibals, m_missionaries, side=1, status=(0, 0)):
        # total number of cannibals and
        # missionaries at the river
        self.c_cannibals = c_cannibals
        self.m_missionaries = m_missionaries

        # initialising the current side and status
        self.side = side
        self.status = status

    # movement method (transition function)
    def move(self, action):
        # a. doing the action
        new_status = tuple(map(operator.add, self.status, tuple([self.side * x for x in action])))

        # b. checking if the action is valid
        if (new_status[0] > new_status[1] and new_status[1] != 0) \
                or ((self.c_cannibals - new_status[0]) > (self.m_missionaries - new_status[1]) \
                    and (self.m_missionaries - new_status[1]) != 0):
            return None
        if new_status[0] < 0 or new_status[1] < 0 \
                or (self.c_cannibals - new_status[0]) < 0 \
                or (self.m_missionaries - new_status[1]) < 0:
            return None

        # c. if is valid, return the new state
        new_state = State(self.c_cannibals, self.m_missionaries, \
                          self.other_side(), new_status)
        return new_state

    # return the other side based on the current side
    def other_side(self):
        return ((-1) * self.side)

    # verify if the current state is equal to another
    def is_equal(self, state):
        if self.side == state.side \
                and self.c_cannibals == state.c_cannibals \
                and self.m_missionaries == state.m_missionaries \
                and self.side == state.side \
                and self.status[0] == state.status[0] \
                and self.status[1] == state.status[1]:
            return True
        return False