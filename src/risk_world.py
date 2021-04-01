import numpy as np
from itertools import product
import random


class RiskWorld:

    def __init__(self, size):
        self.size = size

        self.game_board = self._generate_game()

        self.actions = self._calc_actions()

        #TODO: fix the state math
        self.n_states = self.size #This is 100% wrong but I've forgotten the state math
        self.n_actions = len(self.actions["red"]) + len(self.actions["blue"])
        
        self.game_board = self._protected_move([[(0, 0), (1, 0)], [(2, 1), (2, 0)]])

        print(self.game_board)

        print(*self._transition_prob(self.game_board, [(1, 0), (2, 0)])[0], sep="\n")
        print(*self._transition_prob(self.game_board, [(1, 0), (2, 0)])[1], sep="\n")

        self.actions = self._calc_actions() #update the actions
        

    
    def _calc_actions(self):
        """
        calculates the possible actions given the current state
        actions are stored as a pair of tuples, the start and end position of a troop movement
        """
        actions_red = []
        actions_blue = []
        for i in range(self.size):
            for j in range(self.size):
                actions = []
                if self.game_board[i][j][0] != "x":
                    if i - 1 >= 0:
                        actions.append([(i, j), (i - 1, j)])
                    if i + 1 < self.size:
                        actions.append([(i, j), (i + 1, j)])
                    if j - 1 >= 0:
                        actions.append([(i, j), (i, j - 1)])
                    if j + 1 < self.size:
                        actions.append([(i, j), (i, j + 1)])
                if self.game_board[i][j][0] == "R":
                    actions_red += actions
                elif self.game_board[i][j][0] == "B":
                    actions_blue += actions
        return {"red":actions_red, "blue":actions_blue}
            


        


    def _generate_game(self):
        """
        Builds a board of of size by size

        Returns
        -------
        game_board : 2d np.array
        "b" is a blue square, "B" is a blue square with a troop same applies for red with "r" and "R"
        multiple troops = multiple chars so 3 red troops would be "RRR"
        empty colorless spaces are "x"

        """
        row1 = ["B"] * self.size
        board = [["x"] * self.size for _ in range(self.size - 2)]
        board.insert(0, row1)
        board.append(["R"] * self.size)
        board = np.array(board, dtype=object)
        print(board)

        return board

    
    def _transition_prob_table(self):
        """
        Builds the internal probability transition table.

        Returns:
            The probability transition table of the form

                [state_from, state_to, action]

            containing all transition probabilities. The individual
            transition probabilities are defined by `self._transition_prob'.
        """
        raise NotImplementedError
        #It seems we're transitioning away from using this function. (get it?)

    def _transition_prob(self, s_from, a):
        """
        Compute the transition probability for a single transition.

        Args:
            s_from: The state in which the transition originates. In the form of a board or 2d nparray
            a: The action via which the target state should be reached.

        Returns:
            A vector containing the transition probability from `s_from` 
            to all states under action `a`.
        """
        color = s_from[a[0][0], a[0][1]][0]
        if s_from[a[1][0], a[1][1]].islower() or s_from[a[0][0], a[0][1]][0] == s_from[a[1][0], a[1][1]][0]:
            return [(1.0, self._protected_move(a))] #this action can only lead to one state
        
        #TODO fix the two outcome thing
        p = min(len(s_from[a[1][0], a[1][1]]), len(s_from[a[0][0], a[0][1]])) / (max(len(s_from[a[1][0], a[1][1]]), len(s_from[a[0][0], a[0][1]])) + 1)
        #I'm not 100% sure this is correct ^^
        red_out = s_from.copy()
        blue_out = s_from.copy()
        blue_out[a[0][0], a[0][1]] = blue_out[a[0][0], a[0][1]].lower()
        red_out[a[0][0], a[0][1]] = red_out[a[0][0], a[0][1]].lower()
        blue_out[a[1][0], a[1][1]] = "B"
        red_out[a[1][0], a[1][1]] = "R"
        out = []
        if color == "B": #there are only two outcomes so I don't need 4 conditions I just can't think of a better way atm
            if p > 0.5:
                out = [(1 - p, blue_out), (p, red_out)]
            else:
                out = [(p, blue_out), (1 - p, red_out)]
        else:
            if p > 0.5:
                out = [(1 - p, red_out), (p, blue)]
            else:
                out = [(p, red_out), (1 - p, blue_out)]

        return out
        
        
    def _protected_move(self, actions):
        """
        Note, this peforms no input validation

        Parameters
        ----------
        a : TYPE
            the actions to be performed in the format of [[(x1, y1), (x2, y2)], [(x1, y1), (x2, y2)], ...]
            where x1, y1 are the start positions of the troop and x2, y2 are the end positions
        Returns
        -------
        TYPE
            Returns the new board.
            Note it does not update the internal board or the actions
        """
        next_board = self.game_board.copy()
        for a in actions:
            cur = self.game_board[a[0][0]][a[0][1]]
            if len(cur) == 1:
                next_board[a[0][0]][a[0][1]] = cur.lower()
            else:
                next_board[a[0][0]][a[0][1]] = cur[:-1]
            if next_board[a[1][0]][a[1][1]].islower():
                next_board[a[1][0]][a[1][1]] = cur
            else:
                next_board[a[1][0]][a[1][1]] += cur

        for i in range(self.size):
            for j in range(self.size):
                cur = next_board[i][j]
                if cur != len(cur) * cur[0]:
                    reds = []
                    blues = []
                    for c in cur:
                        if c == "R":
                            reds.append(random.uniform(0, 1))
                        else:
                            blues.append(random.uniform(0, 1))
                    if max(reds) > max(blues):
                        next_board[i][j] = "R"
                    else:
                        next_board[i][j] = "B"

        self.game_board = next_board
        self.actions = self._calc_actions()
        return self.game_board
        
        

    def __repr__(self):
        return self.game_board.__repr__()

