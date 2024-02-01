import math
import torch
from bigtree import Node


class MCTS:
    """
    Class implementing the Monte-Carlo tree search algorithm.
    """

    def __init__(self, exp_const, max_planning_steps, n_actions, p_step, efe):
        """
        Construct the MCTS algorithm
        :param exp_const: the exploration constant of the MCTS algorithm
        :param max_planning_steps: the maximum number of planning iterations to perform during planning
        :param n_actions: the number of actions
        :param p_step: a function implements prediction from one time step to the next
        :param efe: a function computing the expected free energy
        """
        self.exp_const = exp_const
        self.max_planning_steps = max_planning_steps
        self.root = None
        self.n_actions = n_actions
        self.p_step = p_step
        self.efe = efe

    def step(self, state):
        """
        Perform Monte Carlo Tree Search planning
        :param state: the current state of the environment
        :return: a tensor containing the number of visits corresponding to each action
        """

        # Initialize the root node.
        self.root = Node("root", action=-1, visits=1, cost=0, state=state)

        # Perform Monte Carlo Tree Search.
        for i in range(0, self.max_planning_steps):
            node = self.select_node(self.root)
            e_nodes = self.expansion(node)
            self.evaluation(e_nodes)
            self.propagation(e_nodes)

        # Return a tensor containing the number of visits corresponding to each action.
        n_visits = {node.action: node.visits for node in self.root.children}
        return torch.tensor([[n_visits[action] for action in range(self.n_actions)]])

    def select_node(self, root):
        """
        Select the node to be expanded
        :param root: the root of the tree
        """
        current = root
        while len(current.children) != 0:
            current = max(current.children, key=lambda x: self.uct(x, self.exp_const))
        return current

    @staticmethod
    def uct(node, exp_const):
        """
        Compute the UCT criterion.
        :param node: the node for which to compute the criterion
        :param exp_const: the exploration constant.
        :return: nothing.
        """
        return - node.cost / node.visits + exp_const * math.sqrt(math.log(node.parent.visits) / node.visits)

    def expansion(self, parent):
        """
        Expand the node passed as parameters
        :param parent: the node to be expanded
        """
        nodes = []
        for action in range(0, self.n_actions):
            node = Node(str(action), action=action, visits=1, cost=0, state=self.p_step(parent, action), parent=parent)
            nodes.append(node)
        return nodes

    def evaluation(self, nodes):
        """
        Evaluate the input nodes
        :param nodes: the nodes to be evaluated
        """
        for node in nodes:
            node.cost = self.efe(node)

    def propagation(self, nodes):
        """
        Propagate the cost in the tree and update the number of visits
        :param nodes: the nodes that have been expanded
        """
        best_child = min(nodes, key=lambda x: self.efe(x))
        cost = best_child.cost
        current = best_child.parent
        while current is not None:
            current.cost += cost
            current.visits += 1
            current = current.parent
