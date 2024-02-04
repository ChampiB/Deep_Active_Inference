from bigtree import Node
import torch
from zoo.agents.inference.GaussianMixture import GaussianMixture


class HierarchicalGM:

    @staticmethod
    def split_components(gm, init_gm):

        # Recursively learns all sub Gaussian Mixture.
        root = Node("gm", gm=gm)
        HierarchicalGM.learn_sub_gms(root, init_gm)

        # Combine all the Gaussian Mixtures in the tree to create the overall Gaussian Mixture.
        new_gm = HierarchicalGM.combine(root)
        new_gm.learn()
        return new_gm

    @staticmethod
    def learn_sub_gms(node, init_gm, min_data_points=10):

        # Check whether to stop the recursion.
        if node.parent is not None and node.gm.vfe > node.parent.gm.vfe:
            return

        for k in range(node.gm.K):

            # Retrieving data for the Gaussian Mixture corresponding to the current state.
            x = node.gm.data_of_component(k)
            if len(x) < min_data_points:
                continue

            # Learns the Gaussian Mixture corresponding to the current state.
            sub_gm = init_gm(x)
            sub_gm.learn()
            child = Node(str(k), gm=sub_gm, parent=node)
            HierarchicalGM.learn_sub_gms(child, init_gm)

    @staticmethod
    def combine(root):

        # Retrieve the components of the combined Gaussian Mixture.
        components = HierarchicalGM.find_terminal_components(root)

        # Unpack the parameters of the components.
        params = list(zip(*components))
        W, m, v, β, d, W_hat, m_hat, v_hat, β_hat, d_hat = [list(param) for param in params]
        v = torch.stack(v)
        β = torch.stack(β)
        d = torch.stack(d)
        v_hat = torch.stack(v_hat)
        β_hat = torch.stack(β_hat)
        d_hat = torch.stack(d_hat)

        # Create the combined Gaussian Mixture.
        gm = GaussianMixture(x=root.gm.x, W=W, m=m, v=v, β=β, d=d)
        gm.W_hat = W_hat
        gm.m_hat = m_hat
        gm.v_hat = v_hat
        gm.β_hat = β_hat
        gm.d_hat = d_hat
        return gm

    @staticmethod
    def find_terminal_components(parent):

        # Initialize the list of terminal components.
        components = []

        # Retrieve expanded nodes that are terminal.
        if len(parent.children) == 0:
            active_ks = parent.gm.active_components
            for k in active_ks:
                components.append(parent.gm.params(k))
            return components

        # Keep track of active components, and call the function recursively for each child.
        active_ks = parent.gm.active_components
        for child in parent.children:
            active_ks = active_ks.difference({int(child.name)})
            components.extend(HierarchicalGM.find_terminal_components(child))

        # Retrieve non-expanded nodes that are terminal.
        for k in active_ks:
            components.append(parent.gm.params(k))

        return components
