from matplotlib import colors
import matplotlib.image as mpimg
from io import BytesIO
import pydot
import matplotlib as mpl
import matplotlib.pyplot as plt
import torch
import numpy as np


class PlotsBuilder:
    """
    Allows the creation of complex plots to visualize models based on Gaussian Mixture.
    """

    def __init__(self, title, n_rows=1, n_cols=1):

        # Store number of rows and columns.
        self.n_cols = n_cols
        self.n_rows = n_rows

        # Create the subplots, and set the main title.
        self.f, self.axes = plt.subplots(nrows=n_rows, ncols=n_cols)
        self.f.suptitle(title)

        # Create the list of colors to use.
        self.colors = list(colors.CSS4_COLORS.keys())
        first_colors = ['red', 'green', 'blue', 'purple', 'gray', 'pink', 'turquoise', 'orange', 'brown', 'cyan']
        for i, color in enumerate(first_colors):
            index = self.colors.index(color)
            i_color = self.colors[i]
            self.colors[i] = color
            self.colors[index] = i_color

        # Index of the current plot.
        self.current_plot_index = 0
        
    @property
    def current_axis(self):
        try:
            return self.axes[int(self.current_plot_index / self.n_cols)][self.current_plot_index % self.n_cols]
        except TypeError:
            return self.axes

    def draw_gaussian_mixture(self, title="", data=None, r=None, params=None, clusters=False, ellipses=True):

        # Set the subplot title.
        self.current_axis.set_title(title)

        # Draw the data points.
        if data is not None:

            # Draw the data points of t = 0.
            x = [x_tensor[0] for x_tensor in data]
            y = [x_tensor[1] for x_tensor in data]

            c = [tuple(r_n) for r_n in r] if r.shape[1] == 3 else [self.colors[torch.argmax(r_n)] for r_n in r]
            self.current_axis.scatter(x=x, y=y, c=c)

        # Draw the ellipses corresponding to the current model believes.
        if ellipses is True:
            active_components = set(r.argmax(dim=1).tolist())
            self.make_ellipses(active_components, params)

        # Draw the cluster center.
        if clusters is True:
            μ, _, _ = params
            x = [μ_k[0] for μ_k in μ]
            y = [μ_k[1] for μ_k in μ]
            self.current_axis.scatter(x=x, y=y, marker="X", s=100, c="black", edgecolor="white")

        # Move to the next axis.
        self.current_plot_index += 1

    def make_ellipses(self, active_components, params):
        m_hat, v_hat, W_hat = params
        for k in range(len(v_hat)):
            if k not in active_components:
                continue
            color = self.colors[k]
            covariances = torch.inverse(v_hat[k] * W_hat[k])
            v, w = np.linalg.eigh(covariances)
            u = w[0] / np.linalg.norm(w[0])
            angle = np.arctan2(u[1], u[0])
            angle = 180 * angle / np.pi
            v = 3. * np.sqrt(2.) * np.sqrt(np.maximum(v, 0))
            mean = m_hat[k]
            mean = mean.reshape(2, 1)
            ell = mpl.patches.Ellipse(mean, v[0], v[1], 180 + angle, color=color)
            ell.set_clip_box(self.current_axis.bbox)
            ell.set_alpha(0.5)
            self.current_axis.add_artist(ell)
            self.current_axis.set_aspect('equal', 'datalim')

    def draw_transition_graph(self, action, actions, r, title=""):

        # Set the subplot title.
        self.current_axis.set_title(title)

        # Retrieve the dataset size and the number of states.
        dataset_size = r[0].shape[0]
        n_states = r[0].shape[1]

        # Create the adjacency matrix.
        all_z0 = torch.argmax(r[0], dim=1)
        all_z1 = torch.argmax(r[1], dim=1)
        states = [
            [all_z0[n], all_z1[n]] for n in range(dataset_size) if actions[n] == action
        ]
        adjacency_matrix = torch.zeros([n_states, n_states])
        for z0, z1 in states:
            adjacency_matrix[z0][z1] += 1
        all_z = set.union(set(all_z0.tolist()), set(all_z1.tolist()))

        # Create the graph.
        graph = pydot.Dot()
        for state in range(n_states):

            # Skip nodes for states with no data points.
            if state not in all_z:
                continue

            # Add a node to the graph.
            color = colors.to_rgba(self.colors[state])
            color = [hex(int(c * 255)).replace("0x", "") for c in list(color)[0:3]]
            color = [c if len(c) == 2 else c + c for c in list(color)[0:3]]
            color = f"#{''.join(color)}88"
            graph.add_node(pydot.Node(state, label=str(state), style="filled", color=color))

        # Add the edges to the graph and create the graph label.
        sum_columns = adjacency_matrix.sum(dim=1)
        for z0 in range(n_states):
            for z1 in range(n_states):
                if adjacency_matrix[z0][z1] != 0:
                    label = round(float(adjacency_matrix[z0][z1] / sum_columns[z0]), 2)
                    graph.add_edge(pydot.Edge(z0, z1, label=label))

        # Draw the graph.
        png_img = graph.create_png()
        sio = BytesIO()
        sio.write(png_img)
        sio.seek(0)
        img = mpimg.imread(sio)
        self.current_axis.imshow(img)

        # Move to the next axis.
        self.current_plot_index += 1

    def draw_responsibility_histograms(self, r, title=""):

        # Set the subplot title.
        self.current_axis.set_title(title)

        # Retrieve the dataset size and the number of states.
        n_states = r.shape[1]

        # Draw a bar plot representing how many point are attributed to each component.
        x = [state for state in range(n_states)]
        y = r.sum(dim=0).tolist()
        bars = self.current_axis.bar(x, y, align='center')
        for state in range(n_states):
            bars[state].set_color(self.colors[state])
            bars[state].set_alpha(0.53)

        # Move to the next axis.
        self.current_plot_index += 1

    @staticmethod
    def show():
        mng = plt.get_current_fig_manager()
        mng.resize(*mng.window.maxsize())
        plt.show()
