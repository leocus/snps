import torch as t
import numpy as np

from neat.graphs import feed_forward_layers


device = "cpu"


class VectorizedSpikingPFeedForwardNetwork(object):
    def __init__(self, inputs, outputs, node_evals):
        self.input_nodes = inputs
        self.output_nodes = outputs
        self.node_evals = node_evals
        self.values = dict((key, 0.0) for key in inputs + outputs)
        self.reset_state(1)

    def activate(self, inputs):
        rows = len(inputs)
        n_in = len(self.input_nodes)
        n_out = len(self.output_nodes)
        n_hidden = len([a for a in self.node_evals if a[0] not in self.output_nodes])
        n_ho = n_hidden + n_out  # Hidden + outputs
        initial = True

        with t.no_grad():
            inputs = t.Tensor(inputs).to(device)

            tot = n_in + n_ho
            state = t.zeros((rows, tot)).to(device)
            firing = t.zeros((rows, tot)).to(device)
            output = t.zeros((rows, n_out)).to(device)
            T = t.ones(tot).to(device)
            P = t.ones(tot).to(device)
            D = t.zeros(tot).to(device)
            T2 = t.ones(tot).to(device)
            P2 = t.ones(tot).to(device)
            D2 = t.zeros(tot).to(device)
            weights = t.zeros((tot, tot)).to(device)

            # Init firing with inputs
            state[:, :n_in] = inputs

            translation = {i: i + n_in for i in self.input_nodes}
            translation.update({i: tot - n_out + i for i in range(n_out)})

            for i, _, _, _, _, _, _, _ in self.node_evals:
                if i >= n_out:
                    # Hidden node
                    translation[i] = len(translation) - n_out
                assert np.sum([1 if translation[i] == translation[j] else 0 for j in translation.keys()]) == 1, (translation, [k[0] for k in self.node_evals])

            for i, c1, p1, d1, c2, p2, d2, links in self.node_evals:
                for j, val in links:
                    weights[translation[j], translation[i]] = int(val)  #  if val > 0 else 0
                T[translation[i]] = int(c1)
                P[translation[i]] = int(p1)
                D[translation[i]] = int(d1)
                T2[translation[i]] = int(c2)
                P2[translation[i]] = int(p2)
                D2[translation[i]] = int(d2)

            P[T < P] = T[T < P]
            P2[T2 < P2] = T2[T2 < P2]
            T2[T2 >= T] = T[T2 >= T] - 1
            curD = t.zeros((rows, tot)).to(device)
            iterations = 0
            max_iterations = 100

            while (iterations < max_iterations and t.sum(firing) > 0) or initial:
                initial = False
                curD -= 1
                state += t.mm(firing, weights)
                state[state < 0] = 0
                state[state == T2] = 0
                firing = t.logical_and(state >= T, curD <= 0)
                state -= firing * T
                tmp = t.zeros((rows, tot)).to(device)
                tmp[firing] = 1.0
                tmp *= D
                curD += tmp + 1
                firing = firing * P
                output += firing[:, -n_out:]
                iterations += 1

        return output.cpu().detach().numpy()


    @staticmethod
    def create(genome, config):
        """ Receives a genome and returns its phenotype (a FeedForwardNetwork). """

        # Gather expressed connections.
        connections = [cg.key for cg in genome.connections.values() if cg.enabled]

        layers = feed_forward_layers(config.genome_config.input_keys, config.genome_config.output_keys, connections)
        node_evals = []
        for layer in layers:
            for node in layer:
                inputs = []
                node_expr = []  # currently unused
                for conn_key in connections:
                    inode, onode = conn_key
                    if onode == node:
                        cg = genome.connections[conn_key]
                        inputs.append((inode, cg.weight))
                        node_expr.append("v[{}] * {:.7e}".format(inode, cg.weight))

                ng = genome.nodes[node]
                node_evals.append((node, ng.c1, ng.p1, ng.d1, ng.c2, ng.p2, ng.d2, inputs))

        return VectorizedSpikingPFeedForwardNetwork(config.genome_config.input_keys, config.genome_config.output_keys, node_evals)

    def reset_state(self, size):
        self.node_state = {k[0]: t.zeros(size) for k in self.node_evals}
