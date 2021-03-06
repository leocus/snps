import torch as t

from neat.graphs import feed_forward_layers


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
            inputs = t.Tensor(inputs).to("cuda")

            tot = n_in + n_ho
            state = t.zeros((rows, tot)).to("cuda")
            firing = t.zeros((rows, tot)).to("cuda")
            output = t.zeros((rows, n_out)).to("cuda")
            T = t.ones(tot).to("cuda")
            P = t.ones(tot).to("cuda")
            D = t.zeros(tot).to("cuda")
            T2 = t.ones(tot).to("cuda")
            P2 = t.ones(tot).to("cuda")
            D2 = t.zeros(tot).to("cuda")
            weights = t.zeros((tot, tot)).to("cuda")

            # Init firing with inputs
            state[:, :n_in] = inputs

            translation = {i: i + n_in for i in self.input_nodes}

            for i, _, _, _, _, _, _, _ in self.node_evals:
                if i >= n_out:
                    # Hidden node
                    translation[i] = max(translation.values())
                else:
                    # Output node
                    translation[i] = i + tot - n_out

            for i, c1, p1, d1, c2, p2, d2, links in self.node_evals:
                for j, val in links:
                    weights[translation[j], translation[i]] = np.sign(val)
                T[translation[i]] = int(c1)
                P[translation[i]] = int(p1)
                D[translation[i]] = int(d1)
                T2[translation[i]] = int(c2)
                P2[translation[i]] = int(p2)
                D2[translation[i]] = int(d2)

            P[T < P] = T[T < P]
            P2[T2 < P2] = T2[T2 < P2]
            curD = t.zeros((rows, tot)).to("cuda")
            iterations = 0
            max_iterations = 100

            while (iterations < max_iterations and t.sum(firing) > 0) or initial:
                initial = False
                state += t.mm(firing, weights)
                state[state < 0] = 0
                firing = t.logical_and(state >= T, curD <= 0)
                state -= firing * T
                tmp = t.zeros((rows, tot)).to("cuda")
                tmp[firing] = 1.0
                tmp *= D
                curD += tmp
                firing = firing * P
                output += firing[:, -n_out:]
                iterations += 1
                if t.sum(output) > 0:
                    break

                """
                # Execute second rule
                firing2 = t.logical_and(t.logical_and(state > T2, curD == 0), firing == 0)
                state -= firing2 * T2
                tmp = t.zeros((rows, tot)).to("cuda")
                tmp[firing2] = 1.0
                tmp *= D2
                curD += tmp
                firing2 = firing2 * P2
                output += firing2[:, -n_out:]

                curD -= 1
                curD[curD < 0] = 0
                firing += firing2
                """
        return output.cpu().detach().numpy()

    """
    def activate(self, inputs):
        size = len(inputs)
        self.reset_state(size)
        outputs = t.zeros((size, len(self.output_nodes)))
        self.values = dict((key, t.zeros(size)) for key in self.input_nodes + self.output_nodes)

        for k, v in zip(self.input_nodes, inputs.T):
            self.values[k] = v

        new_spikes = True
        while new_spikes:
            new_spikes = False

            for node, act_func, agg_func, bias, response, links in self.node_evals:
                node_spikes = self.node_state[node]

                for i, w in links:
                    node_spikes += int(t.round(w)) * (self.values[i])

                self.values[node] = t.zeros(size)
                fired = node_spikes >= round(bias)
                self.values[node][fired] = 1
                if t.sum(fired) > 0:
                    new_spikes = True
                node_spikes[fired] -= round(bias)

                self.node_state[node] = node_spikes

            for i, on in enumerate(self.output_nodes):
                outputs[:, i] += self.values[on]

            # Remove input
            for k, _ in zip(self.input_nodes, inputs):
                self.values[k] = t.zeros(size)
        return outputs
    """

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
                """
                aggregation_function = config.genome_config.aggregation_function_defs.get(ng.aggregation)
                activation_function = config.genome_config.activation_defs.get(ng.activation)  # For future use
                """
                node_evals.append((node, ng.c1, ng.p1, ng.d1, ng.c2, ng.p2, ng.d2, inputs))

        return VectorizedSpikingPFeedForwardNetwork(config.genome_config.input_keys, config.genome_config.output_keys, node_evals)

    def reset_state(self, size):
        self.node_state = {k[0]: t.zeros(size) for k in self.node_evals}
