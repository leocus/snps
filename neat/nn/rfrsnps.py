import torch as t

from neat.graphs import feed_forward_layers


def elementwise_prod(a, b):
    c = t.zeros((b.shape[0], b.shape[1], a.shape[1]))
    for i in range(a.shape[1]):
        c[:, :, i] = a[:, i] * b
    return c


class VectorizedSpikingPFeedForwardNetwork(object):
    def __init__(self, inputs, outputs, node_evals):
        self.input_nodes = inputs
        self.output_nodes = outputs
        self.node_evals = node_evals

    def activate(self, inputs):
        rows = len(inputs)
        n_in = len(self.input_nodes)
        n_out = len(self.output_nodes)
        n_ho = len(self.node_evals)  # Hidden + outputs
        tot = n_in + n_ho

        theta = t.zeros((rows, tot)).to("cuda")
        delta = t.zeros((rows, tot)).to("cuda")
        d1 = t.zeros((tot, tot)).to("cuda")
        d2 = t.zeros((tot, tot)).to("cuda")
        e = t.zeros((tot, tot)).to("cuda")
        c = t.ones((tot, tot)).to("cuda")

        rules = []
        propositions = [*range(n_in)]
        transform = {-i: i - 1 for i in range(1, n_in + 1)}

        for node_id, _, _, confidence, n, _, _, links in self.node_evals:
            transform[node_id] = len(transform)

        for node_id, _, _, confidence, n, _, _, links in self.node_evals:
            i = transform[node_id]
            if int(n) % 2 == 0:
                rules.append(i)
            else:
                propositions.append(i)
            c[i, i] = confidence

        for node_id, _, p, confidence, n, isand, _, links in self.node_evals:
            i = transform[node_id]
            if i in rules:
                for oth_id_, weight in links:
                    id_ = transform[oth_id_]
                    if id_ in propositions:
                        d1[i, id_] = 1
                        d2[i, id_] = 0 if isand < 0.5 else 1
            else:
                for oth_id_, weight in links:
                    id_ = transform[oth_id_]
                    if id_ in rules:
                        e[i,  id_] = 1

        output_nodes_not_connected = t.sum(t.Tensor([i not in transform for i in range(n_out)])) > 0
        if output_nodes_not_connected:
            # output nodes not connected
            return t.zeros((rows, n_out)).cpu().detach().numpy()

        timestep = 0
        out_indices = [transform[i] for i in range(n_out)]
        output = t.zeros((rows, n_out)).to("cuda")

        while timestep == 0 or t.norm(delta) > 0:
            if timestep < inputs.shape[2]:
                theta[:, :n_in] = t.Tensor(inputs[:, :, timestep]).to("cuda")
            delta = t.mm(theta, d1.T) + t.min(elementwise_prod(d2.T, theta), 2).values.to("cuda")
            theta = t.max(elementwise_prod(e.T, t.mm(delta, c)), 2).values.to("cuda")
            output += theta[:, out_indices] + delta[:, out_indices]
            timestep += 1

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
                """
                aggregation_function = config.genome_config.aggregation_function_defs.get(ng.aggregation)
                activation_function = config.genome_config.activation_defs.get(ng.activation)  # For future use
                """
                node_evals.append((node, ng.c1, ng.p1, ng.d1, ng.c2, ng.p2, ng.d2, inputs))

        return VectorizedSpikingPFeedForwardNetwork(config.genome_config.input_keys, config.genome_config.output_keys, node_evals)
