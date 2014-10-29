import numpy as np
import itertools
from viterbi import viterbi_decoding


class TestViterbiDecoding:

    def setup(self):

        from scipy.stats import norm
        from numpy import vstack

        # sample observation from three states with normal density
        F = [norm(0, 0.3), norm(1, 0.45), norm(2, 0.15)]

        # 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 2 2 2 2 2
        # 1 1 1 1 1 0 0 0 0 0 0 0 0 0 0 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 1 2 2 2 2
        self.states = (
            (10 * [0]) + (20 * [1]) + (5 * [2]) + (5 * [1]) +
            (10 * [0]) + (15 * [2]) + (1 * [1]) + (4 * [2])
        )

        obversation = [F[s].rvs() for s in self.states]

        self.emission = vstack([f.logpdf(obversation) for f in F]).T

        self.transition = np.log([[0.5, 0.1, 0.4],
                                  [0.1, 0.8, 0.1],
                                  [0.2, 0.3, 0.5]])

    def test_simple(self):

        decoded = viterbi_decoding(self.emission, self.transition)
        errors = np.sum(decoded != self.states)
        assert float(errors) / len(self.states) < 0.2

    def test_consecutive(self):
        for consecutive in [1, 2, 5, 10, 20, 50]:
            yield self.check_consecutive, consecutive

    def check_consecutive(self, consecutive):

        decoded = viterbi_decoding(self.emission, self.transition,
                                   consecutive=consecutive)

        changeStatesAt = [-1] + list(np.where(np.diff(decoded) != 0)[0])
        lengths = np.diff(changeStatesAt)
        assert np.min(lengths) >= consecutive

    def test_force(self):
        for ratio in [0.01, 0.05, 0.1, 0.2, 0.5]:
            yield self.check_force, ratio

    def check_force(self, ratio):

        T, K = self.emission.shape

        force = -np.ones((T, ), dtype=int)

        N = int(ratio * T)
        Ts = np.random.randint(T, size=N)
        Ks = np.random.randint(K, size=N)
        for t, k in itertools.izip(Ts, Ks):
            force[t] = k

        decoded = viterbi_decoding(self.emission, self.transition, force=force)

        for t in Ts:
            assert decoded[t] == force[t]
