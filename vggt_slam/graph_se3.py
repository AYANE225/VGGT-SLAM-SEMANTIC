import numpy as np
import gtsam
from gtsam import noiseModel
from gtsam.symbol_shorthand import X


class PoseGraph:
    """
    SE3 pose graph (Pose3).
    """

    def __init__(self):
        self.graph = gtsam.NonlinearFactorGraph()
        self.initial = gtsam.Values()
        self.result = None

        self._relative_sigmas = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1], dtype=float)
        self.relative_noise = noiseModel.Diagonal.Sigmas(self._relative_sigmas)

        self.anchor_noise = noiseModel.Diagonal.Sigmas([1e-6] * 6)

        self.num_loop_closures = 0

    def add_homography(self, key, pose):
        k = X(int(key))
        P = gtsam.Pose3(pose)
        if self.initial.exists(k):
            self.initial.update(k, P)
        else:
            self.initial.insert(k, P)

    def _scaled_relative_noise(self, weight: float):
        w = float(weight)
        if not np.isfinite(w) or w <= 0:
            w = 1.0
        sig = self._relative_sigmas / np.sqrt(w)
        sig = np.maximum(sig, 1e-12)
        return noiseModel.Diagonal.Sigmas(sig)

    def add_between_factor(self, key1, key2, relative_pose, noise):
        k1 = X(int(key1))
        k2 = X(int(key2))
        P = gtsam.Pose3(relative_pose)
        self.graph.add(gtsam.BetweenFactorPose3(k1, k2, P, noise))

    def add_between_factor_weighted(self, key1, key2, relative_pose, weight: float):
        noise = self._scaled_relative_noise(weight)
        self.add_between_factor(key1, key2, relative_pose, noise)

    def add_prior_factor(self, key, pose, noise):
        k = X(int(key))
        P = gtsam.Pose3(pose)
        self.graph.add(gtsam.PriorFactorPose3(k, P, noise))

    def get_homography(self, node_id):
        k = X(int(node_id))
        if self.result is not None and self.result.exists(k):
            return self.result.atPose3(k).matrix()
        if self.initial.exists(k):
            return self.initial.atPose3(k).matrix()
        return None

    def optimize(self):
        params = gtsam.LevenbergMarquardtParams()
        params.setVerbosityLM("ERROR")
        params.setVerbosity("ERROR")
        optimizer = gtsam.LevenbergMarquardtOptimizer(self.graph, self.initial, params)
        self.result = optimizer.optimize()
        return self.result

    def print_estimates(self):
        if self.result is None:
            print("[PoseGraph-SE3] no result yet.")
            return
        keys = list(self.result.keys())
        keys.sort()
        for k in keys:
            try:
                T = self.result.atPose3(k).matrix()
                print(k, "\n", T)
            except Exception:
                pass

    def increment_loop_closure(self):
        self.num_loop_closures += 1

    def get_num_loops(self):
        return self.num_loop_closures
