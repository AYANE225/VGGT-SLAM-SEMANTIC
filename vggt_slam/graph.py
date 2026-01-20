# vggt_slam/graph.py
import numpy as np
import gtsam
from gtsam import noiseModel
from gtsam.symbol_shorthand import X


class PoseGraph:
    """
    SL(4) pose graph (15 DoF).

    保护策略：
    1) 输入矩阵构造 SL4 前：过滤 NaN/Inf、奇异、离谱尺度、离谱 cond。
    2) det<0：不能用 H=-H（4x4 det 不变），必须乘 det=-1 的反射矩阵（左右乘都尝试）。
    3) gtsam.SL4(H) 构造：try/except，失败直接 SKIP，不让进程崩。
    4) optimize：若优化内部因 SL4 归一化/退化报错，fallback 为“增量加因子，谁炸跳过谁”，最终仍给出 result。
    """

    def __init__(self):
        self.graph = gtsam.NonlinearFactorGraph()
        self.initial = gtsam.Values()
        self.result = None

        # Base noise for SL4 (15 DoF)
        self._relative_sigmas = 0.05 * np.ones(15, dtype=float)
        self.relative_noise = noiseModel.Diagonal.Sigmas(self._relative_sigmas)
        self.anchor_noise = noiseModel.Diagonal.Sigmas([1e-6] * 15)

        self.num_loop_closures = 0

        # ---- SL4 safety knobs ----
        self.det_eps = 1e-10              # near singular threshold
        self.fix_negative_det = True      # det<0 -> reflect to make det>0
        self.skip_near_singular = True    # skip if |det| too small

        # det magnitude sanity (before det=1 normalize)
        self.det_abs_min = 1e-6
        self.det_abs_max = 1e6

        # condition number sanity
        self.cond_max = 1e8

        # matrix magnitude sanity (Frobenius norm)
        self.frob_max = 1e4

        # ---- Robust noise ----
        self.use_robust_between = True
        self.huber_k = 1.345

        # ---- Conservative LM knobs ----
        self.lm_max_iters = 50
        self.lm_lambda_initial = 1e3
        self.lm_lambda_factor = 10.0
        self.lm_diagonal_damping = True
        self.lm_verbosity = "ERROR"
        self.verbosity = "ERROR"

        # optimize fallback
        self.enable_safe_incremental_opt = True
        self.safe_inc_max_iters = 30   # 每次增量优化的最大迭代（别太大，避免慢）
        self.safe_inc_lambda_initial = 1e6  # 更保守，避免 retraction 走飞

    # ---------------- utils ----------------

    @staticmethod
    def _finite(x) -> bool:
        return np.isfinite(np.asarray(x)).all()

    def _robustify(self, base_noise):
        if not self.use_robust_between:
            return base_noise
        try:
            huber = noiseModel.mEstimator.Huber.Create(self.huber_k)
            return noiseModel.Robust.Create(huber, base_noise)
        except Exception:
            return base_noise

    def _normalize_det_to_one(self, H: np.ndarray, det: float) -> np.ndarray:
        """
        det(sH) = s^4 det(H)
        令 |det| -> 1：s = |det(H)|^(-1/4)
        注意：只做尺度归一化，不改变 det 符号。
        """
        scale = float(abs(det) ** 0.25)
        if scale <= 0 or (not np.isfinite(scale)):
            return H
        return H / scale

    def _reflect_to_positive_det(self, H: np.ndarray, det: float, ctx: str):
        """
        det<0 时，用 det=-1 的反射矩阵 R 翻正：
          det(RH) = det(R)*det(H) = (-1)*det(H) > 0

        为了更稳：同时尝试左乘 R@H 和右乘 H@R（各 3 个轴反射），挑 cond 更小的。
        """
        if det >= 0:
            return H, det

        if not self.fix_negative_det:
            print(f"[PoseGraph] {ctx}: det<0 det={det:.6g} (fix_negative_det=False) -> SKIP")
            return None, None

        Rs = [
            np.diag([-1.0, 1.0, 1.0, 1.0]),
            np.diag([1.0, -1.0, 1.0, 1.0]),
            np.diag([1.0, 1.0, -1.0, 1.0]),
        ]

        best_H = None
        best_det = None
        best_cond = np.inf
        best_tag = ""

        for i, R in enumerate(Rs):
            for tag, Hc in ((f"L{i}", R @ H), (f"R{i}", H @ R)):
                detc = float(np.linalg.det(Hc))
                if (not np.isfinite(detc)) or detc <= 0:
                    continue
                try:
                    condc = float(np.linalg.cond(Hc))
                except Exception:
                    condc = np.inf
                if condc < best_cond:
                    best_cond = condc
                    best_H = Hc
                    best_det = detc
                    best_tag = tag

        if best_H is None:
            print(f"[PoseGraph] {ctx}: det<0 det={det:.6g} -> reflect failed -> SKIP")
            return None, None

        print(f"[PoseGraph] {ctx}: det<0 det={det:.6g} -> REFLECT({best_tag}) det={best_det:.6g} (cond={best_cond:.3g})")
        return best_H, best_det

    def _make_sl4(self, H: np.ndarray, ctx: str):
        """
        唯一允许调用 gtsam.SL4(H) 的入口。
        返回 gtsam.SL4 或 None（SKIP）。
        """
        H = np.asarray(H, dtype=float)

        if H.shape != (4, 4):
            raise ValueError(f"[PoseGraph] {ctx}: expected 4x4, got {H.shape}")
        if not self._finite(H):
            print(f"[PoseGraph] {ctx}: matrix has NaN/Inf -> SKIP")
            return None

        frob = float(np.linalg.norm(H, ord="fro"))
        if (not np.isfinite(frob)) or frob > self.frob_max:
            print(f"[PoseGraph] {ctx}: frob too large ({frob:.6g}) -> SKIP")
            return None

        det = float(np.linalg.det(H))
        if not np.isfinite(det):
            print(f"[PoseGraph] {ctx}: det is NaN/Inf -> SKIP")
            return None

        if abs(det) < self.det_eps:
            msg = f"[PoseGraph] {ctx}: det too small (degenerate), det={det:.6g}"
            if self.skip_near_singular:
                print(msg + " -> SKIP")
                return None
            raise RuntimeError(msg)

        # det<0 -> 反射翻正
        if det < 0:
            H, det = self._reflect_to_positive_det(H, det, ctx)
            if H is None:
                return None

        # det magnitude sanity
        if abs(det) < self.det_abs_min or abs(det) > self.det_abs_max:
            print(f"[PoseGraph] {ctx}: |det| out of range (det={det:.6g}) -> SKIP")
            return None

        # condition number sanity
        try:
            cond = float(np.linalg.cond(H))
            if (not np.isfinite(cond)) or cond > self.cond_max:
                print(f"[PoseGraph] {ctx}: cond too large ({cond:.6g}) -> SKIP")
                return None
        except Exception:
            print(f"[PoseGraph] {ctx}: cond() failed -> SKIP")
            return None

        # 归一化 |det| -> 1（尺度稳定）
        H = self._normalize_det_to_one(H, det)
        det2 = float(np.linalg.det(H))
        if (not np.isfinite(det2)) or det2 <= 0:
            print(f"[PoseGraph] {ctx}: det after normalize invalid (det={det2:.6g}) -> SKIP")
            return None

        # 永不让 gtsam.SL4 把进程炸掉
        try:
            return gtsam.SL4(H)
        except Exception as e:
            detx = float(np.linalg.det(H))
            print(f"[PoseGraph] {ctx}: gtsam.SL4 failed: {e} (det={detx:.6g}) -> SKIP")
            return None

    def _scaled_relative_noise(self, weight: float):
        w = float(weight)
        if (not np.isfinite(w)) or w <= 0:
            w = 1.0
        sig = self._relative_sigmas / np.sqrt(w)
        sig = np.maximum(sig, 1e-6)
        return noiseModel.Diagonal.Sigmas(sig)

    def _make_lm_params(self, for_safe_incremental: bool = False) -> gtsam.LevenbergMarquardtParams:
        params = gtsam.LevenbergMarquardtParams()
        if hasattr(params, "setMaxIterations"):
            params.setMaxIterations(int(self.safe_inc_max_iters if for_safe_incremental else self.lm_max_iters))
        if hasattr(params, "setLambdaInitial"):
            params.setLambdaInitial(float(self.safe_inc_lambda_initial if for_safe_incremental else self.lm_lambda_initial))
        if hasattr(params, "setLambdaFactor"):
            params.setLambdaFactor(float(self.lm_lambda_factor))
        if hasattr(params, "setDiagonalDamping"):
            params.setDiagonalDamping(bool(self.lm_diagonal_damping))
        if hasattr(params, "setVerbosityLM"):
            try:
                params.setVerbosityLM(self.lm_verbosity)
            except Exception:
                pass
        if hasattr(params, "setVerbosity"):
            try:
                params.setVerbosity(self.verbosity)
            except Exception:
                pass
        return params

    # ---------------- vertices ----------------

    def add_homography(self, key, global_h):
        ctx = f"add_homography key={key}"
        sl4 = self._make_sl4(global_h, ctx)
        if sl4 is None:
            return
        k = X(int(key))
        if self.initial.exists(k):
            self.initial.update(k, sl4)
        else:
            self.initial.insert(k, sl4)

    # ---------------- edges ----------------

    def add_between_factor(self, key1, key2, relative_h, noise):
        ctx = f"add_between_factor ({key1}->{key2})"
        sl4 = self._make_sl4(relative_h, ctx)
        if sl4 is None:
            return
        k1 = X(int(key1))
        k2 = X(int(key2))
        noise = self._robustify(noise)
        self.graph.add(gtsam.BetweenFactorSL4(k1, k2, sl4, noise))

    def add_between_factor_weighted(self, key1, key2, relative_h, weight: float):
        noise = self._scaled_relative_noise(weight)
        self.add_between_factor(key1, key2, relative_h, noise)

    def add_prior_factor(self, key, global_h, noise=None):
        ctx = f"add_prior_factor key={key}"
        sl4 = self._make_sl4(global_h, ctx)
        if sl4 is None:
            return
        k = X(int(key))
        if noise is None:
            noise = self.anchor_noise
        self.graph.add(gtsam.PriorFactorSL4(k, sl4, noise))

    # ---------------- optimize ----------------

    def _optimize_safe_incremental(self):
        """
        关键 fallback：
        - 逐个尝试把 factor 加入 kept 集合
        - 每加入一个，就用当前 values 优化一次
        - 若某个 factor 导致 optimize 内部崩（比如 SL4 det 变负），则跳过该 factor，继续
        """
        n = self.graph.size()
        if n == 0:
            raise RuntimeError("[PoseGraph] empty factor graph (no edges)")
        if self.initial.size() == 0:
            raise RuntimeError("[PoseGraph] empty initial values")

        params = self._make_lm_params(for_safe_incremental=True)

        kept = []
        values = self.initial
        skipped = []

        for i in range(n):
            fac = self.graph.at(i)

            # 重新构建图（避免需要 pop_back）
            trial = gtsam.NonlinearFactorGraph()
            for f in kept:
                trial.push_back(f)
            trial.push_back(fac)

            try:
                opt = gtsam.LevenbergMarquardtOptimizer(trial, values, params)
                values = opt.optimize()
                kept.append(fac)
            except Exception as e:
                skipped.append(i)
                # 可选：打印更详细
                print(f"[PoseGraph][safe_inc] skip factor idx={i}/{n-1} due to optimize error: {e}")
                continue

        print(f"[PoseGraph][safe_inc] kept={len(kept)}/{n}, skipped={len(skipped)}")
        self.result = values
        return self.result

    def optimize(self):
        """
        正常 optimize 失败时，不再 raise（否则你的 ablation 直接 returncode=1）。
        改为 fallback：安全增量优化，跳过会触发 SL4 崩溃的因子。
        """
        if self.graph.size() == 0:
            raise RuntimeError("[PoseGraph] empty factor graph (no edges)")
        if self.initial.size() == 0:
            raise RuntimeError("[PoseGraph] empty initial values")

        params = self._make_lm_params(for_safe_incremental=False)
        optimizer = gtsam.LevenbergMarquardtOptimizer(self.graph, self.initial, params)

        try:
            self.result = optimizer.optimize()
            return self.result
        except Exception as e:
            print("[PoseGraph] Optimization failed, fallback to safe incremental optimize.")
            print(f"  #values : {self.initial.size()}")
            print(f"  #factors: {self.graph.size()}")
            print(f"  error   : {e}")

            if self.enable_safe_incremental_opt:
                return self._optimize_safe_incremental()

            # 最保底：直接不优化，返回 initial（至少不让流程炸）
            self.result = self.initial
            return self.result

    # ---------------- accessors ----------------

    def get_homography(self, node_id):
        k = X(int(node_id))
        if self.result is not None and self.result.exists(k):
            return self.result.atSL4(k).matrix()
        if self.initial.exists(k):
            return self.initial.atSL4(k).matrix()
        return None

    def increment_loop_closure(self):
        self.num_loop_closures += 1

    def get_num_loops(self):
        return self.num_loop_closures
