"""E2 environment adapters — one interface, three worlds (E2_ENV_DECISION.md).

Every environment renders to a token stream the MorphSSM consumes. The Env
interface below is the whole contract; the AGENT (st6_reward.py) owns the frame
codec, the reward token, and the MPC preference so those stay uniform across
envs. Adapters:

    ForagingEnv  — wraps the bespoke foraging World (dense-ish reward, control)
    MinAtarEnv   — MinAtar via the RAW Environment.act API (constant 6 actions
                   across games; game-cycle biomes; verified pip `minatar`)
    (BabyAIEnv   — deferred: sparse episodic reward joins after the 2x2 screen)

Contract (what the agent/baselines rely on):
    env.n_actions  : int (constant across biomes — K for MPC)
    env.n_obs_sym  : int (number of distinct observation cell-symbols)
    env.obs_len    : int (cells per observation)
    env.biome      : int (non-stationarity index, for telemetry/retention)
    env.reset()               -> obs (LongTensor of obs_len cell-symbols in [0,n_obs_sym))
    env.step(a:int)           -> (reward:float, obs, done:bool)
    env.predict_true(a)       -> obs  (OPTIONAL oracle next-obs; foraging only)
"""
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import st5_world as FW                                       # noqa: E402


class ForagingEnv:
    """The bespoke foraging World as an Env. Obs cell-symbols are the tile
    types {empty=0, wall=1, food=2} in the CENTER-FIRST order (so cell 0 is the
    tile the agent stands on — the E1 codec, preserved). Reward = 1.0 on eating.
    """

    name = "foraging"
    n_actions = 5
    n_obs_sym = 3                                            # empty/wall/food
    obs_len = 9
    CENTER_FIRST = FW.CENTER_FIRST
    FOOD_SYM = FW.FOOD                                       # =2, the reward-tile symbol

    def __init__(self, world_seed=0, biome_steps=20_000):
        self.world = FW.World(world_seed, biome_steps=biome_steps)

    @property
    def biome(self):
        return self.world.biome

    def _sym(self, obs9):
        return torch.tensor([int(obs9[i]) for i in self.CENTER_FIRST],
                            dtype=torch.long)

    def reset(self):
        return self._sym(self.world.obs())

    def step(self, a):
        r, o = self.world.step(a)
        return r, self._sym(o), False                       # never-terminating

    def predict_true(self, a):
        return self._sym(self.world.predict_true(a))


class MinAtarEnv:
    """MinAtar via the RAW Environment.act (constant 6-action interface across
    all games — the gym -v1 wrapper's variable per-game sets would break a
    fixed-K MPC head; E2_ENV_DECISION verified). Obs cell-symbol = index of the
    active channel (0=empty, 1..C), priority-resolved for the rare multi-active
    cell. Games cycle as biomes (rotated, revisits). Dense game-score reward.
    """

    name = "minatar"
    GAMES = ("breakout", "freeway", "asterix", "seaquest", "space_invaders")
    n_actions = 6                                            # raw act accepts 0..5
    obs_len = 100                                            # 10x10 board
    # shared codebook across games: max channels = 10 (seaquest) -> symbols 0..10
    n_obs_sym = 11

    def __init__(self, world_seed=0, biome_steps=20_000, n_cycles=7,
                 first_game=0):
        from minatar import Environment
        base = tuple(range(len(self.GAMES)))
        order = []
        for c in range(n_cycles):
            k = (c + first_game) % len(base)
            order.extend(base[k:] + base[:k])
        self.game_order = order
        self.biome_steps = biome_steps
        self._Environment = Environment
        self._seed = world_seed
        self._t = 0
        self._bi = 0
        self._enter(0)

    def _enter(self, bi):
        self._bi = bi
        self._game = self.game_order[bi % len(self.game_order)]
        # sticky_action_prob=0 for a clean deterministic control screen (the
        # 0.1 default injects action noise that confounds a K-forward MPC head;
        # standard-benchmark comparisons can re-enable it later).
        self.env = self._Environment(self.GAMES[self._game],
                                     sticky_action_prob=0.0)
        self.env.seed(self._seed + bi)                      # reproducible
        self.env.reset()

    @property
    def biome(self):
        return self._game                                   # the game index

    def _sym(self, state):
        # state: (10,10,C) binary -> (100,) cell channel-index, priority = later
        # channel wins (object channels are ordered; gauges/trail are last)
        cell = torch.zeros(10, 10, dtype=torch.long)
        st = torch.as_tensor(state, dtype=torch.long)
        C = st.shape[2]
        for ch in range(C):
            cell = torch.where(st[:, :, ch] > 0,
                               torch.full_like(cell, ch + 1), cell)
        return cell.reshape(-1)

    def reset(self):
        return self._sym(self.env.state())

    def step(self, a):
        self._t += 1
        if self._t % self.biome_steps == 0:                 # biome (game) switch
            self._enter(self._bi + 1)
            return 0.0, self._sym(self.env.state()), False
        r, done = self.env.act(int(a))                      # raw: accepts 0..5
        if done:
            self.env.reset()                                # continuing stream
        return float(r), self._sym(self.env.state()), bool(done)


ENVS = {"foraging": ForagingEnv, "minatar": MinAtarEnv}


def make_env(name, world_seed=0, biome_steps=20_000):
    return ENVS[name](world_seed=world_seed, biome_steps=biome_steps)


# --------------------------------------------------------------------------- 
# E5 life support: bit-exact env serialization (E5 plan / Plan-agent design).
# get_state() -> opaque pickle bytes; set_state() restores in place. Both
# wrapped worlds verified bit-exact under pickle roundtrip (FW.World holds a
# torch.Generator — picklable, state-preserving on torch>=2.x; minatar's
# np.random.RandomState pickles natively). Bytes live inside the torch ckpt
# blob so torch.load never needs minatar importable at inspection time.

import pickle


def _env_get_state(self) -> bytes:
    return pickle.dumps(self.__dict__)


def _env_set_state(self, blob: bytes) -> None:
    self.__dict__.update(pickle.loads(blob))


ForagingEnv.get_state = _env_get_state
ForagingEnv.set_state = _env_set_state
MinAtarEnv.get_state = _env_get_state
MinAtarEnv.set_state = _env_set_state
