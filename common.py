"""
COMMON
Utilidades compartidas por PPO, DDPG y el entrenamiento por currículo.
CREADO CON ASISTENCIA DE: Claude
Solo depende de NumPy, así que se puede probar sin PyTorch ni MuJoCo.
"""

import numpy as np


class RunningMeanStd:
    """
    Media y desviación estándar corrientes (algoritmo de Welford por lotes).
    Se usa para normalizar las observaciones: el pitch vale ~0.1 rad y el giroscopio ~10 rad/s,
    y sin normalizar las redes aprenden mucho más lento.
    """

    def __init__(self, shape, epsilon=1e-4):
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = epsilon

    def update(self, x):
        x = np.asarray(x, dtype=np.float64)
        if x.ndim == 1:
            x = x[None, :]
        batch_mean = x.mean(axis=0)
        batch_var = x.var(axis=0)
        batch_count = x.shape[0]

        delta = batch_mean - self.mean
        total = self.count + batch_count
        new_mean = self.mean + delta * batch_count / total
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + delta ** 2 * self.count * batch_count / total
        self.mean = new_mean
        self.var = m2 / total
        self.count = total

    def normalize(self, x, clip=10.0):
        out = (np.asarray(x, dtype=np.float64) - self.mean) / np.sqrt(self.var + 1e-8)
        return np.clip(out, -clip, clip).astype(np.float32)

    def state_dict(self):
        return {"mean": self.mean, "var": self.var, "count": self.count}

    def load_state_dict(self, d):
        self.mean, self.var, self.count = d["mean"], d["var"], d["count"]


def compute_gae(rewards, values, dones, last_value, gamma=0.99, lam=0.95, truncated=None, next_values=None):
    """
    Estimación Generalizada de la Ventaja (GAE-lambda).

    rewards, values, dones : arreglos de longitud T (dones = 1 si el estado SIGUIENTE es terminal real)
    last_value             : V(s_T) para continuar (bootstrap) al final del lote
    truncated              : arreglo T con 1 si el episodio terminó por límite de tiempo (no por caída)
    next_values            : arreglo T con V(s_{t+1}) para los pasos truncados (bootstrap correcto)

    Retorna (advantages, returns) con returns = advantages + values (sin normalizar).
    """
    T = len(rewards)
    rewards = np.asarray(rewards, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    dones = np.asarray(dones, dtype=np.float64)
    if truncated is None:
        truncated = np.zeros(T)
    truncated = np.asarray(truncated, dtype=np.float64)
    if next_values is None:
        next_values = np.zeros(T)
    next_values = np.asarray(next_values, dtype=np.float64)

    advantages = np.zeros(T)
    gae = 0.0
    for t in reversed(range(T)):
        if t == T - 1:
            v_next = last_value
        else:
            v_next = values[t + 1]
        # Si el episodio fue truncado por tiempo, se hace bootstrap con V(s_{t+1}) real
        # y se corta la cadena de GAE (el siguiente paso del lote pertenece a otro episodio).
        if truncated[t]:
            v_next = next_values[t]
            cut = 0.0
        else:
            cut = 1.0 - dones[t]
        delta = rewards[t] + gamma * v_next * (1.0 - dones[t]) - values[t]
        gae = delta + gamma * lam * cut * gae
        advantages[t] = gae
    returns = advantages + values
    return advantages, returns


class EpisodeStats:
    """Acumula estadísticas de episodios recientes (ventana deslizante) para TensorBoard y promoción."""

    def __init__(self, window=30):
        self.window = window
        self.lengths = []
        self.returns = []

    def add(self, length, ret):
        self.lengths.append(length)
        self.returns.append(ret)
        if len(self.lengths) > self.window:
            self.lengths.pop(0)
            self.returns.pop(0)

    def mean_length(self):
        return float(np.mean(self.lengths)) if self.lengths else 0.0

    def mean_return(self):
        return float(np.mean(self.returns)) if self.returns else 0.0

    def __len__(self):
        return len(self.lengths)