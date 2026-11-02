"""
RL UTILS
Utilidades compartidas por los entrenadores de PPO y DDPG:
  - TrainLogger   : escribe las métricas de cada etapa en TensorBoard (una carpeta por etapa)
  - Timelapse     : graba episodios cortos durante el entrenamiento y los muestra en el cuadernillo
  - evaluate      : corre episodios de evaluación sin exploración
CREADO CON ASISTENCIA DE: Claude
"""

import math
import os
import time
from pathlib import Path

import numpy as np


# ======================================================================
# TENSORBOARD
# ======================================================================
class TrainLogger:
    """
    Envoltorio de SummaryWriter. Cada (algoritmo, etapa) escribe en su propia carpeta:

        runs/<algoritmo>/<NN_nombre-de-etapa>/

    Así TensorBoard muestra cada etapa como una curva aparte y se pueden comparar entre sí
    y contra el otro algoritmo. Para abrirlo:

        tensorboard --logdir runs
    """

    def __init__(self, algo, stage, root="runs", run_id=None):
        from torch.utils.tensorboard import SummaryWriter

        self.algo = algo
        self.stage = stage
        tag = f"{stage.index:02d}_{stage.name}"
        if run_id:
            tag = f"{tag}__{run_id}"
        self.logdir = Path(root) / algo / tag
        self.logdir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(log_dir=str(self.logdir))
        self._t0 = time.time()

        # Deja constancia de la configuración de la etapa en la pestaña TEXT de TensorBoard
        filas = "\n".join(f"|{k}|{v}|" for k, v in sorted(vars(stage).items()))
        self.writer.add_text("etapa/config", f"|parámetro|valor|\n|-|-|\n{filas}", 0)

    def scalar(self, tag, value, step):
        if value is None:
            return
        v = float(value)
        if math.isfinite(v):                       # evita que un NaN arruine la gráfica
            self.writer.add_scalar(tag, v, step)

    def log_episode(self, ep, step):
        """Registra el diccionario info['episode'] que devuelve el entorno al terminar."""
        self.scalar("charts/episodic_return", ep["r"], step)
        self.scalar("charts/episodic_length", ep["l"], step)
        self.scalar("charts/mean_abs_pitch_deg", ep["mean_abs_pitch_deg"], step)
        self.scalar("charts/mean_speed_err", ep["mean_speed_err"], step)
        self.scalar("charts/mean_yaw_err", ep["mean_yaw_err"], step)
        self.scalar("charts/final_dist", ep["final_dist"], step)
        self.scalar("charts/caidas", 1.0 if ep["terminated"] else 0.0, step)
        for k, v in ep["terms"].items():           # aporte medio de cada término de la recompensa
            self.scalar(f"recompensa/{k}", v, step)

    def log_losses(self, d, step):
        for k, v in d.items():
            self.scalar(f"losses/{k}", v, step)

    def log_sps(self, step):
        dt = max(time.time() - self._t0, 1e-9)
        self.scalar("charts/SPS", step / dt, step)

    def log_frames(self, frames, step, tag="timelapse"):
        """Guarda una tira de imágenes en TensorBoard (pestaña IMAGES)."""
        if not frames:
            return
        import numpy as _np
        sel = frames[:: max(1, len(frames) // 6)][:6]
        arr = _np.stack(sel).transpose(0, 3, 1, 2)      # (N, C, H, W)
        self.writer.add_images(tag, arr, step)

    def close(self):
        self.writer.flush()
        self.writer.close()


# ======================================================================
# TIMELAPSE
# ======================================================================
class Timelapse:
    """
    Graba episodios cortos con la política actual y los muestra como GIF en el cuadernillo.

    Usa un entorno APARTE con render_mode="rgb_array", para no perturbar el entrenamiento.
    El GIF se submuestrea (frame_skip) para que un episodio de varios segundos se vea como
    un timelapse de unas pocas décimas.
    """

    def __init__(self, env_fn, every_steps=50_000, max_frames=120, frame_skip=8,
                 fps=20, outdir="timelapses", show=True, scale=1):
        self.env_fn = env_fn
        self.every_steps = every_steps
        self.max_frames = max_frames
        self.frame_skip = frame_skip
        self.fps = fps
        self.outdir = Path(outdir)
        self.outdir.mkdir(parents=True, exist_ok=True)
        self.show = show
        self.scale = scale
        self._env = None
        self._next = 0

    def _get_env(self):
        if self._env is None:
            self._env = self.env_fn()
        return self._env

    def due(self, step):
        return step >= self._next

    def record(self, policy_fn, stage, step, algo="", command=None, force=False):
        """
        policy_fn : función obs -> acción determinista (sin ruido de exploración)
        command   : (v, w) para fijar un comando durante la grabación, o None
        Devuelve la lista de frames (puede ir vacía si imageio no está instalado).
        """
        if not force and not self.due(step):
            return []
        self._next = step + self.every_steps

        env = self._get_env()
        env.set_stage(stage)
        if command is not None:
            env.set_command(*command)

        obs, _ = env.reset(seed=int(step) % (2 ** 31))
        frames, i, done = [], 0, False
        ret = 0.0
        while not done and len(frames) < self.max_frames:
            action = policy_fn(obs)
            obs, r, terminated, truncated, _ = env.step(action)
            ret += r
            done = terminated or truncated
            if i % self.frame_skip == 0:
                frames.append(env.render())
            i += 1

        path = self.outdir / f"{algo}_etapa{stage.index}_{step:08d}.gif"
        self._save_and_show(frames, path, extra=f"{algo} · etapa {stage.index} · paso {step:,} · "
                                                f"{i} pasos simulados · retorno {ret:.1f}")
        return frames

    def _save_and_show(self, frames, path, extra=""):
        if not frames:
            return
        try:
            import importlib
            imageio = importlib.import_module("imageio.v2")
        except Exception:
            print("  (instala imageio para guardar los timelapses: pip install imageio)")
            return
        imgs = [np.asarray(f) for f in frames]
        if self.scale != 1:
            imgs = [np.repeat(np.repeat(f, self.scale, axis=0), self.scale, axis=1) for f in imgs]
        imageio.mimsave(path, imgs, duration=1.0 / self.fps, loop=0)
        if self.show:
            try:
                from IPython.display import Image, display, HTML
                display(HTML(f"<div style='font:12px monospace;color:#666'>{extra}</div>"))
                display(Image(filename=str(path)))
            except Exception:
                print(f"  timelapse guardado en {path}")

    def close(self):
        if self._env is not None:
            self._env.close()
            self._env = None


# ======================================================================
# EVALUACIÓN
# ======================================================================
def evaluate(env, policy_fn, episodes=5, stage=None, command=None, seed=0, render=False):
    """
    Corre episodios de evaluación con una política determinista y devuelve un diccionario
    con los promedios. No entrena ni modifica el normalizador de observaciones.
    """
    if stage is not None:
        env.set_stage(stage)
    if command is not None:
        env.set_command(*command)

    acc = {"retorno": [], "duracion": [], "pitch_medio_deg": [], "err_vel": [],
           "err_giro": [], "dist_final": [], "cayo": []}
    for ep in range(episodes):
        obs, _ = env.reset(seed=seed + ep)
        done = False
        while not done:
            obs, _, terminated, truncated, info = env.step(policy_fn(obs))
            if render:
                env.render()
            done = terminated or truncated
        e = info["episode"]
        acc["retorno"].append(e["r"])
        acc["duracion"].append(e["l"])
        acc["pitch_medio_deg"].append(e["mean_abs_pitch_deg"])
        acc["err_vel"].append(e["mean_speed_err"])
        acc["err_giro"].append(e["mean_yaw_err"])
        acc["dist_final"].append(e["final_dist"])
        acc["cayo"].append(float(e["terminated"]))
    return {k: float(np.mean(v)) for k, v in acc.items()}


def imprimir_eval(nombre, res):
    print(f"  {nombre:<28} retorno={res['retorno']:9.1f}  duración={res['duracion']:7.1f}  "
          f"pitch={res['pitch_medio_deg']:5.2f}°  err_v={res['err_vel']:.3f}  "
          f"err_w={res['err_giro']:.3f}  caídas={res['cayo']*100:3.0f}%")