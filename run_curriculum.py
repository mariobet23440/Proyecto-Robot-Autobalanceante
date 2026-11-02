"""
RUN CURRICULUM
CREADO CON ASISTENCIA DE: Claude
Orquesta el entrenamiento por etapas de PPO o DDPG. Cada etapa hereda la red (y el normalizador,
y en DDPG el buffer) de la anterior. Después de cada etapa se evalúa el agente en un conjunto fijo
de situaciones, y se registran las métricas en TensorBoard en runs/<algoritmo>/<etapa>/.

Uso desde el cuadernillo:
    from run_curriculum import run_curriculum
    resumen = run_curriculum("ppo", etapas=[1, 2])
"""

import csv
from pathlib import Path

import numpy as np

from balance_bot_env import BalanceBotEnv
from curriculum import STAGES, get_stage
from rl_utils import TrainLogger, Timelapse, evaluate, imprimir_eval

XML_PATH = "autobalance.xml"

# Situaciones fijas para comparar etapas entre sí (misma semilla y mismos comandos)
EVAL_CASES = {
    "quieto (cmd 0)":        {"command": (0.0, 0.0)},
    "avanza 0.2 m/s":        {"command": (0.2, 0.0)},
    "gira 1.0 rad/s":        {"command": (0.0, 1.0)},
}


def make_env(stage, render_mode=None, max_steps=2000):
    return BalanceBotEnv(mjcf_path=XML_PATH, stage=stage, render_mode=render_mode,
                         max_steps=max_steps)


def build_agent(algo, env):
    """Crea el agente con la configuración del archivo correspondiente."""
    if algo == "ppo":
        from ppo import PPOAgent, PPOConfig
        return PPOAgent(env.observation_space.shape[0], env.action_space.shape[0], PPOConfig())
    if algo == "ddpg":
        from ddpg import DDPGAgent, DDPGConfig
        return DDPGAgent(env.observation_space.shape[0], env.action_space.shape[0],
                         max_action=1.0, config=DDPGConfig())
    raise ValueError("algo debe ser 'ppo' o 'ddpg'")


def evaluar_todo(algo, agent, env, stage, logger=None, paso=0):
    """Evalúa la política determinista en los casos fijos y devuelve un dict de resultados."""
    if algo == "ppo":
        pol = agent.act_deterministic
    else:
        pol = agent.act_deterministic
    resultados = {}
    for nombre, caso in EVAL_CASES.items():
        r = evaluate(env, pol, episodes=3, stage=stage, command=caso["command"], seed=1000)
        resultados[nombre] = r
        imprimir_eval(nombre, r)
        if logger is not None:
            logger.scalar(f"eval/{nombre.split()[0]}_retorno", r["retorno"], paso)
            logger.scalar(f"eval/{nombre.split()[0]}_caidas", r["cayo"], paso)
            logger.scalar(f"eval/{nombre.split()[0]}_err_vel", r["err_vel"], paso)
    return resultados


def run_curriculum(algo, etapas=None, total_timesteps=None, timelapse_every=50_000,
                   timelapse=True, outdir="resultados", evaluar=True):
    """
    Entrena el algoritmo `algo` a través de las etapas indicadas, en orden.

    etapas           : lista de números de etapa (p. ej. [1, 2, 3]). None = todas.
    total_timesteps  : pasos por etapa. None = el presupuesto de cada etapa (stage.max_timesteps).
    timelapse_every  : cada cuántos pasos se graba un episodio para el timelapse.
    Devuelve la lista de resúmenes (uno por etapa) y guarda un CSV en outdir/.
    """
    from train_helpers import train_one_stage  # importación diferida: depende de torch

    etapas = etapas or [s.index for s in STAGES]
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    env = make_env(get_stage(etapas[0]))
    agent = build_agent(algo, env)
    buffer = None
    resumenes = []

    for idx in etapas:
        stage = get_stage(idx)
        print(f"\n===== {algo.upper()} · Etapa {stage.index}: {stage.title} =====")
        logger = TrainLogger(algo, stage)
        tl = None
        if timelapse:
            tl_env = make_env(stage, render_mode="rgb_array")
            tl = Timelapse(lambda e=tl_env: e, every_steps=timelapse_every,
                           outdir=outdir / "timelapses", show=True)

        resumen, buffer = train_one_stage(algo, env, agent, stage, total_timesteps,
                                          logger, tl, buffer)

        if evaluar:
            print(f"  Evaluación al terminar la etapa {stage.index}:")
            res = evaluar_todo(algo, agent, env, stage, logger, resumen["pasos"])
            resumen["eval"] = {k: v["retorno"] for k, v in res.items()}
            resumen["eval_caidas"] = {k: v["cayo"] for k, v in res.items()}

        logger.close()
        resumenes.append(resumen)
        _guardar_csv(resumenes, outdir / f"resumen_{algo}.csv")

    env.close()
    print(f"\nListo. Para ver las gráficas:  tensorboard --logdir runs")
    return resumenes


def _guardar_csv(resumenes, path):
    if not resumenes:
        return
    filas = []
    for r in resumenes:
        fila = {k: v for k, v in r.items() if not isinstance(v, dict)}
        for k, v in r.get("eval", {}).items():
            fila[f"eval_retorno[{k}]"] = v
        for k, v in r.get("eval_caidas", {}).items():
            fila[f"eval_caidas[{k}]"] = v
        filas.append(fila)
    campos = sorted({k for f in filas for k in f}, key=lambda k: (k not in ("algoritmo", "etapa"), k))
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=campos)
        w.writeheader()
        w.writerows(filas)