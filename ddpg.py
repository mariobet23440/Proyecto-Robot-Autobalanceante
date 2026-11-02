'''
Deep Deterministic Policy Gradient (DDPG)
CREADO POR: Mario Betancourt
SOBRE USO DE I.A. : La primera iteración de este script fue generado por Gemini, sin documentación
                    Los comentarios fueron hechos a teclado por Mario Betancourt a modo de entender
                    cómo funciona esta implementación.

VERSIÓN 2 (con asistencia de Claude): se conservan las clases y comentarios originales. Cambios:
  - CORRECCIÓN: el bucle de entrenamiento original no existía en el archivo (solo había train()).
    Se agrega train_stage(), que entrena UNA etapa del currículo con TensorBoard y timelapse.
  - CORRECCIÓN: la pérdida del crítico usaba `done` pero NO distinguía truncamiento por tiempo.
    Ahora, solo una caída real (terminated) corta el bootstrap; un episodio truncado sigue
    usando Q' del estado siguiente.
  - Se agrega ruido Ornstein-Uhlenbeck opcional (el de la versión original, gaussiano, sigue
    disponible con noise_type="gauss").
  - Se agrega normalización de observaciones y gradient clipping en el crítico.
  - Se agregan timelapse y registro de métricas en TensorBoard.
'''

# LIBRERÍAS
import random
from collections import deque   # Double-ended Queue
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from common import RunningMeanStd, EpisodeStats
from rl_utils import TrainLogger


# --- 1. REPLAY BUFFER ---
class ReplayBuffer:
    '''
    Clase para Replay Buffer. Buffer utilizado en DDPG para almacenar secuencias (S_t, A_t, R_{t+1}, S_{t+1}, done)
    y muestrearlas de forma aleatoria.
    '''

    def __init__(self, capacity=100000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        """
        Insertar secuencia (S_t, A_t, R_{t+1}, S_{t+1}, done) en Replay Buffer. El elemento done es un
        valor booleano que indica si el estado S_{t+1} es un estado terminal (caída real).
        (v2) Un episodio truncado por tiempo NO es terminal: done debe ser False ahí.
        """
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        """
        Muestrear una secuencia (S_t, A_t, R_{t+1}, S_{t+1}) aleatoria del Replay Buffer. Retorna
        una tupla de cinco números np.float32.
        """
        state, action, reward, next_state, done = zip(*random.sample(self.buffer, batch_size))
        return (
            np.array(state, dtype=np.float32),
            np.array(action, dtype=np.float32),
            np.array(reward, dtype=np.float32),
            np.array(next_state, dtype=np.float32),
            np.array(done, dtype=np.float32)
        )

    def __len__(self):
        """
        Retorna la longitud del buffer
        """
        return len(self.buffer)


# --- 2. RED ACTOR (política) ---
class Actor(nn.Module):
    """
    Clase que representa la plantilla de la NN del Actor en DDPG.
    """
    def __init__(self, state_dim, action_dim, max_action, hidden=32):
        """
        Se usa un Perceptrón Multicapa (MLP) con una capa de entrada de state_dim nodos,
        dos capas ocultas de 32 nodos y una capa de salida de dimensión ajustable.
        Las capas ocultas utilizan funciones de activación ReLU y la capa de salida
        utiliza una función tangente hiperbólica (para saturar las acciones entre -1 y 1).
        La salida estima la acción determinista de la política en un estado particular.
        """

        super(Actor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden),   # Capa de entrada y capa oculta 1
            nn.ReLU(),
            nn.Linear(hidden, hidden),      # Capa oculta 1 y capa oculta 2
            nn.ReLU(),
            nn.Linear(hidden, action_dim),  # Capa oculta 2 y capa de salida
            nn.Tanh()                       # Escala las acciones al rango [-1, 1]
        )

        # Valor absoluto que puede tomar una acción
        self.max_action = max_action

    def forward(self, state):
        """
        Propagar datos hacia adelante en la NN.
        """
        return self.net(state) * self.max_action


# --- 3. Red Critic (Función Q(s, a)) ---
class Critic(nn.Module):
    """
    Clase que representa la plantilla de la NN del Critic en DDPG.
    """

    def __init__(self, state_dim, action_dim, hidden=32):
        """
        Se usa un Perceptrón Multicapa (MLP) con una capa de entrada de state_dim + action_dim
        nodos, dos capas ocultas de 32 nodos y una capa de salida de dimensión ajustable.
        Las capas ocultas utilizan funciones de activación ReLU. La salida aproxima la función
        de valor de acción.
        """
        super(Critic, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden),  # Capa de entrada y capa oculta 1
            nn.ReLU(),
            nn.Linear(hidden, hidden),                  # Capa oculta 1 y capa oculta 2
            nn.ReLU(),
            nn.Linear(hidden, 1)                        # Capa oculta 2 y capa de salida
        )

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        return self.net(x)


# --- Ruido de exploración ---
class OUNoise:
    """
    Proceso de Ornstein-Uhlenbeck: ruido con memoria (correlacionado en el tiempo). Es el ruido
    que usó el DDPG original. Con control de robots, el ruido gaussiano independiente en cada paso
    se cancela casi por completo al integrarse, y el OU da movimientos más suaves.
    """
    def __init__(self, dim, theta=0.15, sigma=0.2, dt=1.0):
        self.theta, self.sigma, self.dt = theta, sigma, dt
        self.x = np.zeros(dim)

    def reset(self):
        self.x = np.zeros_like(self.x)

    def sample(self):
        self.x = self.x + self.theta * (0.0 - self.x) * self.dt + \
                 self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.x.shape)
        return self.x


@dataclass
class DDPGConfig:
    actor_lr: float = 1e-4         # tasa de aprendizaje del actor
    critic_lr: float = 1e-3        # tasa de aprendizaje del crítico
    gamma: float = 0.99            # factor de descuento
    tau: float = 0.005             # tasa de actualización suave (Polyak)
    buffer_size: int = 100_000     # capacidad del replay buffer
    batch_size: int = 128
    learning_starts: int = 5_000   # pasos de experiencia aleatoria antes de empezar a entrenar
    updates_per_step: int = 1      # actualizaciones de gradiente por paso de entorno
    noise_type: str = "ou"         # "ou" o "gauss"
    noise_sigma: float = 0.2
    hidden: int = 32
    norm_obs: bool = True
    seed: int = 42


# --- 4. Agente DDPG ---
class DDPGAgent:
    """
    Clase que representa el agente DDPG.
    """

    def __init__(self, state_dim, action_dim, max_action=1.0, config=None, device=None):
        self.cfg = config or DDPGConfig()
        cfg = self.cfg
        # Seleccionar el dispositivo para procesar modelos
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_action = max_action

        # Instanciar online networks y target networks de ACTOR
        self.actor = Actor(state_dim, action_dim, max_action, cfg.hidden).to(self.device)
        self.actor_target = Actor(state_dim, action_dim, max_action, cfg.hidden).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=cfg.actor_lr)

        # Instanciar online networks y target networks de CRITIC
        self.critic = Critic(state_dim, action_dim, cfg.hidden).to(self.device)
        self.critic_target = Critic(state_dim, action_dim, cfg.hidden).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=cfg.critic_lr)

        self.gamma = cfg.gamma          # Factor de descuento
        self.tau = cfg.tau              # Tasa de actualización suave (Polyak averaging)

        self.obs_rms = RunningMeanStd(state_dim) if cfg.norm_obs else None
        self.ou = OUNoise(action_dim, sigma=cfg.noise_sigma)

    # ------------------------------------------------------------------
    def normaliza(self, obs, actualizar=False):
        if self.obs_rms is None:
            return np.asarray(obs, dtype=np.float32)
        if actualizar:
            self.obs_rms.update(obs)
        return self.obs_rms.normalize(obs)

    def select_action(self, state, noise=None, actualizar_rms=False):
        """
        Selecciona la acción en un estado determinado. Si noise es None se usa cfg.noise_sigma
        (noise=0 desactiva la exploración).
        """
        obs_n = self.normaliza(state, actualizar=actualizar_rms)

        # Convertir el estado (arreglo de NumPy) en un tensor de PyTorch
        t = torch.as_tensor(obs_n, dtype=torch.float32, device=self.device).unsqueeze(0)

        # Inferencia en modo evaluación
        self.actor.eval()                                               # Pone la red en modo evaluación
        with torch.no_grad():                                           # Desactiva backpropagation
            action = self.actor(t).cpu().numpy().flatten()              # Calcular acción
        self.actor.train()                                              # Restaura la red en estado de entrenamiento

        # Agregar ruido para la exploración
        sigma = self.cfg.noise_sigma if noise is None else noise
        if sigma != 0:
            if self.cfg.noise_type == "ou":
                self.ou.sigma = sigma
                action = action + self.ou.sample()
            else:
                action = action + np.random.normal(0, sigma, size=action.shape)

        # Retornar acción limitada entre valores máximo y mínimo
        return np.clip(action, -self.max_action, self.max_action), obs_n

    def act_deterministic(self, state):
        obs_n = self.normaliza(state, actualizar=False)
        t = torch.as_tensor(obs_n, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            return self.actor(t).cpu().numpy().flatten()

    def train(self, replay_buffer, batch_size=None):
        """
        Entrenar agente DDPG
        """
        batch_size = batch_size or self.cfg.batch_size

        # No empezar a entrenar hasta que el Replay Buffer se haya llenado
        if len(replay_buffer) < batch_size:
            return {}

        # -- 1. Muestrear un batch de secuencias (S_t, A_t, R_{t+1}, S_{t+1}, done) --
        states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)

        # 1.1. Convertir batches a tensores de PyTorch
        state = torch.as_tensor(states, device=self.device)
        action = torch.as_tensor(actions, device=self.device)
        reward = torch.as_tensor(rewards, device=self.device).unsqueeze(1)
        next_state = torch.as_tensor(next_states, device=self.device)
        done = torch.as_tensor(dones, device=self.device).unsqueeze(1)

        # 2. -- Actualizar el Crítico --
        # 2.1. Estimar Q(s,a) con target network del Critic
        with torch.no_grad():                                               # Desactivar backpropagation (cambio de parámetros)
            next_action = self.actor_target(next_state)                     # Calcular acción siguiente con la target network del Actor
            target_Q = self.critic_target(next_state, next_action)          # Calcular Q(s,a) con target network del Critic
            # done = 1 SOLO si hubo caída real. Un truncamiento por tiempo sigue bootstrapeando.
            target_Q = reward + (1 - done) * self.gamma * target_Q          # Actualizar Q(s,a) con la recompensa muestreada
            # ¡Esto es de Q-Learning! ⤴

        # 2.2 Estimar Q(s,a) con online network del Critic
        current_Q = self.critic(state, action)                              # Calcular Q(s,a) con online network del Critic

        # 2.3. Calcular pérdida MSE con las dos NN
        critic_loss = nn.functional.mse_loss(current_Q, target_Q)           # Calcular la pérdida Mean Square Error (MSE)

        # 2.4. Reestablecer gradientes, backpropagation y dar un paso
        self.critic_optimizer.zero_grad()                                   # Pone en cero los gradientes acumulados
        critic_loss.backward()                                              # Ejecuta backpropagation
        nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.critic_optimizer.step()                                        # Realiza un paso de optimización

        # 3. -- Actualizar el Actor --
        actor_loss = -self.critic(state, self.actor(state)).mean()          # Gradiente de la política determinista

        self.actor_optimizer.zero_grad()                                    # Reiniciar gradientes
        actor_loss.backward()                                               # Hacer backpropagation
        self.actor_optimizer.step()                                         # Dar un paso de optimización de parámetros

        # 4. -- Actualización suave de las redes objetivo --
        with torch.no_grad():
            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        # tau es la tasa de actualización suave.
        return {"critic_loss": critic_loss.item(), "actor_loss": actor_loss.item(),
                "q_mean": current_Q.mean().item()}

    # ------------------------------------------------------------------
    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "obs_rms": self.obs_rms.state_dict() if self.obs_rms else None,
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
        }, path)

    def load(self, path):
        ck = torch.load(path, map_location=self.device, weights_only=False)
        self.actor.load_state_dict(ck["actor"])
        self.critic.load_state_dict(ck["critic"])
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())
        if ck.get("obs_rms") and self.obs_rms is not None:
            self.obs_rms.load_state_dict(ck["obs_rms"])
        return self


# ======================================================================
# ENTRENAMIENTO DE UNA ETAPA DEL CURRÍCULO
# ======================================================================
def train_stage(env, agent, stage, total_timesteps=None, logger=None, timelapse=None,
                buffer=None, ckpt_dir="checkpoints/ddpg", log_every=5_000, seed=None):
    """
    Entrena al agente DDPG durante UNA etapa del currículo.

    El replay buffer se pasa de una etapa a otra (buffer=...). Así el agente puede seguir
    aprendiendo de experiencia de la etapa anterior. Si esto causa problemas (la experiencia
    vieja tiene otra dinámica), pasa buffer=None para empezar uno nuevo en cada etapa.
    Devuelve (resumen, buffer).
    """
    cfg = agent.cfg
    env.set_stage(stage)
    total_timesteps = int(total_timesteps or stage.max_timesteps)
    cerrar_logger = logger is None
    if logger is None:
        logger = TrainLogger("ddpg", stage)
    if buffer is None:
        buffer = ReplayBuffer(cfg.buffer_size)

    rng_seed = cfg.seed if seed is None else seed
    np.random.seed(rng_seed)
    random.seed(rng_seed)
    obs, _ = env.reset(seed=rng_seed)
    agent.ou.reset()
    stats = EpisodeStats(window=30)
    paso = 0
    n_episodios = 0
    umbral_promocion = stage.promote_length_frac * env.max_steps
    promovido_por_desempeño = False
    ultimo_log = 0
    ultimas_perdidas = {}

    while paso < total_timesteps:
        # ======== Paso 1: Acción (aleatoria al principio, luego con la política + ruido) ========
        if len(buffer) < cfg.learning_starts and agent.cfg.learning_starts > 0:
            action = env.action_space.sample()
            obs_n = agent.normaliza(obs, actualizar=True)
        else:
            action, obs_n = agent.select_action(obs, actualizar_rms=True)

        next_obs, reward, terminated, truncated, info = env.step(action)

        # ======== Paso 2: Guardar la transición en el buffer ========
        # Se guarda el estado NORMALIZADO para que el entrenamiento y la acción vean la misma escala.
        next_obs_n = agent.normaliza(next_obs)
        buffer.push(obs_n, action, reward, next_obs_n, float(terminated))

        paso += 1
        obs = next_obs

        # ======== Paso 3: Actualizar redes ========
        if len(buffer) >= cfg.learning_starts:
            for _ in range(cfg.updates_per_step):
                m = agent.train(buffer)
                if m:
                    ultimas_perdidas = m

        if terminated or truncated:
            ep = info["episode"]
            stats.add(ep["l"], ep["r"])
            n_episodios += 1
            logger.log_episode(ep, paso)
            obs, _ = env.reset()
            agent.ou.reset()

        # ======== Registro cada log_every pasos ========
        if paso - ultimo_log >= log_every or paso >= total_timesteps:
            ultimo_log = paso
            if ultimas_perdidas:
                logger.log_losses(ultimas_perdidas, paso)
            logger.scalar("charts/buffer_size", len(buffer), paso)
            logger.log_sps(paso)
            print(f"  [ddpg etapa {stage.index}] paso {paso:>7,}/{total_timesteps:,}  "
                  f"retorno={stats.mean_return():8.1f}  duración={stats.mean_length():7.1f}  "
                  f"episodios={n_episodios}  buffer={len(buffer):,}")

            if timelapse is not None and timelapse.due(paso):
                frames = timelapse.record(agent.act_deterministic, stage, paso, algo="ddpg")
                logger.log_frames(frames, paso)

            if (paso >= stage.min_timesteps and len(stats) >= 10
                    and stats.mean_length() >= umbral_promocion):
                promovido_por_desempeño = True
                print(f"  [ddpg etapa {stage.index}] promoción temprana: duración media "
                      f"{stats.mean_length():.0f} >= {umbral_promocion:.0f}")
                break

    ckpt = Path(ckpt_dir) / f"ddpg_etapa{stage.index}_{stage.name}.pt"
    agent.save(ckpt)

    if timelapse is not None:
        frames = timelapse.record(agent.act_deterministic, stage, paso, algo="ddpg", force=True)
        logger.log_frames(frames, paso, tag="timelapse_final")

    resumen = {
        "algoritmo": "ddpg", "etapa": stage.index, "nombre": stage.name,
        "pasos": paso, "episodios": n_episodios,
        "retorno_medio": stats.mean_return(), "duracion_media": stats.mean_length(),
        "promovido_temprano": promovido_por_desempeño,
        "checkpoint": str(ckpt), "logdir": str(logger.logdir),
    }
    if cerrar_logger:
        logger.close()
    return resumen, buffer