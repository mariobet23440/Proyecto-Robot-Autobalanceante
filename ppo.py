'''
Proximal Policy Optimization (PPO)
CREADO POR: Mario Betancourt
SOBRE USO DE I.A. : La primera iteración de este script fue generado por Gemini, sin documentación
                    Los comentarios fueron hechos a teclado por Mario Betancourt a modo de entender
                    cómo funciona esta implementación.
FUENTES DE INFORMACIÓN: Se utilizó la documentación de OpenAI sobre PPO y el pseudocódigo simplificado
                        del video "Simply Explaining Proximal Policy Optimization (PPO) | Deep
                        Reinforcement Learning" de Jhonny Code como guías para programar.

VERSIÓN 2 (con asistencia de Claude): se conservan las clases y la documentación originales. Cambios:
  - CORRECCIÓN: el cálculo de rewards-to-go estaba roto (la variable discounted_reward se calculaba
    pero nunca se guardaba en rewards_to_go, así que el vector returns salía en ceros y la ventaja
    era simplemente -V(s)). Ahora se usa GAE-lambda (ver common.compute_gae).
  - CORRECCIÓN: se normalizaba returns ANTES de restarle values, mezclando dos escalas distintas.
    Ahora se normaliza la VENTAJA (que es lo que se normaliza en PPO) y el crítico se entrena
    contra returns sin normalizar.
  - CORRECCIÓN: el bootstrap al truncar un episodio por límite de tiempo (truncated) usa V(s') en
    lugar de cortar el retorno, que es lo correcto cuando el episodio no terminó por caída.
  - Se agregan: normalización de observaciones, minibatches mezclados, recorte de gradiente,
    annealing de la tasa de aprendizaje, inicialización ortogonal y métricas (KL, clipfrac).
  - Se agrega train_stage(): entrena UNA etapa del currículo, con TensorBoard y timelapse.
'''

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal

from common import RunningMeanStd, compute_gae, EpisodeStats
from rl_utils import TrainLogger

# ==========================================
# CONFIGURACIÓN INICIAL: REDES NEURONALES
# ==========================================

def _init_capa(capa, std=np.sqrt(2), bias=0.0):
    """
    Inicialización ortogonal. Es el truco estándar en las implementaciones de PPO: mantiene la
    varianza de las activaciones al propagar hacia adelante y acelera bastante el aprendizaje.
    """
    nn.init.orthogonal_(capa.weight, std)
    nn.init.constant_(capa.bias, bias)
    return capa


class ActorCritic(nn.Module):
    '''
    Clase para la red neuronal del actor y el critic. Se utiliza un perceptrón multicapas (MLP) con
    dos capas ocultas de 64 nodos y capas de entrada y salida de dimensiones ajustables. Las
    capas ocultas utilizan funciones de activación Tanh y la capa de salida usa una función
    de activación lineal. Se usa la misma arquitectura para las NN de ambos el Actor y el Critic.

    Nota (v2): se cambió ReLU por Tanh en las capas ocultas. Para problemas de control continuo
    con observaciones de pocas dimensiones, Tanh da políticas más suaves (es también lo que usan
    las implementaciones de referencia de PPO). La salida del actor sigue siendo lineal: es la
    MEDIA de la distribución, y las acciones se saturan en el entorno al rango [-1, 1].
    '''
    def __init__(self, state_dim, action_dim, hidden=64, init_log_std=-0.5):
        super(ActorCritic, self).__init__()

        # Paso 1: Inicializar la Red del Actor (Actor Network)
        # Esta red aproxima la política como una distribución de probabilidades pi(a | s)
        self.actor = nn.Sequential(
            _init_capa(nn.Linear(state_dim, hidden)),
            nn.Tanh(),
            _init_capa(nn.Linear(hidden, hidden)),
            nn.Tanh(),
            _init_capa(nn.Linear(hidden, action_dim), std=0.01),   # std pequeño = política casi neutra al inicio
        )

        # Crear un parámetro vector de dimensión action_dim relleno con init_log_std
        # Contiene el logaritmo de la desviación estándar para controlar la exploración.
        # (v2) Antes se inicializaba en 0, o sea std = 1. Para acciones en [-1, 1] eso es
        # enorme: casi toda acción muestreada se saturaba. Con -0.5 la std arranca en ~0.61.
        self.actor_log_std = nn.Parameter(torch.ones(1, action_dim) * init_log_std)

        # Paso 2: Inicializar la Red del Crítico (Critic Network)
        # Se inicializa de forma independiente otra red neuronal encargada de estimar el Valor del estado V(s).
        # El valor es el retorno total esperado que el agente recibirá a largo plazo partiendo de un estado.
        self.critic = nn.Sequential(
            _init_capa(nn.Linear(state_dim, hidden)),
            nn.Tanh(),
            _init_capa(nn.Linear(hidden, hidden)),
            nn.Tanh(),
            _init_capa(nn.Linear(hidden, 1), std=1.0),
        )

    def forward(self, state):
        '''Propagación hacia adelante'''
        value = self.critic(state)              # Aproximación de la función de valor de estado V(state)
        mu = self.actor(state)                  # Aproximación de la media de la política para state pi(a | state)
        std = torch.exp(self.actor_log_std)     # Desviación estándar
        dist = Normal(mu, std)                  # Política - distribución normal parametrizada por la acción
        return dist, value                      # Retornar política Y el valor de estado


@dataclass
class PPOConfig:
    """Hiperparámetros del entrenamiento de PPO."""
    lr: float = 3e-4              # tasa de aprendizaje inicial de Adam
    anneal_lr: bool = True        # bajar la tasa de aprendizaje linealmente a lo largo de la etapa
    gamma: float = 0.99           # factor de descuento
    gae_lambda: float = 0.95      # mezcla de GAE: 0 = error TD puro, 1 = Monte Carlo puro
    eps_clip: float = 0.2         # hiperparámetro de clipeo (epsilon)
    k_epochs: int = 10            # épocas de optimización por lote
    num_steps: int = 2048         # pasos por lote (2048 * 5 ms = ~10 s de simulación)
    minibatch_size: int = 64      # tamaño de minibatch
    vf_coef: float = 0.5          # peso de la pérdida del crítico
    ent_coef: float = 0.0         # peso del bono de entropía
    max_grad_norm: float = 0.5    # recorte de la norma del gradiente
    target_kl: float = 0.03       # si la divergencia KL supera esto, se corta la época (None = nunca)
    norm_obs: bool = True         # normalizar observaciones con media y varianza corrientes
    hidden: int = 64              # nodos por capa oculta
    seed: int = 42


class PPOAgent:
    '''Clase para un agente PPO'''
    def __init__(self, state_dim, action_dim, config=None, device=None):
        self.cfg = config or PPOConfig()
        cfg = self.cfg

        # HIPERPARÁMETROS (se conservan como atributos por compatibilidad con la versión 1)
        self.gamma = cfg.gamma                  # Factor de descuento
        self.eps_clip = cfg.eps_clip            # Hiperparámetro de clipeo
        self.k_epochs = cfg.k_epochs            # Número de épocas de entrenamiento
        self.batch_size = cfg.minibatch_size    # Tamaño de minibatch

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

        # REDES NEURONALES Y OPTIMIZADORES
        # (v2) PPO-Clip no necesita una copia "vieja" de la red: los log-probs viejos se guardan
        # en el momento de recolectar el lote, que es exactamente pi_{theta_old}. Por eso se
        # eliminó policy_old; una red menos que mantener sincronizada.
        self.policy = ActorCritic(state_dim, action_dim, cfg.hidden).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=cfg.lr, eps=1e-5)

        # Normalizador de observaciones (se arrastra de una etapa a la siguiente)
        self.obs_rms = RunningMeanStd(state_dim) if cfg.norm_obs else None

    # ------------------------------------------------------------------
    def normaliza(self, obs, actualizar=False):
        if self.obs_rms is None:
            return np.asarray(obs, dtype=np.float32)
        if actualizar:
            self.obs_rms.update(obs)
        return self.obs_rms.normalize(obs)

    def select_action(self, state, actualizar_rms=False):
        '''Seleccionar acción muestreando de la política (para recolectar experiencia)'''
        obs_n = self.normaliza(state, actualizar=actualizar_rms)
        t = torch.as_tensor(obs_n, dtype=torch.float32, device=self.device)
        with torch.no_grad():                                   # Desactivar backpropagation
            dist, value = self.policy(t)                        # Obtener distribución y valor
            action = dist.sample()                              # Muestrear una acción de la política
            action_logprob = dist.log_prob(action).sum(dim=-1)  # Probabilidades logarítmicas
        return (action.cpu().numpy(), float(action_logprob.item()),
                float(value.item()), obs_n)

    def act_deterministic(self, state):
        '''Acción determinista (la media de la política). Para evaluación y timelapse.'''
        obs_n = self.normaliza(state, actualizar=False)
        t = torch.as_tensor(obs_n, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            mu = self.policy.actor(t)
        return np.clip(mu.cpu().numpy(), -1.0, 1.0)

    def value_of(self, state_norm):
        t = torch.as_tensor(state_norm, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            return float(self.policy.critic(t).item())

    # ------------------------------------------------------------------
    def update(self, memory, lr=None):
        '''
        Actualizar la política y el critic usando los datos almacenados en la memoria.
        - memory : Diccionario con 'obs' (ya normalizadas), 'actions', 'logprobs', 'advantages'
                   y 'returns'. El cálculo de ventajas y retornos se hace fuera (ver train_stage),
                   porque necesita el valor del estado siguiente para el bootstrap.
        Devuelve un diccionario de métricas para TensorBoard.
        '''
        if lr is not None:
            for g in self.optimizer.param_groups:
                g["lr"] = lr

        dev = self.device
        states = torch.as_tensor(np.asarray(memory['obs']), dtype=torch.float32, device=dev)
        actions = torch.as_tensor(np.asarray(memory['actions']), dtype=torch.float32, device=dev)
        old_logprobs = torch.as_tensor(np.asarray(memory['logprobs']), dtype=torch.float32, device=dev)
        advantages = torch.as_tensor(np.asarray(memory['advantages']), dtype=torch.float32, device=dev)
        returns = torch.as_tensor(np.asarray(memory['returns']), dtype=torch.float32, device=dev)

        n = states.shape[0]
        idx = np.arange(n)
        métricas = {}
        rng = np.random.default_rng(self.cfg.seed)

        # Paso 8: Repetir la optimización (Ciclos de entrenamiento para N épocas)
        # Los pasos 6 y 7 se ejecutan continuamente durante K épocas para exprimir el lote de datos.
        for época in range(self.k_epochs):
            rng.shuffle(idx)      # (v2) mezclar: antes los minibatches eran trozos consecutivos
                                  # del lote, o sea muy correlacionados entre sí.
            kl_época = []
            for inicio in range(0, n, self.batch_size):
                mb = idx[inicio:inicio + self.batch_size]
                if len(mb) < 2:
                    continue
                mb_t = torch.as_tensor(mb, device=dev)

                batch_states = states[mb_t]
                batch_actions = actions[mb_t]
                batch_old_logprobs = old_logprobs[mb_t]
                batch_advantages = advantages[mb_t]
                batch_returns = returns[mb_t]

                # (v2) Normalizar la VENTAJA por minibatch. Esto es lo que normaliza PPO:
                # la ventaja indica "mejor o peor de lo esperado" y solo importa su signo
                # y su magnitud relativa, no su escala absoluta.
                batch_advantages = (batch_advantages - batch_advantages.mean()) / \
                                   (batch_advantages.std() + 1e-8)

                # Evaluar la política actual con los estados del lote
                dist, value = self.policy(batch_states)
                logprobs = dist.log_prob(batch_actions).sum(dim=-1)
                dist_entropy = dist.entropy().sum(dim=-1)
                value = value.squeeze(-1)

                # Paso 7 (Parte A): Ratio de probabilidades para el Actor
                # Compara la probabilidad de la red actual frente a la que recolectó los datos.
                logratio = logprobs - batch_old_logprobs
                ratios = torch.exp(logratio)

                with torch.no_grad():
                    # Estimador de KL de Schulman (siempre positivo, menos varianza)
                    approx_kl = ((ratios - 1.0) - logratio).mean().item()
                    clipfrac = ((ratios - 1.0).abs() > self.eps_clip).float().mean().item()
                kl_época.append(approx_kl)

                # Paso 7 (Parte B): Función de pérdida recortada (clipped objective function) de PPO
                # Recorta el ratio dentro de un rango (1-eps a 1+eps) para evitar cambios destructivos.
                surr1 = ratios * batch_advantages
                surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                # Paso 6: Optimizar la Red del Crítico
                # (v2) error cuadrático por muestra y luego promedio (antes el MSE ya venía
                # promediado y se sumaba a un vector, lo cual funcionaba por broadcasting
                # pero era confuso).
                value_loss = 0.5 * ((value - batch_returns) ** 2).mean()

                entropy_loss = -dist_entropy.mean()

                # Pérdida combinada: se invierte el signo de la política para MAXIMIZAR la
                # ganancia usando un optimizador que MINIMIZA.
                loss = policy_loss + self.cfg.vf_coef * value_loss + self.cfg.ent_coef * entropy_loss

                # Aplicar Backpropagation y Descenso de Gradiente (Optimización con Adam)
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.cfg.max_grad_norm)
                self.optimizer.step()

                métricas = {
                    "policy_loss": policy_loss.item(),
                    "value_loss": value_loss.item(),
                    "entropy": dist_entropy.mean().item(),
                    "approx_kl": approx_kl,
                    "clipfrac": clipfrac,
                    "log_std": float(self.policy.actor_log_std.mean().item()),
                }

            # Corte temprano si la política se alejó demasiado de la que recolectó los datos
            if self.cfg.target_kl is not None and kl_época and np.mean(kl_época) > self.cfg.target_kl:
                métricas["epocas_usadas"] = época + 1
                break
        métricas.setdefault("epocas_usadas", self.k_epochs)
        return métricas

    # ------------------------------------------------------------------
    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "policy": self.policy.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "obs_rms": self.obs_rms.state_dict() if self.obs_rms else None,
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
        }, path)

    def load(self, path):
        ck = torch.load(path, map_location=self.device, weights_only=False)
        self.policy.load_state_dict(ck["policy"])
        if ck.get("optimizer"):
            self.optimizer.load_state_dict(ck["optimizer"])
        if ck.get("obs_rms") and self.obs_rms is not None:
            self.obs_rms.load_state_dict(ck["obs_rms"])
        return self


# ======================================================================
# ENTRENAMIENTO DE UNA ETAPA DEL CURRÍCULO
# ======================================================================
def train_stage(env, agent, stage, total_timesteps=None, logger=None, timelapse=None,
                ckpt_dir="checkpoints/ppo", verbose_every=10, seed=None):
    """
    Entrena al agente PPO durante UNA etapa del currículo.

    env              : BalanceBotEnv (la etapa se aplica con env.set_stage)
    agent            : PPOAgent (se reutiliza entre etapas, así se transfiere lo aprendido)
    stage            : StageConfig
    total_timesteps  : pasos de entorno para esta etapa (por defecto stage.max_timesteps)
    logger           : TrainLogger o None (si es None se crea uno para esta etapa)
    timelapse        : Timelapse o None
    Devuelve un diccionario con el resumen de la etapa.

    Paso 9 del pseudocódigo: bucle general. Se repite:
        recolectar un lote de num_steps pasos -> calcular ventajas -> optimizar K épocas
    """
    cfg = agent.cfg
    env.set_stage(stage)
    total_timesteps = int(total_timesteps or stage.max_timesteps)
    cerrar_logger = logger is None
    if logger is None:
        logger = TrainLogger("ppo", stage)

    rng_seed = cfg.seed if seed is None else seed
    obs, _ = env.reset(seed=rng_seed)
    stats = EpisodeStats(window=30)
    paso = 0
    n_iter = 0
    n_episodios = 0
    umbral_promocion = stage.promote_length_frac * env.max_steps
    promovido_por_desempeño = False

    while paso < total_timesteps:
        n_iter += 1

        # --- Annealing de la tasa de aprendizaje ---
        if cfg.anneal_lr:
            frac = 1.0 - paso / total_timesteps
            lr_actual = cfg.lr * max(frac, 0.0)
        else:
            lr_actual = cfg.lr

        # ======== Paso 3: Recolectar un lote de trayectorias ========
        memory = {'obs': [], 'actions': [], 'logprobs': [], 'rewards': [],
                  'dones': [], 'truncs': [], 'values': [], 'next_values': []}

        for _ in range(cfg.num_steps):
            action, logprob, value, obs_n = agent.select_action(obs, actualizar_rms=True)
            next_obs, reward, terminated, truncated, info = env.step(action)

            memory['obs'].append(obs_n)
            memory['actions'].append(action)
            memory['logprobs'].append(logprob)
            memory['rewards'].append(reward)
            memory['dones'].append(float(terminated))     # solo caída real corta el retorno
            memory['truncs'].append(float(truncated))     # el corte por tiempo hace bootstrap
            memory['values'].append(value)
            # V(s') se necesita solo si el episodio se truncó por tiempo
            if truncated:
                memory['next_values'].append(agent.value_of(agent.normaliza(next_obs)))
            else:
                memory['next_values'].append(0.0)

            paso += 1
            obs = next_obs

            if terminated or truncated:
                ep = info["episode"]
                stats.add(ep["l"], ep["r"])
                n_episodios += 1
                logger.log_episode(ep, paso)
                obs, _ = env.reset()

        # ======== Pasos 4 y 5: retornos y ventajas (GAE-lambda) ========
        last_value = agent.value_of(agent.normaliza(obs))
        advantages, returns = compute_gae(
            memory['rewards'], memory['values'], memory['dones'], last_value,
            gamma=cfg.gamma, lam=cfg.gae_lambda,
            truncated=memory['truncs'], next_values=memory['next_values'],
        )
        memory['advantages'] = advantages
        memory['returns'] = returns

        # ======== Pasos 6, 7 y 8: optimizar actor y crítico ========
        métricas = agent.update(memory, lr=lr_actual)
        logger.log_losses(métricas, paso)
        logger.scalar("charts/learning_rate", lr_actual, paso)
        logger.scalar("charts/valor_medio", float(np.mean(memory['values'])), paso)
        logger.log_sps(paso)

        # --- Mensajes y timelapse ---
        if n_iter % verbose_every == 0 or paso >= total_timesteps:
            print(f"  [ppo etapa {stage.index}] paso {paso:>7,}/{total_timesteps:,}  "
                  f"retorno={stats.mean_return():8.1f}  duración={stats.mean_length():7.1f}  "
                  f"episodios={n_episodios}")

        if timelapse is not None and timelapse.due(paso):
            frames = timelapse.record(agent.act_deterministic, stage, paso, algo="ppo")
            logger.log_frames(frames, paso)

        # --- Promoción temprana: el robot ya sobrevive episodios completos ---
        if (paso >= stage.min_timesteps and len(stats) >= 10
                and stats.mean_length() >= umbral_promocion):
            promovido_por_desempeño = True
            print(f"  [ppo etapa {stage.index}] promoción temprana: duración media "
                  f"{stats.mean_length():.0f} >= {umbral_promocion:.0f}")
            break

    # Guardar el modelo de la etapa
    ckpt = Path(ckpt_dir) / f"ppo_etapa{stage.index}_{stage.name}.pt"
    agent.save(ckpt)

    if timelapse is not None:
        frames = timelapse.record(agent.act_deterministic, stage, paso, algo="ppo", force=True)
        logger.log_frames(frames, paso, tag="timelapse_final")

    resumen = {
        "algoritmo": "ppo", "etapa": stage.index, "nombre": stage.name,
        "pasos": paso, "episodios": n_episodios,
        "retorno_medio": stats.mean_return(), "duracion_media": stats.mean_length(),
        "promovido_temprano": promovido_por_desempeño,
        "checkpoint": str(ckpt), "logdir": str(logger.logdir),
    }
    if cerrar_logger:
        logger.close()
    return resumen