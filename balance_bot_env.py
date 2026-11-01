"""
BALANCE BOT ENVIRONMENT
CREADO POR: Mario Betancourt
NOTA: Este trabajo fue realizado en conjunto con Christian Echeverría. Dado que 
      cada uno implementa dos algoritmos de RL, con tal de avanzar cada uno realizó
      su propia realización del entorno de Gymnasium.
SOBRE EL USO DE I.A : Este script de Python fue generado a partir del Script de
                      la serie de videos de Digikey y Shawn Hymel 
                      "Reinforcement Learning for Robotics". La documentación del
                      script original fue traducida al español usando un agente de Gemini.
                      Posteriormente se hicieron modificaciones a este script para
                      replantear la función de recompensa.
"""

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces

class BalanceBotEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 50}

    def __init__(self, mjcf_path):
        super().__init__()
        
        # Cargar modelo y datos de MuJoCo
        self.model = mujoco.MjModel.from_xml_path(mjcf_path)
        self.data = mujoco.MjData(self.model)
        
        # Definir espacio de acciones (torque motores izq y der, normalizado de -1 a 1)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        
        # Definir espacio de observaciones (ej. pitch, pitch_rate, vel_wheel_l, vel_wheel_r)
        high_obs = np.array([np.pi, 20.0, 50.0, 50.0], dtype=np.float32)
        self.observation_space = spaces.Box(low=-high_obs, high=high_obs, dtype=np.float32)
        
        self.max_tilt = np.deg2rad(30) # 30 grados de límite antes de caer

    def reset(self, seed=None, options=None):
        super().reset(seed = seed)
        mujoco.mj_resetData(self.model, self.data)
        
        # Opcional: Aplicar perturbación inicial aleatoria para el entrenamiento
        self.data.qpos[0] = 0.0 # Posición inicial
        
        observation = self._get_observation()
        info = {}
        return observation, info

    def step(self, action):
        # 1. Aplicar la acción a los actuadores
        self.data.ctrl[:] = action
        
        # 2. Avanzar un paso en la simulación de MuJoCo
        mujoco.mj_step(self.model, self.data)
        
        # 3. Obtener nuevas observaciones
        observation = self._get_observation()
        
        # 4. Calcular recompensa y comprobar si el robot cayó (terminated)
        pitch = observation[0]
        terminated = bool(abs(pitch) > self.max_tilt)
        truncated = False
        
        reward = 1.0 if not terminated else 0.0 # Ejemplo básico de recompensa
        info = {}
        
        return observation, reward, terminated, truncated, info

    def _get_observation(self):
        # Extraer datos de sensores / cinemática para armar el vector de observación
        # (Aquí implementarías la lectura de tu IMU y encoders según tu archivo XML)
        pitch = self.data.qpos[1] if self.model.nq > 1 else 0.0
        pitch_rate = self.data.qvel[1] if self.model.nv > 1 else 0.0
        wheel_vel_l = self.data.qvel[2] if self.model.nv > 2 else 0.0
        wheel_vel_r = self.data.qvel[3] if self.model.nv > 3 else 0.0
        
        return np.array([pitch, pitch_rate, wheel_vel_l, wheel_vel_r], dtype=np.float32)

    def render(self):
        # Implementar lógica de renderizado si se usa visor interactivo
        pass

    def close(self):
        pass