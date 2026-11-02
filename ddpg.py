'''
Deep Deterministic Policy Gradient (DDPG)
CREADO POR: Mario Betancourt
SOBRE USO DE I.A. : La primera iteración de este script fue generado por Gemini, sin documentación
                    Los comentarios fueron hechos a teclado por Mario Betancourt a modo de entender 
                    cómo funciona esta implementación.
'''

# LIBRERÍAS
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque   # Double-ended Queue

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
        valor booleano que indica si el estado S_{t+1} es un estado terminal.
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
    Clase que represennta la plantilla de la NN del Actor en DDPG.
    """
    def __init__(self, state_dim, action_dim, max_action):
        """
        Se usa un Perceptrón Multicapa (MLP) con una capa de entrada de state_dim nodos,
        dos capas ocultas de 32 nodos y una capa de salida de dimensión ajustable. 
        Las capas ocultas utilizan funciones de activación ReLU y la capa de salida
        utiliza una función tangente hiperbólica (para saturar las acciones entre -1 y 1). 
        La salida estima el valor de la política en un estado particular.
        """

        super(Actor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 32),   # Capa de entrada y capa oculta 1
            nn.ReLU(),
            nn.Linear(32, 32),          # Capa oculta 1 y capa oculta 2
            nn.ReLU(),
            nn.Linear(32, action_dim), # Capa oculta 2 y capa de salida
            nn.Tanh() # Escala las acciones al rango [-1, 1]
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
    Clase que represennta la plantilla de la NN del Critic en DDPG.
    """

    def __init__(self, state_dim, action_dim):
        """
        Se usa un Perceptrón Multicapa (MLP) con una capa de entrada de state_dim + action_dim 
        nodos, dos capas ocultas de 32 nodos y una capa de salida de dimensión ajustable. 
        Las capas ocultas utilizan funciones de activación ReLU. La salida aproxima la función
        de valor de acción.
        """
        super(Critic, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, 32),  # Capa de entrada y capa oculta 1
            nn.ReLU(),
            nn.Linear(32, 32),                      # Capa oculta 1 y capa oculta 2
            nn.ReLU(),
            nn.Linear(32, 1)                        # Capa oculta 2 y capa de salida
        )

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        return self.net(x)


# --- 4. Agente DDPG ---
class DDPGAgent:
    """
    Clase que representa el agente DDPG.
    """

    def __init__(self, state_dim, action_dim, max_action):
        # Seleccionar el dispositivo para procesar modelos
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # TO DO: Seleccionar GPUs de Intel si hay disponibles.

        # Instanciar online networks y target networks de ACTOR
        self.actor = Actor(state_dim, action_dim, max_action).to(self.device)
        self.actor_target = Actor(state_dim, action_dim, max_action).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=1e-4)

        # Instanciar online networks y target networks de CRITIC
        self.critic = Critic(state_dim, action_dim).to(self.device)
        self.critic_target = Critic(state_dim, action_dim).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=1e-3)

        self.max_action = max_action    # Valor máximo de acción
        self.gamma = 0.99               # Factor de descuento
        self.tau = 0.005                # Tasa de actualización suave (Polyak averaging)

    def select_action(self, state, noise=0.1):
        """
        Selecciona la acción en un estado determinado.
        """

        # Convertir el estado (arreglo de NumPy) en un tensor de PyTorch
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        # Inferencia en modo evaluación
        self.actor.eval()                                               # Pone la red en modo evaluación
        with torch.no_grad():                                           # Desactiva backpropagation
            action = self.actor(state).cpu().data.numpy().flatten()     # Calcular acción
        self.actor.train()                                              # Restaura la red en estado de entrenamiento
        
        # Agregar ruido gaussiano para la exploración
        if noise != 0:
            action += np.random.normal(0, noise, size=action.shape)
        
        # Retornar acción limitada entre valores máximo y mínimo
        return np.clip(action, -self.max_action, self.max_action)

    def train(self, replay_buffer, batch_size=64):
        """
        Entrenar agente DDPG
        """

        # No empezar a entrenar hasta que el Replay Buffer se haya llenado
        if len(replay_buffer) < batch_size:
            return

        # -- 1. Muestrear un batch de secuencias (S_t, A_t, R_{t+1}, S_{t+1}, done) --
        states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)

        # 1.1. Convertir batches a tensores de PyTorch
        state = torch.FloatTensor(states).to(self.device)
        action = torch.FloatTensor(actions).to(self.device)
        reward = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_state = torch.FloatTensor(next_states).to(self.device)
        done = torch.FloatTensor(dones).unsqueeze(1).to(self.device)

        # 2. -- Actualizar el Crítico --
        # 2.1. Estimar Q(s,a) con target network del Critic
        with torch.no_grad():                                               # Desactivar backpropagation (cambio de parámetros)
            next_action = self.actor_target(next_state)                     # Calcular acción siguiente con la target network del Actor
            target_Q = self.critic_target(next_state, next_action)          # Calcular Q(s,a) con target network del Critic
            target_Q = reward + (1 - done) * self.gamma * target_Q          # Actualizar Q(s,a) con la recompensa muestreada (target_Q no se ha actualizado)
            # ¡Esto es de Q-Learning! ⤴

        # 2.2 Estimar Q(s,a) con online network del Critic
        current_Q = self.critic(state, action)                              # Calcular Q(s,a) con online network del Critic
        
        # 2.3. Calcular pérdida MSE con las dos NN
        critic_loss = nn.MSELoss()(current_Q, target_Q)                     # Calcular la pérdida Mean Square Error (MSE) de las Critic Networks

        # 2.4. Reestablecer gradientes, backpropagation y dar un paso
        self.critic_optimizer.zero_grad()                                   # Pone en cero los gradientes acumulados
        critic_loss.backward()                                              # Ejecuta backpropagation
        self.critic_optimizer.step()                                        # Realiza un paso de optimización

        # 3. -- Actualizar el Actor --
        actor_loss = -self.critic(state, self.actor(state)).mean()

        self.actor_optimizer.zero_grad()                                    # Reiniciar gradientes
        actor_loss.backward()                                               # Hacer backpropagation
        self.actor_optimizer.step()                                         # Dar un paso de optimización de parámetros

        # 4. -- Actualización suave de las redes objetivo --
        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):    # Esto recorre los parámetros sincronizados
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)     # Y esto hace una interpolación

        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        # tau es la tasa de actualización suave.

        # Estas cuatro líneas interpolan los vectores de parámetros de la online network y
        # la target network. Los factores tau y (1 - tau) permiten hacer un compromiso de qué
        # tanto contribuye cada red a la interpolación con un mismo parámetro. Un valor más grande de tau
        # le da mayor prioridad a los pesos de la online network, haciendo que la target network "persiga"
        # más de cerca a la online network.