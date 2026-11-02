'''
Proximal Policy Optimization (PPO)
CREADO POR: Mario Betancourt
SOBRE USO DE I.A. : La primera iteración de este script fue generado por Gemini, sin documentación
                    Los comentarios fueron hechos a teclado por Mario Betancourt a modo de entender 
                    cómo funciona esta implementación.
FUENTES DE INFORMACIÓN: Se utilizó la documentación de OpenAI sobre PPO y el pseudocódigo simplificado
                        del video "Simply Explaining Proximal Policy Optimization (PPO) | Deep 
                        Reinforcement Learning" de Jhonny Code como guías para programar.
'''

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal

# ==========================================
# CONFIGURACIÓN INICIAL: REDES NEURONALES
# ==========================================

class ActorCritic(nn.Module):
    '''
    Clase para la red neuronal del actor y el critic. Se utiliza un perceptrón multicapas (MLP) con
    dos capas ocultas de 64 nodos y capas de entrada y salida de dimensiones ajustables. Las
    capas ocultas utilizan funciones de activación ReLU y la capa de salida usa una función
    de activación lineal. Se usa la misma arquitectura para las NN de ambos el Actor y el Critic.
    '''
    def __init__(self, state_dim, action_dim):
        super(ActorCritic, self).__init__()

        # Paso 1: Inicializar la Red del Actor (Actor Network)
        # Esta red aproxima la política como una distribución de probabilidades pi(a | s)
        self.actor = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )

        # Crear un parámetro vector de dimensión action_dim relleno de ceros
        # Contiene el logaritmo de la desviación estándar para controlar la exploración.
        self.actor_log_std = nn.Parameter(torch.zeros(1, action_dim))

        # Paso 2: Inicializar la Red del Crítico (Critic Network)
        # Se inicializa de forma independiente otra red neuronal encargada de estimar el Valor del estado V(s).
        # El valor es el retorno total esperado que el agente recibirá a largo plazo partiendo de un estado.
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, state):
        '''Propagación hacia adelante'''
        value = self.critic(state)              # Aproximación de la función de valor de estado V(state)
        mu = self.actor(state)                  # Aproximación de la media de la política para state pi(a | state)
        std = torch.exp(self.actor_log_std)     # Desviación estándar
        dist = Normal(mu, std)                  # Política - distribución normal parametrizada por la acción
        return dist, value                      # Retornar política Y el valor de estado

class PPOAgent:
    '''Clase para un agente PPO'''
    def __init__(self, state_dim, action_dim, lr=3e-4, gamma=0.99, eps_clip=0.2, k_epochs=4, batch_size=64):
        # HIPERPARÁMETROS
        self.gamma = gamma                  # Factor de descuento
        self.eps_clip = eps_clip            # Hiperparámetro de clipeo
        self.k_epochs = k_epochs            # Número de épocas de entrenamiento
        self.batch_size = batch_size        # Tamaño de batch de Replay Buffer

        # REDES NEURONALES Y OPTIMIZADORES
        self.policy = ActorCritic(state_dim, action_dim)                # Red neuronal para política actual
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)    # Optimizador Adam
        self.policy_old = ActorCritic(state_dim, action_dim)            # Red neuronal para política vieja
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        self.mse_loss = nn.MSELoss()        # Función de pérdida MSE

    def select_action(self, state):
        '''Seleccionar acción'''
        state = torch.FloatTensor(state)                            # Convertir el estado en un tensor de PyTorch
        with torch.no_grad():                                       # Desactivar backpropagation
            dist, value = self.policy_old(state)                    # Obtener distribución de política vieja y valor
            action = dist.sample()                                  # Muestrear una acción de la política (distribución normal)
            action_logprob = dist.log_prob(action).sum(dim=-1)      # Probabilidades logarítmicas
            
        return action.numpy(), action_logprob.item(), value.item()

    def update(self, memory):
        '''
        Actualizar la política y el critic usando los datos almacenados en la memoria.
        - memory : Diccionario que contiene los estados, acciones, logprobs, recompensas, terminales y valores.
        '''

        # Convertir los vectores de memoria en tensores de PyTorch
        states = torch.FloatTensor(np.array(memory['states']))
        actions = torch.FloatTensor(np.array(memory['actions']))
        old_logprobs = torch.FloatTensor(np.array(memory['logprobs']))
        rewards = memory['rewards']
        is_terminals = memory['is_terminals']
        values = torch.FloatTensor(np.array(memory['values']))

        # Paso 4: Calcular las recompensas acumuladas (Rewards-to-go)
        # Se calcula la suma descontada de las recompensas futuras de cada paso usando el factor gamma.
        returns = []                                                                    # Inicializar lista de retornos
        rewards_to_go = 0                                                               # Inicializar recompensa descontada 

        for reward, is_terminal in zip(reversed(rewards), reversed(is_terminals)):      # Iterar en orden inverso
            if is_terminal:                                                             
                rewards_to_go = 0
            discounted_reward = reward + (self.gamma * rewards_to_go)               # Aplicar factor de descuento lineal gamma[cite: 9]
            returns.insert(0, rewards_to_go)                                        # Insertar retorno acumulado

        # Convertir el vector de retornos en un tensor de PyTorch y normalizarlo
        returns = torch.FloatTensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + 1e-7)

        # Paso 5: Calcular las Ventajas (Advantages)
        # Se resta el valor estimado por el crítico (V) a la recompensa acumulada real (Q/Retorno).
        # Una ventaja positiva indica que la acción fue mejor de lo estimado; una negativa, que fue peor.
        advantages = returns - values.detach()

        # Paso 8: Repetir la optimización (Ciclos de entrenamiento para N épocas)
        # Los pasos 6 y 7 se ejecutan continuamente durante K épocas para exprimir el lote de datos.
        for _ in range(self.k_epochs):

            # Dividir los datos de memoria en minibatches para actualizar la política
            for i in range(0, len(states), self.batch_size):
                batch_states = states[i:i+self.batch_size]
                batch_actions = actions[i:i+self.batch_size]
                batch_old_logprobs = old_logprobs[i:i+self.batch_size]
                batch_advantages = advantages[i:i+self.batch_size]
                batch_returns = returns[i:i+self.batch_size]

                # Evaluar la política actual con los estados del lote
                dist, value = self.policy(batch_states)
                logprobs = dist.log_prob(batch_actions).sum(dim=-1)
                dist_entropy = dist.entropy().sum(dim=-1)
                value = value.squeeze(-1)  

                # Paso 7 (Parte A): Ratio de probabilidades para el Actor
                # Compara la probabilidad de la red actual frente a la red vieja.
                ratios = torch.exp(logprobs - batch_old_logprobs)

                # Paso 7 (Parte B): Función de pérdida recortada (clipped objective function) de PPO
                # Recorta el ratio dentro de un rango (ej. 1-eps a 1+eps) para evitar cambios destructivos de golpe.
                surr1 = ratios * batch_advantages
                surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * batch_advantages

                # Paso 6: Optimizar la Red del Crítico y Paso 7 (Parte C): Optimizar la Red del Actor
                # Se calcula la pérdida combinada: se invierte el signo de la política para maximizar la ganancia,
                # se añade la pérdida MSE del crítico y se incluye la entropía[cite: 2, 6, 7].
                loss = -torch.min(surr1, surr2) + 0.5 * self.mse_loss(value, batch_returns) - 0.01 * dist_entropy

                # Aplicar Backpropagation y Descenso de Gradiente (Optimización con Adam)
                self.optimizer.zero_grad()
                loss.mean().backward()
                self.optimizer.step()

        # Sincronizar la política vieja con los pesos actualizados de la red principal
        self.policy_old.load_state_dict(self.policy.state_dict())

def train():
    # Configuración inicial del entorno
    env = gym.make("Pendulum-v1") 
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    # Instanciar el agente PPO
    ppo_agent = PPOAgent(state_dim, action_dim)
    
    # Hiperparámetros globales del bucle de entrenamiento
    n_episodes = 500
    max_timesteps = 200
    update_timestep = 2000
    timestep = 0

    # Crear memoria temporal para trayectorias
    memory = {'states': [], 'actions': [], 'logprobs': [], 'rewards': [], 'is_terminals': [], 'values': []}

    # Paso 9: Bucle general (Ciclos de entrenamiento hasta alcanzar el límite global de pasos)[cite: 5]
    for episode in range(1, n_episodes + 1):
        state, _ = env.reset()
        current_ep_reward = 0

        # Paso 3: Recolectar un lote de trayectorias (Trajectories)
        # El agente interactúa con el entorno usando su política actual para simular experiencias[cite: 2, 4].
        # Se guarda el estado, la acción, la recompensa, el valor del crítico y el log-prob de la acción[cite: 6].
        for t in range(max_timesteps):
            timestep += 1
            action, logprob, value = ppo_agent.select_action(state)
            
            env_action = np.clip(action, env.action_space.low, env.action_space.high)
            next_state, reward, terminated, truncated, _ = env.step(env_action)
            done = terminated or truncated

            # Almacenar transición en la memoria
            memory['states'].append(state)
            memory['actions'].append(action)
            memory['logprobs'].append(logprob)
            memory['rewards'].append(reward)
            memory['is_terminals'].append(done)
            memory['values'].append(value)

            state = next_state
            current_ep_reward += reward

            # Disparar la actualización del agente cada vez que se acumulan suficientes pasos
            if timestep % update_timestep == 0:
                ppo_agent.update(memory)
                memory = {'states': [], 'actions': [], 'logprobs': [], 'rewards': [], 'is_terminals': [], 'values': []}

            if done:
                break

        print(f"Episodio {episode} \t Recompensa: {current_ep_reward:.2f}")

    env.close()

if __name__ == '__main__':
    train()