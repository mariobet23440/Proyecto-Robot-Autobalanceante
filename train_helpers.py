"""Despacha train_stage() según el algoritmo. Separado para no importar torch al cargar curriculum."""


def train_one_stage(algo, env, agent, stage, total_timesteps, logger, timelapse, buffer):
    if algo == "ppo":
        from ppo import train_stage
        resumen = train_stage(env, agent, stage, total_timesteps=total_timesteps,
                              logger=logger, timelapse=timelapse)
        return resumen, None
    if algo == "ddpg":
        from ddpg import train_stage
        return train_stage(env, agent, stage, total_timesteps=total_timesteps,
                           logger=logger, timelapse=timelapse, buffer=buffer)
    raise ValueError(algo)