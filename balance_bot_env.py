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
VERSIÓN 2 (Claude): Reescrita con asistencia de Claude usando como referencia el entorno
                      con Domain Randomization de Shawn Hymel (balance_bot_env_dr.py).
                      Cambios principales:
                        - Lectura correcta de la simulación (el robot tiene un freejoint, así que
                          qpos[1] NO es el pitch; ver notas en _true_state).
                        - Pitch estimado con filtro complementario a partir de la IMU.
                        - Comandos externos (velocidad hacia adelante y velocidad de giro).
                        - Las 8 etapas del currículo (ver curriculum.py) como StageConfig.
                        - Límite de tiempo con truncated=True.
                        - Recompensa nueva (ver REWARD en la clase y el cuadernillo).
"""

import math

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces

from curriculum import StageConfig, get_stage


class BalanceBotEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}

    # ------------------------------------------------------------------
    # PESOS DE LA RECOMPENSA (ver explicación en el cuadernillo, sección 3)
    # ------------------------------------------------------------------
    ALIVE_BONUS = 1.0                    # premio por cada paso que el robot sigue de pie
    K_PITCH = 1.0                        # k1: castigo proporcional a |pitch|          (idea de Mario)
    K_UPRIGHT = 0.5                      # k2: premio si |pitch| <= THETA_MIN          (idea de Mario)
    THETA_MIN = np.deg2rad(2.0)          # theta_min = 2 grados                        (idea de Mario)
    K_ACTION = 0.01                      # castigo a la magnitud del torque
    K_ACTION_RATE = 0.01                 # castigo a cambios bruscos de torque (suavidad)
    K_VEL = 2.0                          # castigo al error de velocidad hacia adelante (etapa 2+)
    K_LAT = 2.0                          # castigo a la velocidad lateral              (etapa 2+)
    K_YAW = 0.1                          # castigo al error de velocidad de giro       (etapa 2+)
    K_POS = 0.001                        # castigo a alejarse del origen (solo si no hay comandos)

    # Límites de aceleración con los que el comando observado sigue al comando objetivo
    CMD_ACCEL_V = 0.6                    # [m/s^2]
    CMD_ACCEL_W = 6.0                    # [rad/s^2]

    def __init__(
        self,
        mjcf_path="autobalance.xml",
        stage=None,
        max_steps=2000,
        render_mode=None,
        use_encoders=True,
        alpha=0.99,
        max_tilt_deg=30.0,
        wheel_radius=0.0225,
        render_size=(240, 320),
    ):
        """
        mjcf_path    : ruta al XML del robot
        stage        : StageConfig (o número de etapa 1..8). Por defecto, etapa 1.
        max_steps    : pasos por episodio antes de truncar (2000 pasos * 5 ms = 10 s)
        render_mode  : None, "human" (visor interactivo) o "rgb_array" (imágenes para el timelapse)
        use_encoders : True  -> la velocidad se mide con los encoders de las ruedas.
                       False -> la velocidad se estima integrando el acelerómetro (con fuga).
                                Sirve para probar la decisión de NO usar encoders, pero es una
                                estimación muy limitada: ver el cuadernillo.
        alpha        : coeficiente del filtro complementario (más alto = confía más en el giroscopio)
        """
        super().__init__()
        assert render_mode in (None, "human", "rgb_array")
        self.render_mode = render_mode
        self.use_encoders = use_encoders
        self.alpha = alpha
        self.max_steps = max_steps
        self.max_tilt = np.deg2rad(max_tilt_deg)
        self.wheel_radius = wheel_radius
        self._render_h, self._render_w = render_size

        if stage is None:
            stage = get_stage(1)
        elif isinstance(stage, int):
            stage = get_stage(stage)
        self.stage = stage

        # --- Cargar modelo y datos de MuJoCo ---
        self.model = mujoco.MjModel.from_xml_path(str(mjcf_path))
        self.data = mujoco.MjData(self.model)
        self.dt = float(self.model.opt.timestep)

        # --- Identificadores (los nombres deben coincidir con autobalance.xml) ---
        def _id(kind, name):
            idx = mujoco.mj_name2id(self.model, kind, name)
            assert idx != -1, f"'{name}' no existe en el XML"
            return idx

        self._chassis = _id(mujoco.mjtObj.mjOBJ_BODY, "chassis")
        self._imu_site = _id(mujoco.mjtObj.mjOBJ_SITE, "imu_site")
        self._ground_geom = _id(mujoco.mjtObj.mjOBJ_GEOM, "ground")
        self._wheel_geoms = [_id(mujoco.mjtObj.mjOBJ_GEOM, "wheel_left_geom"),
                             _id(mujoco.mjtObj.mjOBJ_GEOM, "wheel_right_geom")]
        self._friction_geoms = [self._ground_geom] + self._wheel_geoms

        jl = _id(mujoco.mjtObj.mjOBJ_JOINT, "left_wheel_joint")
        jr = _id(mujoco.mjtObj.mjOBJ_JOINT, "right_wheel_joint")
        self._left_dof = int(self.model.jnt_dofadr[jl])
        self._right_dof = int(self.model.jnt_dofadr[jr])

        self._act_ids = [_id(mujoco.mjtObj.mjOBJ_ACTUATOR, "left_motor"),
                         _id(mujoco.mjtObj.mjOBJ_ACTUATOR, "right_motor")]

        def _sensor_adr(name):
            return int(self.model.sensor_adr[_id(mujoco.mjtObj.mjOBJ_SENSOR, name)])

        self._s_accel = _sensor_adr("imu_accel")
        self._s_gyro = _sensor_adr("imu_gyro")
        self._s_lvel = _sensor_adr("left_wheel_vel")
        self._s_rvel = _sensor_adr("right_wheel_vel")

        # --- Valores originales (para restaurar antes de cada randomización) ---
        m = self.model
        self._orig_mass = float(m.body_mass[self._chassis])
        self._orig_inertia = m.body_inertia[self._chassis].copy()
        self._orig_ipos = m.body_ipos[self._chassis].copy()
        self._orig_imu_pos = m.site_pos[self._imu_site].copy()
        self._orig_friction = {g: float(m.geom_friction[g, 0]) for g in self._friction_geoms}
        self._orig_gear = {a: float(m.actuator_gear[a, 0]) for a in self._act_ids}

        # --- Espacios ---
        # Acción: torque normalizado [izquierdo, derecho] en [-1, 1]
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

        # Observación (8):
        #   0 pitch estimado (filtro complementario IMU) [rad]
        #   1 velocidad angular de pitch (giroscopio y)  [rad/s]
        #   2 velocidad de giro / yaw (giroscopio z)     [rad/s]
        #   3 velocidad hacia adelante (encoders, o IMU si use_encoders=False) [m/s]
        #   4 comando de velocidad hacia adelante        [m/s]
        #   5 comando de velocidad de giro               [rad/s]
        #   6-7 acción anterior (izq, der)  -> ayuda a compensar los retardos
        self._obs_high = np.array([np.pi, 50.0, 50.0, 5.0, 2.0, 6.0, 1.0, 1.0], dtype=np.float32)
        self.observation_space = spaces.Box(low=-self._obs_high, high=self._obs_high, dtype=np.float32)

        # --- Estado interno ---
        self._viewer = None
        self._renderer = None
        self._cam = None
        self._manual_cmd = None
        self._reset_internal()

    # ==================================================================
    # API pública extra
    # ==================================================================
    def set_stage(self, stage):
        """Cambia de etapa (se aplica desde el siguiente reset)."""
        self.stage = get_stage(stage) if isinstance(stage, int) else stage

    def set_command(self, v=None, w=None):
        """
        Fija un comando externo constante: v [m/s] hacia adelante y w [rad/s] de giro.
        set_command() sin argumentos devuelve el control a los comandos aleatorios de la etapa.
        Nota: un robot de dos ruedas NO puede moverse lateralmente, por eso el comando es
        (velocidad hacia adelante, velocidad de giro) y no (X, Y) en el sentido cartesiano.
        """
        if v is None and w is None:
            self._manual_cmd = None
        else:
            self._manual_cmd = np.array([0.0 if v is None else v, 0.0 if w is None else w])

    # ==================================================================
    # Gymnasium
    # ==================================================================
    def _reset_internal(self):
        self._step = 0
        self._pitch_est = 0.0
        self._v_imu = 0.0
        self._last_action = np.zeros(2)
        self._fifo = []
        self._delay = 0
        self._push_left = 0
        self._push_force = np.zeros(2)
        self._pulse_left = 0
        self._pulse_torque = np.zeros(2)
        self._cmd = np.zeros(2)
        self._cmd_target = np.zeros(2)
        self._origin_xy = np.zeros(2)
        self._motor_gain = 1.0
        self._ep_ret = 0.0
        self._ep_abs_pitch = 0.0
        self._ep_speed_err = 0.0
        self._ep_yaw_err = 0.0
        self._ep_terms = {}

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        st = self.stage
        self._reset_internal()
        if options is not None and "command" in options:
            self.set_command(*options["command"])

        # 1) Randomizar parámetros físicos (antes de resetear los datos)
        self._randomize_physics()

        # 2) Resetear simulación
        mujoco.mj_resetData(self.model, self.data)

        # 3) Condiciones iniciales: pitch, yaw (rumbo) y velocidad angular de pitch aleatorios.
        #    El robot tiene un freejoint: qpos = [x, y, z, qw, qx, qy, qz]; qvel = [vx, vy, vz, wx, wy, wz]
        #    (la velocidad angular del freejoint está en el marco local del cuerpo).
        pitch0 = float(self.np_random.normal(0.0, st.init_pitch_std))
        yaw0 = float(self.np_random.uniform(-math.pi, math.pi))
        c, s = math.cos(pitch0 / 2.0), math.sin(pitch0 / 2.0)    # rotación en y (pitch)
        a, d = math.cos(yaw0 / 2.0), math.sin(yaw0 / 2.0)        # rotación en z (yaw)
        # q = q_yaw * q_pitch
        self.data.qpos[3:7] = [a * c, -d * s, a * s, d * c]
        self.data.qvel[4] = float(self.np_random.normal(0.0, st.init_pitch_rate_std))
        mujoco.mj_forward(self.model, self.data)
        self._origin_xy = self.data.qpos[0:2].copy()

        # 4) Retardo de acción para este episodio (buffer FIFO relleno con ceros)
        if st.max_delay_steps > 0:
            self._delay = int(self.np_random.integers(0, st.max_delay_steps + 1))
        self._fifo = [np.zeros(2) for _ in range(self._delay)]

        # 5) Inicializar el filtro complementario con la lectura del acelerómetro
        accel = self.data.sensordata[self._s_accel:self._s_accel + 3]
        self._pitch_est = -math.atan2(float(accel[0]), float(accel[2]))

        return self._get_obs(), {}

    def step(self, action):
        st = self.stage
        action = np.clip(np.asarray(action, dtype=np.float64).reshape(2), -1.0, 1.0)

        # 1) Retardo de la acción (FIFO)
        if self._delay > 0:
            self._fifo.append(action.copy())
            applied = self._fifo.pop(0)
        else:
            applied = action

        # 2) Ruido en la señal de control
        if st.control_noise_std > 0.0:
            applied = np.clip(applied + self.np_random.normal(0.0, st.control_noise_std, size=2), -1.0, 1.0)

        self.data.ctrl[self._act_ids[0]] = applied[0]
        self.data.ctrl[self._act_ids[1]] = applied[1]

        # 3) Perturbaciones y comando
        self._apply_disturbances()
        self._update_command()

        # 4) Avanzar la simulación
        mujoco.mj_step(self.model, self.data)
        self._step += 1

        # 5) Observación (con sensores ruidosos) y estado verdadero (solo para recompensa/terminación)
        obs = self._get_obs()
        pitch, v_fwd, v_lat, w_yaw, pos_xy = self._true_state()

        # 6) Recompensa
        terms = {
            "alive": self.ALIVE_BONUS,
            "pitch": -self.K_PITCH * abs(pitch),
            "upright": self.K_UPRIGHT * float(abs(pitch) <= self.THETA_MIN),
            "action": -self.K_ACTION * float(np.sum(action ** 2))
                      - self.K_ACTION_RATE * float(np.sum((action - self._last_action) ** 2)),
        }
        speed_err = v_fwd - self._cmd[0]
        yaw_err = w_yaw - self._cmd[1]
        if st.penalize_drift:
            terms["vel"] = -self.K_VEL * speed_err ** 2 - self.K_LAT * v_lat ** 2
            terms["yaw"] = -self.K_YAW * abs(yaw_err)
            if not st.use_commands and self._manual_cmd is None:
                dxy = pos_xy - self._origin_xy
                terms["pos"] = -self.K_POS * float(dxy @ dxy)
        reward = float(sum(terms.values()))
        self._last_action = action

        # 7) Terminación / truncamiento
        terminated = bool(abs(pitch) > self.max_tilt)
        truncated = bool(self._step >= self.max_steps) and not terminated

        # 8) Estadísticas del episodio
        self._ep_ret += reward
        self._ep_abs_pitch += abs(pitch)
        self._ep_speed_err += abs(speed_err)
        self._ep_yaw_err += abs(yaw_err)
        for k, v in terms.items():
            self._ep_terms[k] = self._ep_terms.get(k, 0.0) + v

        info = {}
        if terminated or truncated:
            n = float(self._step)
            dxy = pos_xy - self._origin_xy
            info["episode"] = {
                "r": self._ep_ret,
                "l": self._step,
                "mean_abs_pitch_deg": math.degrees(self._ep_abs_pitch / n),
                "mean_speed_err": self._ep_speed_err / n,
                "mean_yaw_err": self._ep_yaw_err / n,
                "final_dist": float(np.linalg.norm(dxy)),
                "terms": {k: v / n for k, v in self._ep_terms.items()},
                "terminated": terminated,
            }
        return obs, reward, terminated, truncated, info

    # ==================================================================
    # Auxiliares
    # ==================================================================
    def _randomize_physics(self):
        """Restaura los parámetros originales y aplica la randomización de la etapa actual."""
        st = self.stage
        m = self.model

        # Restaurar
        m.body_mass[self._chassis] = self._orig_mass
        m.body_inertia[self._chassis] = self._orig_inertia
        m.body_ipos[self._chassis] = self._orig_ipos
        m.site_pos[self._imu_site] = self._orig_imu_pos
        for g, f in self._orig_friction.items():
            m.geom_friction[g, 0] = f
        for a, gear in self._orig_gear.items():
            m.actuator_gear[a, 0] = gear
        self._motor_gain = 1.0

        # Etapa 5: masa (y su inercia, para mantener la densidad) y centro de masa del chasis
        if st.randomize_mass:
            scale = float(self.np_random.uniform(*st.mass_scale_range))
            m.body_mass[self._chassis] = self._orig_mass * scale
            m.body_inertia[self._chassis] = self._orig_inertia * scale
            if st.com_shift_range > 0.0:
                shift = self.np_random.uniform(-st.com_shift_range, st.com_shift_range, size=3)
                m.body_ipos[self._chassis] = self._orig_ipos + shift

        # Etapa 5: fricción. Se escalan piso Y ruedas con el mismo coeficiente, porque MuJoCo usa el
        # MÁXIMO de las fricciones de los dos geoms en contacto; si solo se bajara la del piso por
        # debajo de la de las ruedas (0.8), no cambiaría nada.
        if st.randomize_friction:
            mu = float(self.np_random.uniform(*st.friction_range))
            for g in self._friction_geoms:
                m.geom_friction[g, 0] = mu

        # Etapa 3: posición de la IMU
        if st.imu_position_range > 0.0:
            r = st.imu_position_range
            m.site_pos[self._imu_site] = self._orig_imu_pos + self.np_random.uniform(-r, r, size=3)

        # Etapa 6: ganancia de los motores (caída de voltaje)
        if st.motor_gain_range != (1.0, 1.0):
            self._motor_gain = float(self.np_random.uniform(*st.motor_gain_range))
            for a, gear in self._orig_gear.items():
                m.actuator_gear[a, 0] = gear * self._motor_gain

    def _apply_disturbances(self):
        """Empujes (etapa 4) y pulsos de torque en los ejes (etapa 7)."""
        st = self.stage
        xfrc = self.data.xfrc_applied
        qfrc = self.data.qfrc_applied

        # Empujes externos en X e Y (marco del mundo) sobre el chasis
        if self._push_left > 0:
            self._push_left -= 1
        elif st.push_prob > 0.0 and self.np_random.random() < st.push_prob:
            self._push_force = self.np_random.uniform(-st.push_force_max, st.push_force_max, size=2)
            self._push_left = st.push_duration_steps
        if self._push_left > 0:
            xfrc[self._chassis, 0] = self._push_force[0]
            xfrc[self._chassis, 1] = self._push_force[1]
        else:
            xfrc[self._chassis, 0] = 0.0
            xfrc[self._chassis, 1] = 0.0

        # Pulsos aleatorios de torque en los ejes de las ruedas (estrías de las llantas)
        if self._pulse_left > 0:
            self._pulse_left -= 1
        elif st.axle_pulse_prob > 0.0 and self.np_random.random() < st.axle_pulse_prob:
            self._pulse_torque = self.np_random.uniform(-st.axle_pulse_torque_max,
                                                        st.axle_pulse_torque_max, size=2)
            self._pulse_left = st.axle_pulse_duration_steps
        if self._pulse_left > 0:
            qfrc[self._left_dof] = self._pulse_torque[0]
            qfrc[self._right_dof] = self._pulse_torque[1]
        else:
            qfrc[self._left_dof] = 0.0
            qfrc[self._right_dof] = 0.0

    def _update_command(self):
        """Actualiza el comando (v, w) que ve el robot. Sigue al objetivo con aceleración limitada."""
        st = self.stage
        if self._manual_cmd is not None:
            self._cmd_target = self._manual_cmd
        elif st.use_commands:
            if self._step % st.cmd_resample_steps == 0:
                if self.np_random.random() < st.cmd_zero_prob:
                    self._cmd_target = np.zeros(2)
                else:
                    self._cmd_target = np.array([
                        self.np_random.uniform(-st.cmd_v_max, st.cmd_v_max),
                        self.np_random.uniform(-st.cmd_w_max, st.cmd_w_max),
                    ])
        else:
            self._cmd_target = np.zeros(2)

        dv = float(np.clip(self._cmd_target[0] - self._cmd[0], -self.CMD_ACCEL_V * self.dt, self.CMD_ACCEL_V * self.dt))
        dw = float(np.clip(self._cmd_target[1] - self._cmd[1], -self.CMD_ACCEL_W * self.dt, self.CMD_ACCEL_W * self.dt))
        self._cmd = self._cmd + np.array([dv, dw])

    def _get_obs(self):
        """
        Observación a partir de los SENSORES (con ruido si la etapa lo pide), como la vería el
        firmware del robot real. Nada de aquí usa el estado verdadero de la simulación.
        """
        st = self.stage
        sd = self.data.sensordata
        accel = np.array(sd[self._s_accel:self._s_accel + 3], dtype=np.float64)
        gyro = np.array(sd[self._s_gyro:self._s_gyro + 3], dtype=np.float64)
        w_left = float(sd[self._s_lvel])
        w_right = float(sd[self._s_rvel])

        if st.sensor_noise:
            accel += self.np_random.normal(0.0, st.accel_noise_std, size=3)
            gyro += self.np_random.normal(0.0, st.gyro_noise_std, size=3)
            w_left += float(self.np_random.normal(0.0, st.encoder_noise_std))
            w_right += float(self.np_random.normal(0.0, st.encoder_noise_std))

        # Filtro complementario: giroscopio (corto plazo) + acelerómetro (largo plazo)
        accel_pitch = -math.atan2(accel[0], accel[2])
        self._pitch_est = self.alpha * (self._pitch_est + gyro[1] * self.dt) + (1.0 - self.alpha) * accel_pitch

        if self.use_encoders:
            # Velocidad angular absoluta de cada rueda = velocidad relativa al chasis + pitch rate.
            # Con una rueda que rueda sin deslizar, v = R * omega_absoluta.
            v_fwd = self.wheel_radius * (0.5 * (w_left + w_right) + gyro[1])
        else:
            # Integración con fuga de la aceleración horizontal (componente del acelerómetro en el
            # eje x del mundo). Se "olvida" con constante de tiempo de 1.5 s para no divergir.
            a_fwd = math.cos(self._pitch_est) * accel[0] + math.sin(self._pitch_est) * accel[2]
            self._v_imu = self._v_imu * (1.0 - self.dt / 1.5) + a_fwd * self.dt
            v_fwd = self._v_imu

        obs = np.array([self._pitch_est, gyro[1], gyro[2], v_fwd,
                        self._cmd[0], self._cmd[1], self._last_action[0], self._last_action[1]],
                       dtype=np.float32)
        return np.clip(obs, -self._obs_high, self._obs_high)

    def _true_state(self):
        """
        Estado VERDADERO (solo existe en simulación; se usa para la recompensa y la terminación).

        OJO: el robot tiene un freejoint, entonces
            qpos = [x, y, z, qw, qx, qy, qz, rueda_izq, rueda_der]   (nq = 9)
            qvel = [vx, vy, vz, wx, wy, wz, vel_izq, vel_der]         (nv = 8)
        Por eso qpos[1] es la posición en Y (NO el pitch) y qvel[2] es la velocidad vertical.
        El pitch se obtiene de la matriz de rotación del chasis.
        """
        R = self.data.xmat[self._chassis].reshape(3, 3)
        pitch = math.asin(float(np.clip(-R[2, 0], -1.0, 1.0)))     # + = inclinado hacia adelante
        heading = math.atan2(R[1, 0], R[0, 0])
        vx, vy = float(self.data.qvel[0]), float(self.data.qvel[1])
        v_fwd = math.cos(heading) * vx + math.sin(heading) * vy
        v_lat = -math.sin(heading) * vx + math.cos(heading) * vy
        w_yaw = float((R @ self.data.qvel[3:6])[2])                # velocidad de giro en el mundo
        return pitch, v_fwd, v_lat, w_yaw, self.data.qpos[0:2].copy()

    # ==================================================================
    # Render
    # ==================================================================
    def render(self):
        if self.render_mode == "rgb_array":
            if self._renderer is None:
                self._renderer = mujoco.Renderer(self.model, height=self._render_h, width=self._render_w)
                self._cam = mujoco.MjvCamera()
                self._cam.type = mujoco.mjtCamera.mjCAMERA_FREE
                self._cam.distance = 0.45
                self._cam.azimuth = 135.0
                self._cam.elevation = -20.0
            self._cam.lookat[:] = self.data.xpos[self._chassis]      # la cámara sigue al robot
            self._renderer.update_scene(self.data, camera=self._cam)
            return self._renderer.render()

        if self.render_mode == "human":
            if self._viewer is None:
                from mujoco import viewer as mj_viewer   # alias local: no reasignar 'mujoco' dentro de la función
                self._viewer = mj_viewer.launch_passive(self.model, self.data)
                self._viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
                self._viewer.cam.distance = 0.6
                self._viewer.cam.azimuth = 135
                self._viewer.cam.elevation = -20
            self._viewer.cam.lookat[:] = self.data.xpos[self._chassis]
            self._viewer.sync()
        return None

    def viewer_is_running(self):
        return self._viewer is None or self._viewer.is_running()

    def close(self):
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None