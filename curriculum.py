"""
CURRICULUM
CREADO POR: Mario Betancourt (diseño de las etapas) / implementación con asistencia de Claude
SOBRE EL USO DE I.A. : Las 7 etapas fueron descritas por Mario en el cuadernillo (basadas en la serie
                       de Digikey y Shawn Hymel). Claude las tradujo a una configuración (StageConfig)
                       y añadió una etapa 8 para los comandos externos.

Cada etapa es ACUMULATIVA: la etapa k conserva todo lo activado en las etapas anteriores.
Este archivo solo usa la librería estándar, así que se puede importar sin MuJoCo ni PyTorch.
"""

from dataclasses import dataclass, field, replace
from typing import Tuple, List


@dataclass(frozen=True)
class StageConfig:
    # --- Identificación ---
    index: int
    name: str                 # nombre corto, se usa como carpeta de TensorBoard
    title: str                # título legible

    # --- Condiciones iniciales (reset) ---
    init_pitch_std: float = 0.02          # [rad] desviación estándar del ángulo inicial
    init_pitch_rate_std: float = 0.10     # [rad/s] desviación estándar de la velocidad angular inicial

    # --- Recompensa ---
    penalize_drift: bool = False          # Etapa 2+: penaliza velocidad lineal y yaw (cmd = 0)

    # --- Sensores (Etapa 3) ---
    sensor_noise: bool = False
    gyro_noise_std: float = 0.02          # [rad/s]
    accel_noise_std: float = 0.15         # [m/s^2]
    encoder_noise_std: float = 0.05       # [rad/s]
    imu_position_range: float = 0.0       # [m] +- desplazamiento aleatorio de la IMU en x, y, z
    max_delay_steps: int = 0              # retardo máximo de la acción (pasos de control)

    # --- Perturbaciones mecánicas (Etapa 4) ---
    control_noise_std: float = 0.0        # ruido sobre la señal de control (unidades normalizadas)
    push_prob: float = 0.0                # probabilidad por paso de iniciar un empuje
    push_force_max: float = 0.0           # [N] fuerza máxima del empuje en X o Y
    push_duration_steps: int = 10         # duración de cada empuje (pasos)

    # --- Variación de parámetros físicos (Etapa 5) ---
    randomize_mass: bool = False
    mass_scale_range: Tuple[float, float] = (1.0, 1.0)
    com_shift_range: float = 0.0          # [m] +- desplazamiento del centro de masa del chasis
    randomize_friction: bool = False
    friction_range: Tuple[float, float] = (0.8, 0.8)   # coeficiente de deslizamiento

    # --- Ganancia del motor (Etapa 6) ---
    motor_gain_range: Tuple[float, float] = (1.0, 1.0)

    # --- Pulsos de torque en el eje (Etapa 7) ---
    axle_pulse_prob: float = 0.0
    axle_pulse_torque_max: float = 0.0    # [N m]
    axle_pulse_duration_steps: int = 3

    # --- Comandos externos (Etapa 8) ---
    use_commands: bool = False
    cmd_v_max: float = 0.0                # [m/s] velocidad hacia adelante máxima comandada
    cmd_w_max: float = 0.0                # [rad/s] velocidad de giro (yaw) máxima comandada
    cmd_resample_steps: int = 400         # cada cuántos pasos se cambia el comando (2 s a 200 Hz)
    cmd_zero_prob: float = 0.3            # probabilidad de que el nuevo comando sea (0, 0)

    # --- Presupuesto de entrenamiento (pasos de entorno) ---
    min_timesteps: int = 50_000           # mínimo antes de poder promover la etapa
    max_timesteps: int = 300_000          # presupuesto máximo de la etapa
    promote_length_frac: float = 0.90     # promover si la duración media >= frac * max_steps


def build_stages() -> List[StageConfig]:
    """Devuelve la lista de las 8 etapas (7 de Mario + comandos). Cada una hereda de la anterior."""
    s1 = StageConfig(
        index=1, name="etapa1_balance_basico", title="Balance básico",
        init_pitch_std=0.05, init_pitch_rate_std=0.30,     # aleatorización inicial ligera
        penalize_drift=False,
    )
    s2 = replace(
        s1, index=2, name="etapa2_balance_origen", title="Balance en el origen",
        penalize_drift=True,
    )
    s3 = replace(
        s2, index=3, name="etapa3_ruido_retardos", title="Ruido en sensores y retardos",
        sensor_noise=True, max_delay_steps=3, imu_position_range=0.01,
    )
    s4 = replace(
        s3, index=4, name="etapa4_empujes", title="Empujes y ruido en actuadores",
        control_noise_std=0.05, push_prob=0.002, push_force_max=0.3, push_duration_steps=10,
    )
    s5 = replace(
        s4, index=5, name="etapa5_masa_friccion", title="Variación de masa y fricción",
        randomize_mass=True, mass_scale_range=(0.8, 1.2), com_shift_range=0.004,
        randomize_friction=True, friction_range=(0.4, 1.2),
    )
    s6 = replace(
        s5, index=6, name="etapa6_ganancia_motor", title="Ganancia y tolerancia de voltaje",
        motor_gain_range=(0.6, 1.0),
    )
    s7 = replace(
        s6, index=7, name="etapa7_imperfecciones", title="Imperfecciones mecánicas (tire ridges)",
        axle_pulse_prob=0.01, axle_pulse_torque_max=0.01, axle_pulse_duration_steps=3,
    )
    s8 = replace(
        s7, index=8, name="etapa8_comandos", title="Seguimiento de comandos",
        use_commands=True, cmd_v_max=0.3, cmd_w_max=1.5,
        max_timesteps=500_000,
    )
    return [s1, s2, s3, s4, s5, s6, s7, s8]


STAGES = build_stages()


def get_stage(index: int) -> StageConfig:
    """Obtiene una etapa por su número (1 a 8)."""
    for s in STAGES:
        if s.index == index:
            return s
    raise ValueError(f"Etapa {index} no existe. Etapas válidas: 1..{len(STAGES)}")


if __name__ == "__main__":
    for s in STAGES:
        print(s.index, s.name, "| ruido:", s.sensor_noise, "| delay:", s.max_delay_steps,
              "| push:", s.push_prob, "| masa:", s.randomize_mass, "| motor:", s.motor_gain_range,
              "| pulsos:", s.axle_pulse_prob, "| cmd:", s.use_commands)