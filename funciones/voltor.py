import numpy as np
from scipy.integrate import dblquad

# Parámetros del semitoro
R = 3.0  # Radio mayor
r = 1.0  # Radio menor
z0 = 1.5  # Altura del plano de corte

# Función a integrar
def integrando(phi, r_val):
    # Calculamos el denominador
    denominador = R + r_val * np.cos(phi)
    # Verificamos que z0 esté dentro del rango válido
    if np.abs(z0 / denominador) > 1:
        return 0  # Fuera del rango, contribución nula
    # Calculamos theta_min
    theta_min = np.arcsin(z0 / denominador)
    # Definimos el integrando
    return (R + r_val * np.cos(phi)) * r_val * (np.pi - theta_min)

# Límites de integración para phi (0 a 2pi) y r (0 a r)
phi_limites = (0, 2 * np.pi)
r_limites = (0, r)

# Aproximación numérica de la integral
resultado, error = dblquad(integrando, r_limites[0], r_limites[1], lambda x: phi_limites[0], lambda x: phi_limites[1])

# Mostramos el resultado
print(f"Volumen sobre el plano z = {z0}: {resultado}")
print(f"Error estimado: {error}")