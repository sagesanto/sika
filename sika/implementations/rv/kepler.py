import numpy as np
from orbitize.kepler import _calc_ecc_anom


# adapted from jerry xuan:

# Function to solve Kepler’s equation for eccentric anomaly
def solve_keplers_equation(M, e, tol=1e-10):
    E = M if e < 0.8 else np.pi  # Initial guess
    F = E - e * np.sin(E) - M
    while np.abs(F) > tol:
        E = E - F / (1 - e * np.cos(E))
        F = E - e * np.sin(E) - M
    return E

# Keplerian function
# tau between 0 and 1 as fraction of orbit
def keplerian_function(t, amplitude, period, eccentricity, omega, tau, offset):
    mean_anomaly = 2 * np.pi * (t - tau*period) / period
    mean_anomaly %= 2 * np.pi
    eccentric_anomaly = np.array(_calc_ecc_anom(mean_anomaly,eccentricity,max_iter=1000))
    # eccentric_anomaly = np.array([solve_keplers_equation(M, eccentricity) for M in mean_anomaly])
    true_anomaly = 2 * np.arctan2(np.sqrt(1 + eccentricity) * np.sin(eccentric_anomaly / 2),
                                  np.sqrt(1 - eccentricity) * np.cos(eccentric_anomaly / 2))
    # true_anomaly = approx_true_anomaly(mean_anomaly,eccentricity)
    return amplitude * (np.cos(true_anomaly + omega) + eccentricity * np.cos(omega)) + offset

# Combined fitting function for two datasets
def combined_keplerian(t1, t2, amp1, amp2, period, eccentricity, omega_planet, tau, offset1, offset2, drift):
    t0 = np.min(np.concatenate((t1,t2)))
    y1 = keplerian_function(t1, amp1, period, eccentricity, omega_planet, tau, offset1) + drift * (t1 - t0)
    y2 = keplerian_function(t2, amp2, period, eccentricity, omega_planet + np.pi, tau, offset2) + drift * (t2 - t0)
    return y1, y2