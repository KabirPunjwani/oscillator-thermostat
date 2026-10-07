"""Fourth-order Runge-Kutta stepper (works for one trajectory or a whole batch)."""


def rk4_step(fun, dt, t, z):
    """Advance z by one RK4 step of size dt.

    fun(t, z) must return dz/dt with the same shape as z, so the same stepper
    serves a single state of shape (6,) and a batch of shape (M, 6).
    """
    k1 = fun(t, z)
    k2 = fun(t + dt / 2, z + (dt / 2) * k1)
    k3 = fun(t + dt / 2, z + (dt / 2) * k2)
    k4 = fun(t + dt, z + dt * k3)
    return z + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
