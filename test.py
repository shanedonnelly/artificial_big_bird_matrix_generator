import numpy as np
import matplotlib.pyplot as plt

# Polynôme de degré 3
def f(x):
    return 3 * x**3 - 2 * x**2 + 0.5 * x  # défini sur [0,1]

# Étape 1 : Échantillonnage
x = np.linspace(0, 1, 200)
y = f(x)

# Étape 2 : inversion des couples (y, x)
# -> y devient l'entrée, x la sortie cible
# Attention : y doit être trié pour polyfit fiable
sort_idx = np.argsort(y)
y_sorted = y[sort_idx]
x_sorted = x[sort_idx]

# Étape 3 : ajustement polynôme de degré n
n = 3 # degré du polynôme inverse approx.
coeffs = np.polyfit(y_sorted, x_sorted, deg=n)

# Créer la fonction polynomiale inverse approx.
f_inv_poly = np.poly1d(coeffs)

# Test visuel
y_test = np.linspace(min(y), max(y), 200)
x_pred = f_inv_poly(y_test)

plt.plot(x, y, label='f(x)', color='blue')
plt.plot(y_test, x_pred, label='approx f⁻¹(y)', color='green')
plt.xlabel("x ou y")
plt.ylabel("y ou x")
plt.legend()
plt.grid()
plt.title("Polynôme et son inverse approché")
plt.show()
