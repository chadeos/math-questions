import numpy as np
import matplotlib.pyplot as plt

# ----- Hard-coded quadratic data (with noise) -----
# True function: y = 2x^2 - 3x + 1
X = np.array([-2, -1, 0, 1, 2], dtype=float)
y = np.array([15.,  6.,  1.,  0.,  3.], dtype=float)

# ----- Parameters for quadratic model -----
# y_hat = a2*x^2 + a1*x + a0
a2 = np.random.randn()
a1 = np.random.randn()
a0 = np.random.randn()

alpha = 0.01
epochs = 2000
n = len(X)

loss_history = []
a2_hist, a1_hist, a0_hist = [a2], [a1], [a0]

for epoch in range(epochs):
    # Calculate error
    # ...

    # Calculate Mean Squared Error
    # ...
    
    loss_history.append(loss)

    # Calculate gradient
    # ...

    # Update parameters
    # ...

    # Store history
    a2_hist.append(a2); a1_hist.append(a1); a0_hist.append(a0)

# ----- Plots -----
plt.figure(figsize=(14,5))

# Parameter evolution
plt.subplot(1,3,1)
plt.plot(a2_hist, label="a2")
plt.plot(a1_hist, label="a1")
plt.plot(a0_hist, label="a0")
plt.xlabel("Epoch"); plt.ylabel("Value")
plt.title("Parameter Evolution")
plt.legend()

# Loss curve
plt.subplot(1,3,2)
plt.plot(loss_history)
plt.xlabel("Epoch"); plt.ylabel("Loss")
plt.title("Loss Curve")

# Fit vs data
plt.subplot(1,3,3)
xx = np.linspace(X.min()-0.5, X.max()+0.5, 200)
yy = a2*xx**2 + a1*xx + a0
plt.scatter(X, y, s=40, label="Data")
plt.plot(xx, yy, 'r-', label="Fit")
plt.xlabel("x"); plt.ylabel("y")
plt.title("Quadratic Fit")
plt.legend()

plt.tight_layout()
plt.show()

print(f"Learned coefficients: a2={a2:.3f}, a1={a1:.3f}, a0={a0:.3f}")
print(f"Final loss: {loss_history[-1]:.6f}")