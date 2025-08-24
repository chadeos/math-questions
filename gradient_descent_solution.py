import numpy as np
import matplotlib.pyplot as plt

# Example data
X = np.array([1, 2, 3, 4], dtype=float)
y = np.array([3, 5, 7, 9], dtype=float)

# Parameters initialized from standard normal (no seed)
w = np.random.randn()
b = np.random.randn()
alpha = 0.01  # Learning rate
epochs = 1000

n = len(X)

# Store values for plotting
w_history = [w]
b_history = [b]
loss_history = []

for epoch in range(epochs):
    # Compute error vector
    y_pred = w * X + b
    error = y - y_pred
    
    # Compute loss (MSE)
    loss = np.mean(error**2)
    loss_history.append(loss)
    
    # Compute gradients
    dw = -(2/n) * np.dot(X, error)
    db = -(2/n) * np.sum(error)
    
    # Update parameters
    w -= alpha * dw
    b -= alpha * db
    
    # Store history
    w_history.append(w)
    b_history.append(b)

# Plot parameter evolution
plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
plt.plot(w_history, label="w")
plt.plot(b_history, label="b")
plt.xlabel("Epoch")
plt.ylabel("Parameter Value")
plt.title("Evolution of Parameters")
plt.legend()

# Plot loss curve
plt.subplot(1,2,2)
plt.plot(loss_history)
plt.xlabel("Epoch")
plt.ylabel("Loss (MSE)")
plt.title("Loss Curve")

plt.tight_layout()
plt.show()

# Print final values
print(f"Final w: {w:.6f}, Final b: {b:.6f}, Final loss: {loss_history[-1]:.6f}")
