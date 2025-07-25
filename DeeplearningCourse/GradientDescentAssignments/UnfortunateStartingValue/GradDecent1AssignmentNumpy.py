# import all necessary modules
import numpy as np
import matplotlib.pyplot as plt

# function (as a function)
def fx(x):
  return np.cos(2 * np.pi * x) + x**2

# derivative function
def deriv(x):
  return -2 * np.pi * np.sin(2 * np.pi * x) + 2 * x

# define a range for x
x = np.linspace(-2, 2, 2001)

# plotting initial function and derivative
plt.plot(x, fx(x), x, deriv(x))
plt.xlim(x[[0, -1]])
plt.grid()
plt.xlabel('x')
plt.ylabel('f(x)')
plt.legend(['y', 'dy'])
plt.title('Function and Derivative')
plt.savefig("initial_plot.png")
plt.close()

# random starting point
localmin = np.random.choice(x, 1)

# learning parameters
learning_rate = .01
training_epochs = 100

# run through training
for i in range(training_epochs):
  grad = deriv(localmin)
  localmin = localmin - learning_rate * grad

# plot the results
plt.plot(x, fx(x), x, deriv(x))
plt.plot(localmin, deriv(localmin), 'ro')
plt.plot(localmin, fx(localmin), 'ro')

plt.xlim(x[[0, -1]])
plt.grid()
plt.xlabel('x')
plt.ylabel('f(x)')
plt.legend(['f(x)', 'df', 'f(x) min'])
plt.title('Empirical local minimum: %.5f' % localmin[0])
plt.savefig("localmin_result.png")
plt.close()

# print result
print(f"Final estimated local minimum: x = {localmin[0]:.5f}, f(x) = {fx(localmin)[0]:.5f}")
