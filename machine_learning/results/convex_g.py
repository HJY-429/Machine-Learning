import numpy as np
import matplotlib.pyplot as plt

def g(x1, x2):
    return (np.sqrt(np.abs(x1)) + np.sqrt(np.abs(x2))) ** 2

x = np.linspace(-6, 6, 801)
y = np.linspace(-6, 6, 801)
X, Y = np.meshgrid(x, y)
G = g(X, Y)

plt.figure(figsize=(6, 6))
# Shade the region g <= 4
plt.contourf(X, Y, G, levels=[-1, 4], cmap="Blues", alpha=0.8)
# Draw the boundary g = 4
contour = plt.contour(X, Y, G, levels=[4], colors='k')
# plt.clabel(contour, fmt='g=4', inline=True, fontsize=9)
A = (-4, 0)
B = (0, 4)
M = ((A[0]+B[0])/2, (A[1]+B[1])/2)
plt.scatter([A[0], B[0], M[0]], [A[1], B[1], M[1]], color='red')
plt.text(-4.2, 0.2, 'A(4,0)')
plt.text(0.2, 4.2, 'B(0,4)')
plt.text(-2.2, 2.2, 'M(-2,2)')
plt.title('$Set \\quad \\{(x_1, x_2): g(x_1, x_2) \\leq 4\\}$', fontdict=dict(size=18))
plt.xlabel('$x_1$', fontdict=dict(size=13))
plt.ylabel('$x_2$', fontdict=dict(size=13))
# plt.legend()
# plt.gca().set_aspect('equal', adjustable='box')
plt.grid(True)
plt.savefig('Figure B1c.pdf')
plt.show()