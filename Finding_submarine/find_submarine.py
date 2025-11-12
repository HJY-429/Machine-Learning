import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fftn, ifftn, fftshift, ifftshift
import plotly.graph_objs as go
from IPython.display import clear_output

data_path = '/Users/hjy/AMATH/AMATH 582/HW 1/documents/subdata.npy'
data = np.load(data_path)


# NOTE: L we defined in class is 2L here, i.e. the domain here is [-L,L].
L = 10; # length of spatial domain
N = 64; # number of grid points
x2 = np.linspace(-L, L, N + 1) #spatial grid in x
x = x2[0 : N]
y = x 
z = x

# frequency grid for one coordinate
K_grid = (2*np.pi/(2*L)) * np.linspace(-N/2, N/2 - 1, N)
xv, yv, zv = np.meshgrid(x, y, z) # generate 3D meshgrid for plotting
kx, ky, kz = np.meshgrid(K_grid, K_grid, K_grid)


signal_fs = []
for j in range(0,49):
  signal = np.reshape(data[:, j], (N, N, N))
  signal_f = fftn(signal)
  signal_fs.append(fftshift(signal_f))

signal_avg = np.abs(np.mean(signal_fs, axis=0))
normal_sig_avg_abs = np.abs(signal_avg)/np.abs(signal_avg).max()

fig_data = go.Isosurface( x = kx.flatten(), y = ky.flatten(), z = kz.flatten(),
                           value = normal_sig_avg_abs.flatten(), isomin=0.6, isomax=1, showscale=False)
clear_output(wait=True)
fig = go.Figure( data = fig_data )
fig.update_layout(
    title=dict(text="Frequency Domain", font=dict(size=28)),
    scene=dict(xaxis=dict(title=dict(text='KX', font=dict(size=20))),
        yaxis=dict(title=dict(text='KY', font=dict(size=20))),
        zaxis=dict(title=dict(text='KZ', font=dict(size=20)))),
    legend=dict(font=dict(size=20)))
fig.write_image("Frequency Domain.pdf")
fig.show()

idx = np.unravel_index(np.argmax(signal_avg), signal_avg.shape)
center_freq = [kx[idx], ky[idx], kz[idx]]
xc, yc, zc = center_freq[0], center_freq[1], center_freq[2]
print("Dominant Frequency (kx, ky, kz):", xc, yc, zc)

fig_data = go.Isosurface( x = kx.flatten(), y = ky.flatten(), z = kz.flatten(),
                           value = (signal_avg / N).flatten(), isomin=2.5, isomax=3, showscale=False)
clear_output(wait=True)
fig = go.Figure( data = fig_data )
point_x, point_y, point_z = 5.340708, 2.199115, -6.911504
point_value = 0.6493587

# Add a scatter3d trace for the point
fig.add_trace(go.Scatter3d(x=[point_x], y=[point_y], z=[point_z], mode='markers+text',
    marker=dict(size=3, color='red'), text=[f"x: {point_x}, y: {point_y}, z: {point_z} <br> value: {point_value}"],
    textfont=dict(size=12, color='purple'), textposition='top right'))
fig.update_layout(
    title=dict(text="Dominant Frequency", font=dict(size=28)),
    scene=dict(xaxis=dict(title=dict(text='KX', font=dict(size=20))),
        yaxis=dict(title=dict(text='KY', font=dict(size=20))),
        zaxis=dict(title=dict(text='KZ', font=dict(size=20)))),
    legend=dict(font=dict(size=20)))
fig.write_image("Dominant Frequency.pdf")
fig.show()


# Design Gaussian filter around the dominant frequency

# sigma = 1
sigma = 5
# sigma = 10

tau = - 1 / (2 * sigma**2)
filter = np.exp(tau * ((kx-xc)**2 + (ky-yc)**2 + (kz-zc)**2))

xp = np.zeros(49); yp = np.zeros(49); zp = np.zeros(49)
data_clean = []

for j in range(0,49):
  S_fft = fftn(np.reshape(data[:, j], (N, N, N)))
  S_fs = fftshift(S_fft)
  S_filter_t = filter * S_fs
  S_filter_ift = ifftshift(S_filter_t)
  S_filter = ifftn(S_filter_ift)
  data_clean.append(S_filter)
  idx2 = np.unravel_index(np.argmax(np.abs(S_filter)), S_filter.shape)
  xp[j] = xv[idx2]
  yp[j] = yv[idx2]
  zp[j] = zv[idx2]


# Plot 3D trajectory (matplotlib.pyplot)
fig1 = plt.figure()
ax1 = fig1.add_subplot(111, projection='3d')
ax1.plot(xp, yp, zp, label='Trajectory', color='blue', linewidth=3)
ax1.plot(xp[-1], yp[-1], zp[-1], label='Latest Point', marker='*', color='purple', markersize=10)

ax1.set_title('3D Trajectory', font=dict(size=15))
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.set_zlabel('Z')
ax1.legend()
plt.savefig('3D Trajectory.pdf')
plt.show()

# Plot trajectory in x, y coordinates (matplotlib.pyplot)
plt.plot(xp, yp, color='blue', linewidth=3)
plt.plot(xp[-1], yp[-1], marker='*', color='purple', markersize=10)
plt.title('2D Path')
plt.xlabel('x')
plt.ylabel('y')
plt.grid(True)
plt.savefig('2D Path.pdf')
plt.show()


# Plot 3D trajectory (plotly.graph_objs)
fig3D = go.Figure()
fig3D.add_trace(go.Scatter3d(
    x = xp, y = yp, z = zp,
    mode='markers+lines', marker=dict(size=3, color='purple'), line=dict(color='blue', width=5), name='Trajectory'))

fig3D.add_trace(go.Scatter3d(
    x = [xp[-1]], y = [yp[-1]], z = [zp[-1]],
    mode='markers', marker=dict(size=5, color='purple', symbol='x'), name='Latest Point'))

fig3D.update_layout(
    title=dict(text="3D Trajectory of Submarine", font=dict(size=28)),
    scene=dict(xaxis=dict(title=dict(text='X', font=dict(size=20))),
        yaxis=dict(title=dict(text='Y', font=dict(size=20))),
        zaxis=dict(title=dict(text='Z', font=dict(size=20)))),
    legend=dict(font=dict(size=20)))
fig3D.write_image('3D Trajectory graph_objs.pdf')
fig3D.show()

# plot trajectory in x, y coordinates (plotly.graph_objs)
fig2D = go.Figure()
fig2D.add_trace(go.Scatter(x = xp, y = yp,
    mode='markers+lines', marker=dict(size=3, color='purple'), 
    line=dict(color='blue', width=2), name='Trajectory'))

fig2D.add_trace(go.Scatter(x = [xp[-1]], y = [yp[-1]],
    mode='markers', marker=dict(size=6, color='purple', symbol='x'), name='Latest Point'))

fig2D.update_layout(
    title=dict(text="2D Trajectory of Submarine", font=dict(size=28)),
    xaxis=dict(title=dict(text='X', font=dict(size=20))),
    yaxis=dict(title=dict(text='Y', font=dict(size=20))),
    legend=dict(font=dict(size=20)))
fig2D.write_image('2D Trajectory graph_objs.pdf')
fig2D.show()

'''
# print(np.shape(idx))
# print(idx)

# print(np.shape(data_clean))
# print(np.shape(xv.flatten()))

# print(signal_avg)
# print(np.shape(signal_avg))


for j in range(0,49,7):
  fig_data = go.Isosurface( x = xv.flatten(), y = yv.flatten(), z = zv.flatten(), value = np.abs(np.array(data_clean)[j, :]).flatten(), isomin=0.6, isomax=1)
  fig = go.Figure( data = fig_data )
  fig.show()

# print in latex table type
for i in range(1, 25):
  print("{} & {} & {} & {} & {} & {} & {} & {} \\\\".format(i, xp[i-1], yp[i-1], zp[i-1], i+25, xp[i+24], yp[i+24], zp[i+24]))
print("{} & {} & {} & {} \\\\ \\hline ".format(25, xp[24], yp[24], zp[24]))
'''
