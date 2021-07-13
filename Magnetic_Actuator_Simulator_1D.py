import magpylib as mp
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve


# Compute the force on a magnetic moment from a coil at a point
def get_coil_force_1d(z, coil, moment, dz, rel):
    # z is distance from the end of the coil in mm
    # rp is rest (unloaded) position of post from the end of the coil at the coil's center
    # moment is axial with the coil
    # rel is relative permeability of coil core
    # returns 1d difference in force on the magnet

    # scalar product of fields and moment at point x
    bz = np.zeros((3, 3))
    bz[0, :] = coil.getB([0, 0, 1e3 * (z - dz)]) #z and dz are converted to just for this calculation
    bz[2, :] = coil.getB([0, 0, 1e3 * (z + dz)])
    bz = bz*1e-3  #mT to T
    return rel * moment * (bz[0, 2] - bz[2, 2]) / 2 / dz #Taylor series derivative approximation


# get the difference between the force on a magnet from the coil and a force on the magnet from the spring to which it's attached
def force_balance_1d(z, coil, moment, rp, dz, k, rel):
    return k*(z-rp) - get_coil_force_1d(z, coil, moment, dz, rel)


def make_Coil(I, wire_dia, layers, wraps, radius):
    # generate coil
    current = []

    # I: current Amps
    # wire_dia: mm, 34 awg
    # layers: how many layers in the coil, zero position is front of coil
    # wraps: how many loops along the axis
    # radius: radius of core, mm

    for wrap in range(0, wraps):
        for layer in range(0, layers):
            current.append(mp.source.current.Circular(curr=I, dim=2 * (radius + layer * wire_dia), pos=[0, 0,
                                                                                                        -wrap * wire_dia]))  # coil starts at z=0 and goes backwards

    return mp.Collection(current)

#Find moment of post magnet
ua = 1.26e-6 # permeability of air H/m
# testm = mp.source.magnet.Box(mag=(0, 0, 1400), dim=(.5, .5, .5), pos=[0,0,-.25])
testm = mp.source.magnet.Cylinder(mag=(0, 0, 1400), dim=(.75, 1), pos=[0, 0, -.5])
M0 = testm.getB([0, 0, 5])[2] / 2 / ua * .005**3 * 4 * np.pi * 1e-3 #A/m**2, simualate magenet and estimate moment, matches well with 1/ua*Br*v
print("Dipole magnitude of permanent magnet")
print(M0)

E = 2e4 #Pa, Young's Modulus of tissue
A = np.pi * (.0005)**2 #m sq.
L = .008 #m Length of tissue
k_Tissue = E*A/L
print("Spring constant of tissue (N/m)")
print(k_Tissue)

### 1d model
# Simplified, at what x is force_func(x) equal to k*(x - x0)

# Model parameters
layers = 5 # layers of wraps
wraps = 10 # wraps along coil
radius = 1 # core radius mm
mu0 = 2e4 # Minimum relaive permeability of 4N iron, H/m
rps = .005 # post resting position, m
wire_d = .21 # coil wire diameter, mm

# Sweep parameters
start = 0
end = 125
step = .00001 #A, increment of coil current
dist = []

for i in range(start, end):
    co = make_Coil(-i*step, wire_d, layers, wraps, radius)
    dist.append(fsolve(force_balance_1d, rps, args=(co, M0, rps, 1e-6, k_Tissue, mu0))) # coil, moment, distance from coil in m, dx in m, spring constant N/m

coil_curr = np.arange(step * start, step * end, step)
dist = np.reshape(dist, len(coil_curr))
disp = dist - rps
plt.plot(coil_curr*1e3, disp) # negative values correspond to  dist < rps, or deflection toward the coil
plt.ylabel("Magnet displacement with post {0} mm away from coil end \n (Starting magnet position - final distance from coil, m)".format(rps*1e3))
plt.xlabel("Coil current (mA)")
plt.grid(True)
plt.show()

print("Maximum Current (A)")
imax = np.argwhere(disp == np.amin(disp))*step
print(imax)

print("Maximum Power (W)")
R = 0
ohms_per_meter = 1.5
for wrap in range(0, wraps):
    for layer in range(0, layers):
        R = R + 2 * np.pi * (radius + layer * wire_d) / 1e3 * ohms_per_meter
print(abs(imax) * R**2)

print("Maximum field at sensor (T)")
print(make_Coil(-imax, wire_d, layers, wraps, radius).getB([3, 0, 3]) * mu0 / 1e3)
