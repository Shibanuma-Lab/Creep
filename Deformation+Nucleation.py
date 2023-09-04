# Imported packages
import numpy as np
import math
import itertools
import os
import pickle
import json

# ImportFunction
with open('n125-id01.tess','r') as f:
    original_data = f.read()

original_data = original_data.split('\n')                # Split data by lines
original_data = [row.strip() for row in original_data]   # Remove whitespace
original_data = [row for row in original_data if row]    # Remove empty rows

data = []
for row in original_data:
    data.append(row)

# Basic information
# Number of grains
nsS = data.index("**cell")
ns = int(data[nsS + 1])

# Length of one period
periodS = data.index("**periodicity")
perdist = data[periodS + 3]
perdist = perdist.split()
perdistx = float(perdist[0])
perdisty = float(perdist[1])
perdistz = float(perdist[2])

#　Center point
centerS = data.index("1  0.000000000000 0.000000000000 0.000000000000 x0y0z0")
centerSS = data[centerS].split()
center = [float(centerSS[1]) + perdistx/2,
          float(centerSS[2]) + perdisty/2,
          float(centerSS[3]) + perdistz/2]

# Extracting nodes
vtxS = data.index("**vertex")
nvtx = int(data[vtxS + 1])
vtx0 = [row.split()[1:4] for row in data[vtxS + 2: vtxS + nvtx + 2]]
vtx = list(map(lambda x, y: [x] + y, range(1,nvtx+1), vtx0))
vtx= [[row[0], float(row[1]), float(row[2]), float(row[3])] for row in vtx]

# Extracting of edges
edgS = data.index("**edge")
nedg = int(data[edgS + 1])
edg = [row.split()[0:3] for row in data[edgS + 2: edgS + nedg + 2]]
edg = [[int(x) for x in sublist] for sublist in edg]

# Extracting faces
fceS = data.index("**face")
nfce = int(data[fceS+1])
fce0 = [data[i:i+4] for i in range(fceS + 2, fceS + 2 + 4 * nfce, 4)]
fce00 = [fce0[i][0].split(" ")[2:] for i in range(nfce)]
fcev = [[x] + y for x, y in zip(range(1, nfce + 1), fce00)]
fce01 = [[abs(float(num)) for num in fce0[i][1].split(" ")[1:]] for i in range(nfce)]
fce = [[x] + y for x, y in zip(range(1, nfce + 1), fce01)]
fce = [[int(x) for x in sublist] for sublist in fce]

# Extracting polygons
plyS = data.index("**polyhedron")
nply = int(data[plyS + 1])
sublist = data[plyS + 2 : plyS + 2 + nply]
modified_sublist = [list(map(int, element.split()))[:1] +
                    list(map(int, element.split()))[2:] for element in sublist]
ply = [[abs(num) for num in sublist_element] for sublist_element in modified_sublist]

# Extracting duplicate elements
# Extracting duplicate nodes
dvtxS = data.index("*vertex", data.index("*vertex") + 1)
ndvtx = int(data[dvtxS + 1])
dvtx = data[dvtxS + 2 : dvtxS + ndvtx + 2]
dvtx = [item.split() for item in dvtx]
dvtx= [[float(row[0]), float(row[1]), float(row[2]), float(row[3]), float(row[4])] for row in dvtx]

# Extracting the duplicate edges
dedgS = data.index("*edge", data.index("*edge") + 1)
ndedg = int(data[dedgS + 1])
dedg0 = [list(map(int, data[i].split())) for i in range(dedgS + 2, dedgS + ndedg + 2)]
dedg = [sublist[:5] for sublist in dedg0]

# Extracting the duplicate faces
dfceS = data.index("*face", data.index("*face") + 1)
ndfce = int(data[dfceS + 1])
dfce0 = [list(map(int, data[i].split())) for i in range(dfceS + 2, dfceS + ndfce + 2)]
dfce = [sublist[:5] for sublist in dfce0]

# Initial parameter settings 1
# Material
dt = 10. * 60. ** 2  # time increment
step = 100  # number of steps

# Variables
T = 923.  # Temperature
# Applied stress
sigma_xx = 0.
sigma_yy = 45. * 10 ** 6
sigma_zz = 0.
sigma_xy = 0.
sigma_yz = 0.
sigma_zx = 0. * 10 ** 6

# Physical constant
kB = 1.380649 * 10 ** -23  # Boltzmann's constant
Rth = 8.3144626181532

# Material constant
delta = 5.0 * 10 ** -10  # grain boundary thickness
Db0 = 7.0 * 10 ** -6  # grain boundary diffusion coefficient at T=0K
Qb = 1.15 * 10 ** 5  # activation energy for grain boundary diffusion
Db = Db0 * math.exp(-Qb / (Rth * T))  # grain boundary diffusion coefficient
CapitalOmega = 1.09 * 10 ** -29  # atomic volume

b = 0.249 * 10 ** -9  # inter-atomic distance
gmma = 0.93  # boundary energy per unit length
lmbd = (7.08 * 10 ** -25 * T) / gmma  # static grain growth constant
Cn = 10. ** 16 / dt  # static grain growth constant

# Initial parameter settings 2
# Material constant
Ds0 = 7.0 * 10 ** -6
delta_s = 5.0 * 10 ** -10  # grain boundary thickness
Qs = 9.67 * 10 ** 4  # activation energy for grain boundary diffusion

gmma_s = 2.49  # surface energy per unit area
Alpha_p = 4. * 10 ** 12
Alpha_p2 = 0.05 * 10 ** 8
nss = 20  # number of subdivisions for Simpson's rule
M1 = 10
M2 = 8
Qtip = 20 + 3 * M2
QL0 = 10 * M1
Qeq = 20 + 3 * M2

# All grain boundaries or specific grain boundaries
calcflag = 1
# calcflag = 0
# calcGB = [11, 267]  # Specify the GBs to be calculated

Ds = Ds0 * math.exp(-Qs / (Rth * T))  # grain boundary diffusion coefficient

# Dependent variables
# Calculate the angle delta_0 between the grain boundary and the cavity
delta_0 = math.acos(gmma / (2 * gmma_s))
h_del_0 = (1 / (1 + math.cos(delta_0)) - (1 / 2) * math.cos(delta_0)) / math.sin(delta_0)
Dgb = (CapitalOmega * delta * Db) / (kB * T)

exaveragediameter = 118.901833921134

with open("rn.json") as f:
    rn = json.load(f)

# Initializing

QP = [None] * (step + 1)
QP27 = [None] * (step + 1)
TL = [None] * (step + 1)
GB = [None] * (step + 1)
Grain = [None] * (step + 1)
nQP = [None] * (step + 1)
nTL = [None] * (step + 1)
nGB = [None] * (step + 1)
nG = [None] * (step + 1)
epsl = [None] * (step + 1)
domain = [None] * (step + 1)
GBS = [None] * (step + 1)
GBG = [None] * (step + 1)
GBNV = [None] * (step + 1)
vtxc = [None] * (step + 1)
Q = [None] * (step + 1)
v_del_mag = [None] * step

sigma = np.array([[sigma_xx, sigma_xy, sigma_zx],
                  [sigma_xy, sigma_yy, sigma_yz],
                  [sigma_zx, sigma_yz, sigma_zz]])

QP[0] = [
    [
        vtx[i][1] % perdistx + (center[0] - perdistx / 2),
        vtx[i][2] % perdisty + (center[1] - perdisty / 2),
        vtx[i][3] % perdistz + (center[2] - perdistz / 2)
    ]
    for i in range(len(vtx))
    if vtx[i][0] not in [val[0] for val in dvtx]
        ]

nQP[0] = len(QP[0])

perx = [-1.0, 0.0, 1.0, -1.0, 0.0, 1.0, -1.0, 0.0, 1.0, -1.0, 0.0, 1.0, -1.0, 0.0, 1.0, -1.0, 0.0, 1.0, -1.0, 0.0,
            1.0, -1.0, 0.0, 1.0, -1.0, 0.0, 1.0]
pery = [-1.0, -1.0, -1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, -1.0, -1.0,
            -1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0]
perz = [-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0,
            1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
domain = [None] * (step + 1)
domain[0] = [[perdistx, 0.0, 0.0], [0.0, perdisty, 0.0], [0.0, 0.0, perdistz]]

QP270_temp = [[np.array(domain[0][0]) * x + np.array(domain[0][1]) * y + np.array(domain[0][2]) * z] for x, y, z in zip(perx, pery, perz)]
QP27[0] = np.array(QP[0]) + QP270_temp


dvtx2_1 = [[sublist[0], dvtx[next(index for index, val in enumerate(dvtx)
        if val[0] == sublist[0])][1] if any(val[0] == sublist[0] for val in dvtx) else sublist[0]
        ] for sublist in vtx]
dvtx2_2 = [[(sublist[1] // perdistx) + (center[0] - perdistx / 2), ] for sublist in vtx]
dvtx2_3 = [[(sublist[2] // perdisty) + (center[1] - perdisty / 2), ] for sublist in vtx]
dvtx2_4 = [[(sublist[3] // perdistz) + (center[2] - perdistz / 2)] for sublist in vtx]
dvtx2 = [elem1 + elem2 + elem3 + elem4 for elem1, elem2, elem3, elem4 in zip(dvtx2_1, dvtx2_2, dvtx2_3, dvtx2_4)]

#medg1 = [vtx[int(edge[1])][1:3] for edge in edg]

medg_1 = [int(edg[i][1]) for i in range(len(edg))]
medg_2 = [vtx[medg_1[i] - 1][1:4] for i in range(len(medg_1))]    # needed
medg_3 = [int(edg[i][2]) for i in range(len(edg))]
medg_4 = [vtx[medg_3[i] - 1][1:4] for i in range(len(medg_1))]    # needed
medg = [[(x + y) / 2 for x, y in zip(sublist1, sublist2)] for sublist1, sublist2 in zip(medg_2, medg_4)]

dedg2_1 = [[sublist[0], dedg[next(index for index, val in enumerate(dedg)
        if val[0] == sublist[0])][1] if any(val[0] == sublist[0] for val in dedg) else sublist[0]
        ] for sublist in edg]
dedg2_2 = [[(sublist[0] // perdistx) + (center[0] - perdistx / 2), ] for sublist in medg]
dedg2_3 = [[(sublist[1] // perdisty) + (center[1] - perdisty / 2), ] for sublist in medg]
dedg2_4 = [[(sublist[2] // perdistz) + (center[2] - perdistz / 2)] for sublist in medg]
dedg2 = [elem1 + elem2 + elem3 + elem4 for elem1, elem2, elem3, elem4 in zip(dedg2_1, dedg2_2, dedg2_3, dedg2_4)]

mfce = []
decimal = 16
rest_fce = [fce[i][1 :] for i in range(len(fce))]
for i in range(len(rest_fce)):
    temp_list = []
    for j in range(len(rest_fce[i])):
        x = rest_fce[i][j]
        y = medg[int(x) - 1][0 : 3]
        temp_list.append(y)
    ave = list(np.round(np.sum(temp_list, axis = 0)/len(rest_fce[i]), decimal))
    mfce.append(ave)

dfce2_1 = [[sublist[0], dfce[next(index for index, val in enumerate(dfce)
        if val[0] == sublist[0])][1] if any(val[0] == sublist[0] for val in dfce) else sublist[0]
        ] for sublist in fce]
dfce2_2 = [[(sublist[0] // perdistx) + (center[0] - perdistx / 2), ] for sublist in mfce]
dfce2_3 = [[(sublist[1] // perdisty) + (center[1] - perdisty / 2), ] for sublist in mfce]
dfce2_4 = [[(sublist[2] // perdistz) + (center[2] - perdistz / 2)] for sublist in mfce]
dfce2 = [elem1 + elem2 + elem3 + elem4 for elem1, elem2, elem3, elem4 in zip(dfce2_1, dfce2_2, dfce2_3, dfce2_4)]



dvtx_0 = [sublist[0] for sublist in dvtx]
def f1(j):
    return j - sum(1 if x <= j else 0 for x in dvtx_0)

def changevtx(i):
    return [14 + 1 * dvtx2[i][2] + 3 * dvtx2[i][3] + 9 * dvtx2[i][4] , f1(dvtx2[i][1])]

edg_0_ = [edg[i][1:] for i in range(len(edg))]
def TLf(i):
    change_val_1 = changevtx(edg_0_[i][0] - 1)
    change_val_2 = changevtx(edg_0_[i][1] - 1)
    out_list = []
    out_list.append(list(np.array(change_val_1) + np.array([-1*dedg2[i][2] - 3*dedg2[i][3] - 9*dedg2[i][4] , 0])))
    out_list.append(list(np.array(change_val_2) + np.array([-1*dedg2[i][2] - 3*dedg2[i][3] - 9*dedg2[i][4] , 0])))
    return out_list

edg_all_1 = [edg[i][0] for i in range(len(edg))]
dedg_all_1 = [dedg[i][0] for i in range(len(dedg))]
TL_comp = list(set(edg_all_1).difference(set(dedg_all_1)))
TL[0] = [TLf(TL_comp[i] - 1) for i in range(len(TL_comp))]
TL[0] = [[[int(element) for element in sublist] for sublist in sublist_list] for sublist_list in TL[0]]
nTL[0] = len(TL[0])


dedg_0 = [sublist[0] for sublist in dedg]
def f2(j):
    return j - sum(1 if x <= j else 0 for x in dedg_0)

def changeedg(i):
    return [14 + 1 * dedg2[i][2] + 3 * dedg2[i][3] + 9 * dedg2[i][4] , f2(dedg2[i][1])]

fce_0_ = [fce[i][1:] for i in range(len(fce))]
def GBf(i):
    change_val_ =[changeedg(fce_0_[i][j] - 1) for j in range(len(fce_0_[i]))]
    out_list = []
    out_list.append(list(np.array(change_val_) + np.array([-1*dfce2[i][2] - 3*dfce2[i][3] - 9*dfce2[i][4] , 0])))
    out_list = [list(arr) for arr in out_list[0]]
    return out_list

fce_all_1 = [fce[i][0] for i in range(len(fce))]
dfce_all_1 = [dfce[i][0] for i in range(len(dfce))]
GB_comp = list(set(fce_all_1).difference(set(dfce_all_1)))
GB[0] = [GBf(GB_comp[i] - 1) for i in range(len(GB_comp))]
nGB[0] = len(GB[0])


dfce_0 = [sublist[0] for sublist in dfce]
def f3(j):
    return j - sum(1 if x <= j else 0 for x in dfce_0)

def changefce(i):
    return [14 + 1 * dfce2[i][2] + 3 * dfce2[i][3] + 9 * dfce2[i][4] , f3(dfce2[i][1])]

ply_0_ = [ply[i][1:] for i in range(len(ply))]
def Grainf(i):
    change_val_ = [changefce(ply_0_[i][j] - 1) for j in range(len(ply_0_[i]))]
    out_list = []
    out_list.append(list(np.array(change_val_)))
    out_list = [list(arr) for arr in out_list[0]]
    return out_list

ply_all_1 = [ply[i][0] for i in range(len(ply))]

Grain[0] = [Grainf(ply_all_1[i] - 1) for i in range(len(ply_all_1))]
Grain[0] = [[[int(num) for num in sublist] for sublist in outerlist] for outerlist in Grain[0]]

nG[0] = len(Grain[0])

epsl[0] = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]


def QPnextf(i):
    j = [[k, l] for k in range(len(TL[0])) for l in range(len(TL[0][k])) if TL[0][k][l][1] == i]

    def f(k, l):
        return [TL[0][k][3 - l - 2][0] + 14 - TL[0][k][l][0], TL[0][k][3 - l - 2][1]]

    result = list(map(lambda x: f(x[0], x[1]), j))
    return result

QPnext = [QPnextf(i) for i in range(1 , 1 + nQP[0])]


def QPTLf(i):
    j = [[k, l] for k in range(len(TL[0])) for l in range(len(TL[0][k])) if TL[0][k][l][1] == i]
    result = [[14 + 14 - TL[0][item[0]][item[1]][0], item[0]] for item in j]
    return result

QPTL = [QPTLf(i) for i in range(1, 1 + nQP[0])]
QPTL = [[list(map(int, sublist)) for sublist in sublist_list] for sublist_list in QPTL]


# Less one than Mathematica
def TLnextf(i):
    def f(pair):
        k, l = pair
        return [item for item in map(lambda x: [x[0] - (14 - k), x[1]], QPTL[l - 1]) if item != [14, i]]

    return [item for sublist in map(f, TL[0][i]) for item in sublist]

TLnext = [TLnextf(i) for i in range(nTL[0])]


# Less one than Mathematica
def TLGBf(i):
    return [[14 + 14 - GB[0][pair[0]][pair[1]][0], pair[0]] for pair in [[k, l] for k in range(len(GB[0]))
            for l in range(len(GB[0][k])) if GB[0][k][l][1] == i]]

TLGB = [TLGBf(i) for i in range(1, 1 + nTL[0])]

# Modification
QPTL = [[[x[0], x[1]+1] for x in lst] for lst in QPTL]
TLnext = [[[x[0], x[1]+1] for x in lst] for lst in TLnext]
TLGB = [[[x[0], x[1]+1] for x in lst] for lst in TLGB]


def QPGBf(i):
    def f(pair):
        k, l = pair
        return [(x[0] - (14 - k), x[1]) for x in TLGB[l - 1]]
    return sorted(list(set([tuple(x) for sublist in [f(pair)
            for pair in QPTL[i]]
            for x in sublist])), key=lambda x: x[1])

QPGB = [QPGBf(i) for i in range(nQP[0])]


def getTL(GBp, GBn):
    return [[item[0] - (14 - GBp), item[1]] for item in GB[0][GBn]]

def getQP(TLp, TLn):
    return [[item[0] - (14 - TLp), item[1]] for item in TL[0][TLn - 1]]


GBQP = [[getQP(int(getTL(14, i)[k][0]), int(getTL(14, i)[k][1])) for k in range(len(getTL(14, i)))]
        for i in range(nGB[0])]


def GBnextf(i):
    def f(lm, k):
        n = []
        for pos1 in range(len(GB[0])):
            for pos2 in range(len(GB[0][pos1])):
                if GB[0][pos1][pos2][1] == lm[1] and (pos1, pos2) != (i, k):
                    n.append([pos1, pos2])
        return [[14 + lm[0] - GB[0][pos[0]][pos[1]][0], pos[0] + 1] for pos in n]

    return [item for sublist in [f(lm, k) for lm, k in zip(GB[0][i], range(len(GB[0][i])))] for item in sublist]

GBnext = [GBnextf(i) for i in range(nGB[0])]


def GBgrainf(i):
    positions = []
    for pos1 in range(len(Grain[0])):
        for pos2 in range(len(Grain[0][pos1])):
            if Grain[0][pos1][pos2][1] == i:
                positions.append([pos1, pos2])
    return [[14 + 14 - Grain[0][pos[0]][pos[1]][0], pos[0] + 1] for pos in positions]

GBgrain = [GBgrainf(i) for i in range(1, 1 + nGB[0])]


def Gnextf(i):
    def f(lm, k):
        n = [[pos[0], pos[1]]
            for pos in [(pos1, pos2)
                for pos1 in range(len(Grain[0]))
                for pos2 in range(len(Grain[0][pos1]))
                if Grain[0][pos1][pos2][1] == lm[1] and (pos1, pos2) != (i, k)]]
        return [[14 + lm[0] - Grain[0][pos[0]][pos[1]][0], pos[0] + 1]
            for pos in n]

    return [f(lm, k) for lm, k in zip(Grain[0][i], range(len(Grain[0][i])))]

Gnext = [[item[0] for item in sublist] for sublist in [Gnextf(i) for i in range(nG[0])]]

def norm(t):
    return math.sqrt(sum(map(lambda x: x**2, t)))

def trianglearea(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    return 0.5 * norm(np.cross(a - c, b - c))

def polygonarea(x):
    G1 = np.array(x).flatten()
    G2 = [G1[i:i + 3] for i in range(0, len(G1), 3)]
    G2 = [G2.tolist() for G2 in G2]
    G = np.mean(G2, axis=0)
    return sum(map(lambda p: trianglearea(p[0], p[1], G), x))

def polygonG(x):
    g1 = np.array(x).flatten()
    g2 = [g1[i:i + 3] for i in range(0, len(g1), 3)]
    g2 = [g2.tolist() for g2 in g2]
    g = np.mean(g2, axis=0)
    return sum(map(lambda p: trianglearea(g, p[0], p[1])/polygonarea(x)*(g+p[0]+p[1])/3, x))


def polygonNV(x):
    X1 = np.array(x).flatten()
    X2 = [X1[i:i + 3] for i in range(0, len(X1), 3)]
    X3 = [X2.tolist() for X2 in X2]
    X4 = sorted(set(tuple(row) for row in X3), key=lambda x: x[0])
    X = [list(row) for row in X4]
    if len(X) == 3:
        return np.cross(np.array(X[1])-np.array(X[0]), np.array(X[2])-np.array(X[0]))/np.linalg.norm(np.cross(np.array(X[1])-np.array(X[0]), np.array(X[2])-np.array(X[0])))
    else:
        Y = [[x * 100000 for x in sublist] for sublist in X]
        Sigma_x = [sum(column) for column in list(zip(*Y))][0]
        Sigma_y = [sum(column) for column in list(zip(*Y))][1]
        Sigma_z = [sum(column) for column in list(zip(*Y))][2]
        Sigma_x2 = sum([y[0]**2 for y in Y])
        Sigma_y2 = sum([y[1]**2 for y in Y])
        Sigma_z2 = sum([y[2]**2 for y in Y])
        Sigma_xy = sum([x[0]*x[1] for x in Y])
        Sigma_xz = sum([x[0]*x[2] for x in Y])
        Sigma_yz = sum([x[1]*x[2] for x in Y])
        A = np.array([[Sigma_x2, Sigma_xy, Sigma_xz], [Sigma_xy, Sigma_y2, Sigma_yz], [Sigma_xz, Sigma_yz, Sigma_z2]])
        b = np.array([Sigma_x, Sigma_y, Sigma_z])
        t = np.linalg.pinv(A).dot(b)
        return t/norm(t)

# Static grain growth
def static(o, X):
    A = []
    if len(X) == 4:
        A = [[1, 2], [1, 3], [1, 4], [2, 3], [2, 4], [3, 4]]
    else:
        A = []
        for i in range(len(X)):
            for j in range(i + 1, len(X)):
                if i <= j:
                    A.append([i+1, j+1])
                else:
                    A.append([j+1, i+1])
        A = [pair for pair in A if pair not in [[i, i] for i in range(len(X))]]

    adj = [[X[i-1] for i in pair] for pair in A]
    Sadj = [trianglearea(pair[0], pair[1], o) for pair in adj]
    Nadj = [np.cross((np.array(item[0]) - np.array(o)), (np.array(item[1]) - np.array(o))) / np.linalg.norm(np.cross((np.array(item[0]) - np.array(o)), (np.array(item[1]) - np.array(o)))) for item in adj]

    def G(x, y, z):
        def sf(pi, pj):
            return 0.5 * np.linalg.norm(np.cross((np.array(pi) - np.array([x, y, z])), (np.array(pj) - np.array([x, y, z]))))

        return gmma * sum([sf(*pair) for pair in adj])

    h = 1e-10    # Set an infinitesimal
    n = np.array([
        -((G(o[0] + h, o[1], o[2]) - G(o[0], o[1], o[2])) / h),
        -((G(o[0], o[1] + h, o[2]) - G(o[0], o[1], o[2])) / h),
        -((G(o[0], o[1], o[2] + h) - G(o[0], o[1], o[2])) / h),
        1
                ])    # Set the minus signal latter
    n = n / np.linalg.norm(n)
    #n = [float(n[0]), float(n[1]), float(n[2]), float(n[3])]

    S = sum(Sadj)
    Sr = sum(s * np.abs(np.dot(n_adj, [n[0], n[1], n[2]]) / (np.linalg.norm(n_adj) * np.linalg.norm([n[0], n[1], n[2]])))
             for s, n_adj in zip(Sadj, Nadj))

    delta_G = lmbd * gmma * np.sqrt(n[0] ** 2 + n[1] ** 2 + n[2] ** 2) / n[3]
    delta_V = Cn * S * b * (1 - np.exp(-(b ** 2 / (kB * T)) * delta_G / S))

    delta_X = delta_V / Sr
    v = [n[0], n[1], n[2]] / np.linalg.norm([n[0], n[1], n[2]]) * delta_X

    return v

def base(vector):
    u, v, w = vector
    if w != 0. and (u != 0. or v != 0.):
        return [[-v, u, 0.] / np.linalg.norm([-v, u, 0.]),
                [-u, -v, (u ** 2 + v ** 2) / w] / np.linalg.norm([-u, -v, (u ** 2 + v ** 2) / w])]
    elif v != 0. and (u != 0. or w != 0.):
        return [[-v, 0., w] / np.linalg.norm([-v, 0., w]),
                [-u, (u ** 2 + w ** 2) / v, -w] / np.linalg.norm([-u, (u ** 2 + w ** 2) / v, -w])]
    elif u != 0. and (v != 0. or w != 0.):
        return [[0., -v, w] / np.linalg.norm([0., -v, w]),
                [(v ** 2 + w ** 2) / u, -v, -w] / np.linalg.norm([(v ** 2 + w ** 2) / u, -v, -w])]
    elif w != 0.:
        return [[0, 1, 0], [1, 0, 0]]
    elif v != 0.:
        return [[0, 0, 1], [1, 0, 0]]
    else:
        return [[0, 0, 1], [0, 1, 0]]

def complex(p, o, e):
    e_inv = np.linalg.pinv(e)
    result = np.dot(np.array(p) - np.array(o), e_inv)
    return result[:2]

def TransformCoordinate(x, ebefore, eafter):
    return np.dot(np.dot(x, ebefore), np.linalg.inv(eafter))

def getcoor(QPp, QPn):
    return QP27[0][QPp-1][QPn-1]

def GBcoorf(GBn):
    return [[getcoor(item[0], item[1]) for item in sublist] for sublist in GBQP[GBn]]

GBcoor = [GBcoorf(GBn) for GBn in range(nGB[0])]

GBS[0] = [polygonarea(item) for item in GBcoor]

GBG[0] = [polygonG(item) for item in GBcoor]

GBNV[0] = [polygonNV(item) for item in GBcoor]

def vtxcf(x, g, n):
    re, im = base(n)
    return [[complex(item, g, [re, im, n]) for item in sublist] for sublist in x]

vtxc[0] = [vtxcf(GBcoor[i], GBG[0][i], GBNV[0][i]) for i in range(len(GBcoor))]


# Cavitation
cavity = [None] * (step + 1)

omega_cavity = [None] * (step + 1)

omega_all = [None] * (step + 1)

ncavity = [None] * (step + 1)

calcrecord = [None] * (step + 1)

cavity[0] = [[] for _ in range(nGB[0])]

cavity[1] = [[] for _ in range(nGB[0])]

omega_cavity[0] = [0.0] * nGB[0]

ncavity[0] = [0.0] * nGB[0]

omega_all[0] = 0.0

if calcflag == 1:
    calc = [1] * nGB[0]
else:
    calc = [0] * nGB[0]


calcrecord[0] = calc

GBnextStatus = [[[x, 0, 0.0] for x in lst] for lst in GBnext]


def plfoot(x, p1, p2):
    x = np.array(x)
    p1 = np.array(p1)
    p2 = np.array(p2)
    t = (np.dot(p1 - x, p1 - x) - np.dot(p1 - x, p2 - x)) / \
        (np.dot(p1 - x, p1 - x) - 2 * np.dot(p1 - x, p2 - x) + np.dot(p2 - x, p2 - x))
    return p1 + t * (p2 - p1)


def pldist(x, p1, p2):
    x = np.array(x)
    p1 = np.array(p1)
    p2 = np.array(p2)
    t = (np.dot(p1 - x, p1 - x) - np.dot(p1 - x, p2 - x)) / \
        (np.dot(p1 - x, p1 - x) - 2 * np.dot(p1 - x, p2 - x) + np.dot(p2 - x, p2 - x))

    if 0 <= t <= 1:
        return np.linalg.norm(p1 + t * (p2 - p1) - x)
    elif t < 0:
        return np.linalg.norm(p1 - x)
    else:
        return np.linalg.norm(p2 - x)


nucleationprobability = np.zeros((nGB[0], 2))
rni = 0

aini = (4 * (1 - math.cos(delta_0)) * gmma_s - (math.sin(delta_0)) ** 2 * gmma) * math.sin(delta_0) / (
                (2 + math.cos(delta_0)) * (1 - math.cos(delta_0)) ** 2
                * ((sigma[0][0] + sigma[1][1] + sigma[2][2]) / 3))

deleteV = (2/3) * math.pi * (aini / math.sin(delta_0))**3 * (math.cos(delta_0) - 1)**2 * (math.cos(delta_0) + 2)
smallVCalc = (2/3) * math.pi * (math.sin(delta_0) / ((1 - math.cos(delta_0)) * math.sin(delta_0)))**3 * (math.cos(delta_0) - 1)**2 * (math.cos(delta_0) + 2)
hCalc = (3 * math.tan(delta_0)**2) / (2 * math.pi) * (math.sin(math.pi / 2 - delta_0)**2) / ((math.sin(math.pi / 2 - delta_0) + 2) * (math.sin(math.pi / 2 - delta_0) + 1))

sin_term = math.sin(math.pi / 2 - delta_0)
cos_term = math.cos(math.pi / 2 - delta_0)
aCalc = (2 * math.pi * math.tan(delta_0)) / 3 * (sin_term * (sin_term**3 - 3 * sin_term + 2)) / (cos_term**4)

thetaCalc = 3 / (2 * math.pi * math.tan(delta_0))
aratefConst = (Ds * delta_s) / (kB * T)

check = 0

def execution(is_):
    global GBcoor, rni, check

    # Deformation

    # Static grain growth
    vst = list(map(static, QP[is_], [[QP27[is_][i[0] - 1][i[1] - 1] for i in sublist] for sublist in QPnext]))

    # Dynamic grain growth
    # Define GB parameters
    TLlength = [np.linalg.norm(np.array(QP27[is_][i[0] - 1][i[1] - 1]) - np.array(QP27[is_][j[0] - 1][j[1] - 1]))
                for i, j in TL[is_]]

    TLc1 = [[QP27[is_][elem[0] - 1][elem[1] - 1] for elem in sublist] for sublist in TL[is_]]
    TLcenter = [arr.tolist() for arr in [np.mean(np.array(TLc1), axis=1)]][0]

    GBcenter_temp = [np.array(GBcoor[i]) for i in range(len(GBcoor))]
    GBcenter = [np.mean(np.reshape(GBcenter_temp[i], (-1, GBcenter_temp[i].shape[-1])), axis=0) for i in
                range(len(GBcenter_temp))]
    GBcenter = [list(arr) for arr in GBcenter]

    def Qf(i):
        def g0(k, j):
            k = int(k)
            j = int(j)
            term_GB = [[pair[0] + (k - 14), pair[1]] for pair in GB[is_][j-1]]
            intersection = [pair for pair in GB[is_][i] if pair in term_GB]
            tl = intersection[0]
            l = TLlength[int(tl[1]) - 1]
            Gi = GBcenter[i]
            M = np.array(TLcenter[int(tl[1]) - 1]) + np.array(domain[is_][0]) * perx[int(tl[0]) - 1] \
                + np.array(domain[is_][1]) * pery[int(tl[0]) - 1] + np.array(domain[is_][2]) * perz[int(tl[0]) - 1]

            Gj = np.array(GBcenter[j-1]) + np.array(domain[is_][0]) * perx[k-1] + np.array(domain[is_][1]) * pery[k-1] + np.array(domain[is_][2]) * perz[k-1]

            term1 = (Db * delta * np.dot(GBNV[is_][i], np.dot(sigma, GBNV[is_][i]))
                     if calc[i] != 3 else Ds * delta_s * np.dot(GBNV[is_][i], np.dot(sigma, GBNV[is_][i])))
            term2 = (Db * delta * np.dot(GBNV[is_][j-1], np.dot(sigma, GBNV[is_][j-1]))
                     if calc[j-1] != 3 else Ds * delta_s * np.dot(GBNV[is_][j-1], np.dot(sigma, GBNV[is_][j-1])))

            return -((l / (kB * T)) * 1 / (np.linalg.norm(Gi - M) + np.linalg.norm(M - Gj))) * (term1 - term2)

        return [g0(item[0], item[1]) for item in GBnext[i]]

    Q[is_] = [Qf(i) for i in list(range(nGB[is_]))]

    def v_del_f(i):
        return (-CapitalOmega * np.sum(Q[is_][i])) / GBS[is_][i]

    # Magnitudes of relative GB velocities
    v_del_mag[is_] = [v_del_f(item) for item in list(range(nGB[is_]))]

    epsl_rate = sum((v_del_mag[is_][i] * GBS[is_][i] * np.outer(GBNV[is_][i], GBNV[is_][i]))
                    / (domain[0][0][0] * domain[0][1][1] * domain[0][2][2])
                    for i in range(len(v_del_mag[is_])))  # They have the same length

    # Define grain parameters
    GGBcenter = [[np.array(GBcenter[pair[1] - 1]) + (np.array(domain[is_][0]) * perx[pair[0] - 1]
                + np.array(domain[is_][1]) * pery[pair[0] - 1] + np.array(
                domain[is_][2]) * perz[pair[0] - 1])
                  for pair in sublist] for sublist in Grain[is_]]
    GGBcenter = [[list(arr) for arr in sublist] for sublist in GGBcenter]

    Gcenter = [np.mean(sublist, axis=0) for sublist in GGBcenter]
    Gcenter = [array.tolist() for array in Gcenter]

    def GGBNVf(i):
        n = [GBNV[is_][pair[1] - 1] for pair in Grain[is_][i]]
        return [pair[0] if np.dot(pair[0], (np.array(pair[1]) - np.array(Gcenter[i]))) >= 0 else -pair[0]
                for pair in zip(n, GGBcenter[i])]

    GGBNV = [GGBNVf(item) for item in list(range(nG[is_]))]

    # Calculate velocities
    Gv_del_mag = [[v_del_mag[is_][pair[1] - 1] for pair in sublist] for sublist in Grain[is_]]

    # Relative GB velocities of each grain toward outside grains
    Gv_del = [ elem1 * np.transpose(elem2) for elem1, elem2 in zip(Gv_del_mag, GGBNV)]
    Gv_del = [np.transpose(item) for item in Gv_del]

    GGBS = [[GBS[is_][pair[1] - 1] for pair in sublist] for sublist in Grain[is_]]

    def GGBwaitf(i):
        Stotal = sum(GGBS[i])

        def f(x):
            return x / Stotal * len(Grain[is_][i])

        return [f(item) for item in GGBS[i]]

    GGBwait = [GGBwaitf(item) for item in list(range(nG[is_]))]

    # Function for calculating coeficients of equations
    def kf(i):
        n = len(Gnext[i])
        kk = [0.0] * nG[is_]
        Gnext_temp1 = [sublist[1] for sublist in Gnext[i]]
        if len(Gnext_temp1) == len(sorted(np.unique(Gnext_temp1))):
            for idx, val in zip(Gnext_temp1, GGBwait[i]):
                kk[idx - 1] = -val
        else:
            for j in range(len(Gnext_temp1)):
                kk[Gnext[i][j][1] - 1] = kk[Gnext[i][j][1] - 1] - GGBwait[i][j]
        kk[i] = n
        return kk

    # Coefficients of equations
    k0 = [kf(item) for item in list(range(nG[is_]))]

    # Grain number for reference speed
    dropn = 1

    k = [k0[i][1:] for i in range(len(k0))][1:]

    Ik = np.linalg.inv(k)

    Gv_del_s1 = [np.transpose(a) * b for a, b in zip(Gv_del, GGBwait)]
    Gv_del_s2 = [np.transpose(item) for item in Gv_del_s1][1:]
    Gv_del_s3 = [np.sum(array, axis=0) for array in Gv_del_s2]
    Gv_del_s = np.vstack(Gv_del_s3).T

    # Total of adjacent GB velocities, Gvave
    Gvave1 = Gnext[dropn:]
    Gvave2 = GGBwait[dropn:]
    Gvave_sum1 = []
    for i in range(len(Gvave1)):
        Gvave_sum2 = []
        for j in range(len(Gvave1[i])):
            val = perx[Gvave1[i][j][0] - 1] * (domain[is_][0][0] * epsl_rate[0] + domain[is_][0][1] * epsl_rate[1]
                                             + domain[is_][0][2] * epsl_rate[2]) * Gvave2[i][j] \
                + pery[Gvave1[i][j][0] - 1] * (domain[is_][1][0] * epsl_rate[0] + domain[is_][1][1] * epsl_rate[1]
                                             + domain[is_][1][2] * epsl_rate[2]) * Gvave2[i][j] \
                + perz[Gvave1[i][j][0] - 1] * (domain[is_][2][0] * epsl_rate[0] + domain[is_][2][1] * epsl_rate[1]
                                             + domain[is_][2][2] * epsl_rate[2]) * Gvave2[i][j]
            Gvave_sum2.append(val)
        Gvave_sum1.append(Gvave_sum2)
    sum2 = [sum(sublist) for sublist in Gvave_sum1]
    Gvave = np.transpose(sum2)


    Gv1 = np.dot(Ik, [np.array(Gv_del_s) + np.array(Gvave)][0][0])
    Gv2 = np.dot(Ik, [np.array(Gv_del_s) + np.array(Gvave)][0][1])
    Gv3 = np.dot(Ik, [np.array(Gv_del_s) + np.array(Gvave)][0][2])
    Gv = np.transpose([Gv1, Gv2, Gv3]).tolist()
    Gv.insert(dropn-1, [0, 0, 0])


    # Dynamic grain growth
    def dynamic(Go, Gi, GBFi):
        Gceo = np.array(Gcenter[int(Go[1])-1]) + np.array((
                    domain[is_][0] * np.array(perx[int(Go[0])-1]) + domain[is_][1] * np.array(pery[int(Go[0])-1]) + domain[is_][2] * np.array(perz[int(Go[0])-1])))
        Gcei = np.array(Gcenter[Gi[1]-1]) + np.array((
                    domain[is_][0] * np.array(perx[int(Gi[0])-1]) + domain[is_][1] * np.array(pery[int(Gi[0])-1]) + domain[is_][2] * np.array(perz[int(Gi[0])-1])))
        M = GBcenter[GBFi]    # the center of the GBFi

        vGBF_map = [getQP(int(x[0]), int(x[1])) for x in getTL(14, GBFi)]
        vGBF_flatten = [item for sublist in vGBF_map for item in sublist]
        vGBF_union = [list(x) for x in set(tuple(x) for x in vGBF_flatten)]
        mapped_elements = [vst[element[1] - 1] for element in vGBF_union]
        vGBF1 = sum(mapped_elements)
        vGBF = vGBF1 / len(GB[is_][GBFi])
         #vst is constant for zero strain, regardless of the periodic boundary conditions.

        if np.dot(vGBF, (Gceo - M)) >= 0:
            high = Go
            low = Gi
        else:
            high = Gi
            low = Go

        vhigh = np.array(Gv[high[1] - 1]) + np.array(
            (perx[high[0] - 1] * (domain[is_][0][0] * epsl_rate[0] + domain[is_][0][1] * epsl_rate[1]
                                + domain[is_][0][2] * epsl_rate[2])
           + pery[high[0] - 1] * (domain[is_][1][0] * epsl_rate[0] + domain[is_][1][1] * epsl_rate[1]
                                + domain[is_][1][2] * epsl_rate[2])
           + perz[high[0] - 1] * (domain[is_][2][0] * epsl_rate[0] + domain[is_][2][1] * epsl_rate[1]
                                + domain[is_][2][2] * epsl_rate[2])))
        vlow = np.array(Gv[low[1] - 1]) + np.array(
            (perx[low[0] - 1] * (domain[is_][0][0] * epsl_rate[0] + domain[is_][0][1] * epsl_rate[1]
                               + domain[is_][0][2] * epsl_rate[2])
           + pery[low[0] - 1] * (domain[is_][1][0] * epsl_rate[0] + domain[is_][1][1] * epsl_rate[1]
                               + domain[is_][1][2] * epsl_rate[2])
           + perz[low[0] - 1] * (domain[is_][2][0] * epsl_rate[0] + domain[is_][2][1] * epsl_rate[1]
                               + domain[is_][2][2] * epsl_rate[2])))

        Chigh = np.array(Gcenter[high[1] - 1]) + (
                    np.array(domain[is_][0]) * perx[high[0] - 1] + np.array(domain[is_][1]) * pery[high[0] - 1]
                  + np.array(domain[is_][2]) * perz[high[0] - 1])
        Clow = np.array(Gcenter[low[1] - 1]) + (
                    np.array(domain[is_][0]) * perx[low[0] - 1] + np.array(domain[is_][1]) * pery[low[0] - 1]
                  + np.array(domain[is_][2]) * perz[low[0] - 1])

        if np.dot(vhigh - vlow, Chigh - Clow) > 0:
            return vhigh
        else:
            return vlow

    GBv = [dynamic(item1[0], item1[1], item2) for item1, item2 in zip(GBgrain, range(nGB[is_]))]

    # Translational velocities of edge
    TLv = [sum(GBv[int(item[1]) - 1] + perx[int(item[0]) - 1] * (domain[is_][0][0] * epsl_rate[0] + domain[is_][0][1] * epsl_rate[1]
                                     + domain[is_][0][2] * epsl_rate[2])
                                     + pery[int(item[0]) - 1] * (domain[is_][1][0] * epsl_rate[0] + domain[is_][1][1] * epsl_rate[1]
                                     + domain[is_][1][2] * epsl_rate[2])
                                     + perz[int(item[0]) - 1] * (domain[is_][2][0] * epsl_rate[0] + domain[is_][2][1] * epsl_rate[1]
                                     + domain[is_][2][2] * epsl_rate[2]) for item in sublist) / len(sublist) for sublist in TLGB]
    # Translationan velocity of QPs
    QPv = [sum(TLv[int(item[1]) - 1] + perx[int(item[0]) - 1] * (domain[is_][0][0] * epsl_rate[0] + domain[is_][0][1] * epsl_rate[1]
                                     + domain[is_][0][2] * epsl_rate[2])
                                     + pery[int(item[0]) - 1] * (domain[is_][1][0] * epsl_rate[0] + domain[is_][1][1] * epsl_rate[1]
                                     + domain[is_][1][2] * epsl_rate[2])
                                     + perz[int(item[0]) - 1] * (domain[is_][2][0] * epsl_rate[0] + domain[is_][2][1] * epsl_rate[1]
                                     + domain[is_][2][2] * epsl_rate[2]) for item in sublist) / len(sublist) for sublist in QPTL]


    # Deformation: updates
    epsl[is_ + 1] = np.array(epsl[is_]) + np.array(epsl_rate) * dt

    domain[is_ + 1] = np.array(domain[0]) + np.array(
        [domain[0][0][0] * np.array(epsl[is_][0]), domain[0][1][1] * np.array(epsl[is_][1]), domain[0][2][2] * np.array(epsl[is_][2])])

    QP[is_ + 1] = np.array(QP[is_]) + np.array(vst) * dt + np.array(QPv) * dt

    nQP[is_ + 1] = len(QP[is_ + 1])

    QP27is_temp = [[np.array(domain[is_ + 1][0]) * x + np.array(domain[is_ + 1][1]) * y
                  + np.array(domain[is_ + 1][2]) * z] for x, y, z in zip(perx, pery, perz)]
    QP27[is_ + 1] = np.array(QP[is_ + 1]) + QP27is_temp

    TL[is_ + 1] = TL[is_]
    nTL[is_ + 1] = len(TL[is_ + 1])

    GB[is_ + 1] = GB[is_]
    nGB[is_ + 1] = len(GB[is_ + 1])

    Grain[is_ + 1] = Grain[is_]
    nG[is_ + 1] = len(Grain[is_ + 1])


    # Find the area and normal vector of each GBF

    def getcoor(QPp, QPn):
        return QP27[is_ + 1][QPp - 1][QPn - 1]

    def GBcoorf(GBn):
        return [[getcoor(item[0], item[1]) for item in sublist] for sublist in GBQP[GBn]]

    GBcoor = [GBcoorf(GBn) for GBn in range(nGB[is_ + 1])]

    # area of each GBF
    GBS[is_ + 1] = [polygonarea(item) for item in GBcoor]

    # barycenter of each GBF
    GBG[is_ + 1] = [polygonG(item) for item in GBcoor]

    # normal vector of each GBf
    GBNV[is_ + 1] = [polygonNV(item) for item in GBcoor]

    # x: vertex of grain boundary, [ [p1, p2], [p2, p3], ...... ]
    # g: barycenter of grain boundary
    # n: normal vector of grain boundary
    def vtxcf(x, g, n):
        re, im = base(n)
        return [[complex(item, g, [re, im, n]) for item in sublist] for sublist in x]

    vtxc[is_ + 1] = [vtxcf(GBcoor[i], GBG[is_ + 1][i], GBNV[is_ + 1][i]) for i in range(len(GBcoor))]


    # Cavitation

    GBNumber = list(range(nGB[is_]))

    # determin the number of generation of cavities
    def nucleationprobabilityf(S0, v_del_i, GBi):
        if calc[GBi] != 1:
            return [0.0, 0.0]
        if v_del_i <= 0:
            return [0.0, 0.0]

        new = (S0 * Alpha_p2 * v_del_i / delta) * dt    # grain boundary thickness
        if new >= 0:
            nuc_temp1 = math.floor(new)
            nuc_temp2 = new - math.floor(new)
        else:
            nuc_temp1 = math.floor(new) + 1
            nuc_temp2 = new - math.floor(new)
        return [nuc_temp1, nuc_temp2]

    nucleationprobability = [nucleationprobabilityf(elem1, elem2, elem3)
                             for elem1, elem2, elem3 in zip(GBS[is_], v_del_mag[is_], GBNumber)]

    def nucleation(cavity0, S0, vtxc0, nucleationprobability0, GBi):
        global rni

        if calc[GBi] != 1:
            return cavity0

        if rn[rni] < nucleationprobability0[1]:
            nucleationnumber = nucleationprobability0[0] + 1
        else:
            nucleationnumber = nucleationprobability0[0]

        rni = (rni) % len(rn) + 1

        if nucleationnumber < 1:
            return cavity0

        temp = cavity0

        s = [abs((0.9 * elem[0][0]) * (0.9 * elem[1][1]) - (0.9 * elem[1][0]) * (0.9 * elem[0][1])) for elem in vtxc0]
        ps = [sum(s[:i + 1]) / sum(s) for i in range(len(s))]

        def nucleationf(temp0):
            global rni
            p = rn[rni]
            rni = (rni) % len(rn) + 1

            # using the random value p to get the number of triple line
            si = sum([1 if p > x else 0 for x in ps]) + 1

            randoma = rn[rni]
            rni = (rni) % len(rn) + 1
            randomb = rn[rni]
            rni = (rni) % len(rn) + 1

            cnew = randoma * (0.9 * vtxc0[si - 1][0] * randomb + 0.9 * vtxc0[si - 1][1] * (1 - randomb))
            theta_new = (math.pi / 2) - delta_0
            beta_new = math.tan(delta_0) * math.tan(theta_new) * (aini / math.cos(theta_new))

            cavtldist = [0 if pldist(cnew, x[0], x[1]) > aini else 1 for x in vtxc0]

            trials = 0

            while sum(cavtldist) > 0:
                p = rn[rni]
                rni = (rni) % len(rn) + 1
                si = sum([1 if p > x else 0 for x in ps]) + 1

                randoma = rn[rni]
                rni = (rni) % len(rn) + 1
                randomb = rn[rni]
                rni = (rni) % len(rn) + 1

                cnew = randoma * (0.9 * vtxc0[si - 1][0] * randomb + 0.9 * vtxc0[si - 1][1] * (1 - randomb))
                cavtldist = [0 if pldist(cnew, x[0], x[1]) > aini else 1 for x in vtxc0]
                theta_new = (math.pi / 2) - delta_0
                beta_new = math.tan(delta_0) * math.tan(theta_new) * (aini / math.cos(theta_new))
                trials = trials + 1

                if trials > 100:
                    print(GBi, " nucleationE")
                    calc[GBi] = 2
                    break

            if len(cavity0) == 0:
                ntime = rn[rni]
                rni = (rni) % len(rn) + 1
                return [temp0 + [aini, cnew, beta_new, theta_new, ntime]]

            # Check if it crosses other cavities or triple lines
            cavdist = [0 if np.linalg.norm(cnew - x[1]) > (aini * 2. + x[0]) else 1 for x in temp0]
            cavtldist = [0 if pldist(cnew, x[0], x[1]) > aini else 1 for x in vtxc0]
            trials = 0

            while sum(cavdist) > 0 or sum(cavtldist) > 0:

                p = rn[rni]
                rni = (rni) % len(rn) + 1
                si = sum(1 if p > x else 0 for x in ps) + 1

                randoma = rn[rni]
                rni = (rni) % len(rn) + 1
                randomb = rn[rni]
                rni = (rni) % len(rn) + 1

                cnew = randoma * (0.9 * vtxc0[si - 1][0] * randomb + 0.9 * vtxc0[si - 1][1] * (1 - randomb))
                cavdist = [0 if np.linalg.norm(cnew - x[1]) > (aini * 2. + x[0]) else 1 for x in temp0]
                cavtldist = [0 if pldist(cnew, x[0], x[1]) > aini else 1 for x in vtxc0]

                theta_new = (math.pi / 2) - delta_0
                beta_new = math.tan(delta_0) * math.tan(theta_new) * (aini / math.cos(theta_new))
                trials = trials + 1

                if trials > 100:
                    print(GBi, " nucleationE:")
                    calc[GBi] = 2
                    break

            ntime = rn[rni]
            rni = (rni) % len(rn) + 1
            return temp0 + [[aini, cnew, beta_new, theta_new, ntime] for i in range(nucleationnumber)]

        for i in range(nucleationnumber):
            temp = nucleationf(temp)

        return temp


    cavity[is_ + 1] = [nucleation(c, g, v, n, u) for c, g, v, n, u in
                       zip(cavity[is_], GBS[is_], vtxc[is_], nucleationprobability, GBNumber)]


    def growth(cavity0, vtxc0, S0, Q0, v_del_mag0, GBNVi, GBi):

        if calc[GBi] != 1:
            return cavity0

        if len(cavity0) == 0:
            return cavity0

        nc = len(cavity0)

        a = [item[0] for item in cavity0]

        c = [item[1] for item in cavity0]
        c = np.array(c)

        beta = [item[2] for item in cavity0]

        theta = [item[3] for item in cavity0]

        ntime = [1 if len(element) == 4 else element[4] for element in cavity0]

        h = [betai * (1 - math.sin(thetai)) for betai, thetai in zip(beta, theta)]

        Rmean = math.sqrt(S0 / math.pi)

        R = 1.1 * max([np.linalg.norm(v) for sublist in vtxc0 for v in sublist])

        # sigma_ tip: stress at the tip of cavity_k
        def sigma_tip(k):
            return (gmma_s * math.tan(delta_0) * math.tan(theta[k]) / beta[k]) * (
                        math.sin(delta_0) / math.cos(theta[k])) * (1 + math.cos(delta_0) ** 2 / math.sin(theta[k]) ** 2)

        # special solution of sigma_0
        def sigma_0(x, y):
            return -(v_del_mag0 / (4 * Dgb)) * (x ** 2 + y ** 2)

        def sigma_0x(x, y):
            return -(v_del_mag0 / (2 * Dgb)) * x

        def sigma_0y(x, y):
            return -(v_del_mag0 / (2 * Dgb)) * y

        def sigma_0r(r, theta, x0, y0):
            return -(v_del_mag0 / (2 * Dgb)) * (r + x0 * math.cos(theta) + y0 * math.sin(theta))

        # Coefficient of sigma_h
        def X0f(m, x, y):
            complex_expression = ((x / R) + 1j * (y / R)) ** m
            return complex_expression.real

        def Y0f(m, x, y):
            complex_expression = ((x / R) + 1j * (y / R)) ** m
            return -complex_expression.imag

        def Xf(m, k, x, y):
            complex_expression = ((x / a[k] - c[k][0] / a[k]) + 1j * (y / a[k] - c[k][1] / a[k])) ** (-m)
            return complex_expression.real

        def Yf(m, k, x, y):
            complex_expression = ((x / a[k] - c[k][0] / a[k]) + 1j * (y / a[k] - c[k][1] / a[k])) ** (-m)
            return -complex_expression.imag

        def lmbd_f(k, x, y):
            return 0.5 * math.log((x - c[k][0]) ** 2 + (y - c[k][1]) ** 2) - math.log(R)

        def X0xf(m, x, y):
            complex_expression = ((x / R) + 1j * (y / R)) ** (m - 1)
            return (m / R) * complex_expression.real

        def Y0xf(m, x, y):
            complex_expression = ((x / R) + 1j * (y / R)) ** (m - 1)
            return -(m / R) * complex_expression.imag

        def Xxf(m, k, x, y):
            complex_expression = ((x / a[k] - c[k][0] / a[k]) + 1j * (y / a[k] - c[k][1] / a[k])) ** (-m - 1)
            return -(m / a[k]) * complex_expression.real

        def Yxf(m, k, x, y):
            complex_expression = ((x / a[k] - c[k][0] / a[k]) + 1j * (y / a[k] - c[k][1] / a[k])) ** (-m - 1)
            return -(m / a[k]) * (-complex_expression.imag)

        def lmbd_xf(k, x, y):
            return (x - c[k][0]) / ((x - c[k][0]) ** 2 + (y - c[k][1]) ** 2)

        # Coefficient of Dsigma_h/Dy
        def X0yf(m, x, y):
            complex_expression = ((x / R) + 1j * (y / R)) ** (m - 1)
            return -(m / R) * complex_expression.imag

        def Y0yf(m, x, y):
            complex_expression = ((x / R) + 1j * (y / R)) ** (m - 1)
            return -(m / R) * complex_expression.real

        def Xyf(m, k, x, y):
            complex_expression = (x / a[k] - c[k][0] / a[k]) + 1j * (y / a[k] - c[k][1] / a[k])
            return m / a[k] * (complex_expression ** (-m - 1)).imag

        def Yyf(m, k, x, y):
            complex_expression = (x / a[k] - c[k][0] / a[k]) + 1j * (y / a[k] - c[k][1] / a[k])
            return m / a[k] * (complex_expression ** (-m - 1)).real

        def lmbd_yf(k, x, y):
            return (y - c[k][1]) / ((x - c[k][0]) ** 2 + (y - c[k][1]) ** 2)

        # Coefficient of Dsigma_h(r,theta)/Dr
        def X0rf(m, r, theta, x0, y0):
            z = ((r * np.cos(theta) + x0) / R + 1j * (r * np.sin(theta) + y0) / R) ** (m - 1)
            return m / R * np.real(z * (np.cos(theta) + 1j * np.sin(theta)))

        def Y0rf(m, r, theta, x0, y0):
            z = ((r * np.cos(theta) + x0) / R + 1j * (r * np.sin(theta) + y0) / R) ** (m - 1)
            return -(m / R) * np.imag(z * (np.cos(theta) + 1j * np.sin(theta)))

        def Xrf(m, k, r, theta, x0, y0):
            z = ((r * np.cos(theta) + x0 - c[k][0]) / a[k] + 1j * (r * np.sin(theta) + y0 - c[k][1]) / a[
                k]) ** (-m - 1)
            return -(m / a[k]) * np.real(z * (np.cos(theta) + 1j * np.sin(theta)))

        def Yrf(m, k, r, theta, x0, y0):
            z = ((r * np.cos(theta) + x0 - c[k][0]) / a[k] + 1j * (r * np.sin(theta) + y0 - c[k][1]) / a[
                k]) ** (-m - 1)
            return (m / a[k]) * np.imag(z * (np.cos(theta) + 1j * np.sin(theta)))

        def lmbd_rf(k, r, theta, x0, y0):
            return ((r * np.cos(theta) + x0 - c[k][0]) * np.cos(theta) + (r * np.sin(theta) + y0
                    - c[k][1]) * np.sin(theta)) / ((r * np.cos(theta) + x0 - c[k][0]) ** 2
                    + (r * np.sin(theta) + y0 - c[k][1]) ** 2)

        # Coefficient of General Solution

        def sigma_h(x, y):
            X00 = [1.0]
            X0m = [X0f(m, x, y) for m in range(1, M1 + 1)]
            Y0m = [Y0f(m, x, y) for m in range(1, M1 + 1)]
            Xkm = [[Xf(m, k, x, y) for m in range(1, M2 + 1)] for k in range(nc)]
            Ykm = [[Yf(m, k, x, y) for m in range(1, M2 + 1)] for k in range(nc)]
            lmbd_k = [lmbd_f(k, x, y) for k in range(nc)]

            return X00 + X0m + Y0m + [val for sublist in Xkm for val in sublist] \
                + [val for sublist in Ykm for val in sublist] + lmbd_k


        def sigma_hx(x, y):
            X00 = [0.]
            X0m = [X0xf(m, x, y) for m in range(1, M1 + 1)]
            Y0m = [Y0xf(m, x, y) for m in range(1, M1 + 1)]
            Xkm = [[Xxf(m, k, x, y) for m in range(1, M2 + 1)] for k in range(nc)]
            Ykm = [[Yxf(m, k, x, y) for m in range(1, M2 + 1)] for k in range(nc)]
            lmbd_k = [lmbd_xf(k, x, y) for k in range(nc)]

            return X00 + X0m + Y0m + [val for sublist in Xkm for val in sublist] \
                + [val for sublist in Ykm for val in sublist] + lmbd_k

        def sigma_hy(x, y):
            X00 = [0.]
            X0m = [X0yf(m, x, y) for m in range(1, M1 + 1)]
            Y0m = [Y0yf(m, x, y) for m in range(1, M1 + 1)]
            Xkm = [[Xyf(m, k, x, y) for m in range(1, M2 + 1)] for k in range(nc)]
            Ykm = [[Yyf(m, k, x, y) for m in range(1, M2 + 1)] for k in range(nc)]
            lmbd_k = [lmbd_yf(k, x, y) for k in range(nc)]

            return X00 + X0m + Y0m + [val for sublist in Xkm for val in sublist] \
                + [val for sublist in Ykm for val in sublist] + lmbd_k


        def sigma_hr(r, theta, x0, y0):
            X00 = [0.]
            X0m = [X0rf(m, r, theta, x0, y0) for m in range(1, M1 + 1)]
            Y0m = [Y0rf(m, r, theta, x0, y0) for m in range(1, M1 + 1)]
            Xkm = [[Xrf(m, k, r, theta, x0, y0) for m in range(1, M2 + 1)] for k in range(nc)]
            Ykm = [[Yrf(m, k, r, theta, x0, y0) for m in range(1, M2 + 1)] for k in range(nc)]
            lmbd_k = [lmbd_rf(k, r, theta, x0, y0) for k in range(nc)]

            return X00 + X0m + Y0m + [val for sublist in Xkm for val in sublist] \
                + [val for sublist in Ykm for val in sublist] + lmbd_k

        # Points of equation
        zcavity = [[c[i - 1] + [a[i - 1] * np.cos((2 * np.pi) / Qtip * j), a[i - 1] * np.sin((2 * np.pi) / Qtip * j)]
                    for j in range(1, Qtip + 1)] for i in range(1, nc + 1)]

        l = [np.linalg.norm(np.array(sublist[0]) - np.array(sublist[1])) for sublist in vtxc0]

        d0 = np.sum(l) / (QL0 + len(vtxc0))

        m = [max(math.ceil(l_temp / d0), 2) for l_temp in l]

        def zTLf(mi, vtxc0i):
            return [(mi - i) / mi * vtxc0i[0] + i / mi * vtxc0i[1] for i in range(mi + 1)]

        # Calculation of points to be calculated
        zTL = list(map(zTLf, m, vtxc0))

        # 1. Boundary conditions for stress at the tip of the cavity
        X1 = [[sigma_h(x[0], x[1]) for x in sublist] for sublist in zcavity]

        y1 = [[-sigma_0(element1[0], element1[1]) + sigma_tip(element2)
               for element1, element2 in zip(inner_list, [k] * Qtip)]
               for inner_list, k in zip(zcavity, range(nc))]


        X1N = [[np.array(x) / sigma_tip(j) for x in sublist] for sublist, j in zip(X1, range(len(X1)))]

        y1N = [[np.array(x) / sigma_tip(j) for x in sublist] for sublist, j in zip(y1, range(len(y1)))]

        weight1 = np.full(Qtip * nc, 1. / np.sqrt(Qtip))

        # 2.1 Boundary condition 1 for diffusion flow at the grain boundary edge (triple line)
        def sf(vtxc0i):
            diff = np.array(vtxc0i[0]) - np.array(vtxc0i[1])
            si = np.array([diff[1], -diff[0]])
            si = si / np.linalg.norm(diff)

            if np.dot(si, np.array(vtxc0i[0])) < 0:
                si = -si
            else:
                si = si
            return si

        s = [sf(item) for item in vtxc0]

        j = [CapitalOmega * ((Q0[2 * i] + Q0[2 * i + 1]) /
             np.linalg.norm(np.array(vtxc0[i][0]) - np.array(vtxc0[i][1]))) for i in range(len(vtxc0))]

        sigma_tip_next = np.array([min(np.array(GBnextStatus[GBi][2 * i][2]), np.array(GBnextStatus[GBi][2 * i + 1][2]))
                         if (np.array(GBnextStatus[GBi][2 * i][1]) == 1 and np.array(GBnextStatus[GBi][2 * i + 1][1]) == 1)
                         else max(np.array(GBnextStatus[GBi][2 * i][2]), np.array(GBnextStatus[GBi][2 * i + 1][2]))
                         for i in range(len(vtxc0))])

        dropn = 0    # not necessary here.

        def X21f(ji, zTLi, si, sigma_tip_nexti):
            if sigma_tip_nexti == 0:
                def g(z):
                    return Dgb * (si[0] * np.array(sigma_hx(z[0], z[1])) + si[1] * np.array(sigma_hy(z[0], z[1])))
            else:
                def g(z):
                    return sigma_h(z[0], z[1])
            return list(map(g, zTLi))

        def y21f(ji, zTLi, si, sigma_tip_nexti):
            if sigma_tip_nexti == 0:
                def g(z):
                    return -Dgb * (si[0] * sigma_0x(z[0], z[1]) + si[1] * sigma_0y(z[0], z[1])) + ji
            else:
                def g(z):
                    return -sigma_0(z[0], z[1]) + sigma_tip_nexti
            return list(map(g, zTLi))

        X21 = list(map(X21f, j, zTL, s, sigma_tip_next))

        y21 = list(map(y21f, j, zTL, s, sigma_tip_next))

        X21N = list(map(lambda x, y, z: x / (y if z == 0 else z), X21, j, sigma_tip_next))

        y21N = list(map(lambda x, y, z: x / (y if z == 0 else z), y21, j, sigma_tip_next))

        weight21 = [math.sqrt(0.001 / len(y21)) for sublist in y21 for _ in sublist]

        # 2.2 Boundary condition 2 for diffusion flow at the grain boundary edge (triple line)
        l = list(map(lambda x: np.linalg.norm(x[0] - x[1]), np.array(vtxc0)))

        def zTL2f(vtxc0i):
            return [((nss - z) / nss * vtxc0i[0] + z / nss * vtxc0i[1]) for z in range(nss + 1)]

        zTL2 = [zTL2f(vtxc0i) for vtxc0i in vtxc0]

        def X22f(zTL2i, li, si, sigma_tip_next_i):
            if sigma_tip_next_i == 0:
                def g(z):
                    return Dgb * (si[0] * np.array(sigma_hx(z[0], z[1])) + si[1] * np.array(sigma_hy(z[0], z[1])))
            else:
                def g(z):
                    return sigma_h(z[0], z[1])
            return li / (6 * nss) * sum(np.array(g(z1)) + 4 * np.array(g((z1 + z2) / 2)) + np.array(g(z2)) for z1, z2 in zip(zTL2i[:-1], zTL2i[1:]))

        def y22f(zTL2i, li, si, ji, sigma_tip_nexti):
            if sigma_tip_nexti == 0:
                def g(z):
                    return -Dgb * (si[0] * np.array(sigma_0x(z[0], z[1])) + si[1] * np.array(sigma_0y(z[0], z[1])))
            else:
                def g(z):
                    return -sigma_0(z[0], z[1]) + sigma_tip_nexti
            return li / (6 * nss) * sum(np.array(g(z1)) + 4 * np.array(g((z1 + z2) / 2)) + np.array(g(z2))
                                         for z1, z2 in zip(zTL2i[:-1], zTL2i[1:])) + ji * li

        X22 = [X22f(zTL2i, li, si, sigma_tip_nexti)
               for zTL2i, li, si, sigma_tip_nexti in zip(zTL2, l, s, sigma_tip_next)]

        y22 = [y22f(zTL2i, li, si, ji, sigma_tip_nexti)
               for zTL2i, li, si, ji, sigma_tip_nexti in zip(zTL2, l, s, j, sigma_tip_next)]

        X22N = [x / ((j_val if sigma_tip_next_i == 0 else sigma_tip_next_i) * l_val)
                for x, l_val, j_val, sigma_tip_next_i in zip(X22, l, j, sigma_tip_next)]

        y22N = [y / ((j_val if sigma_tip_next_i == 0 else sigma_tip_next_i) * l_val)
                for y, l_val, j_val, sigma_tip_next_i in zip(y22, l, j, sigma_tip_next)]

        weight22 = [np.sqrt(l_val / np.sum(l)) for l_val in l]


        # 3. Stress conditions
        def X3lf(zTL2i, li, si):
            def g(z):
                term1 = (2 * z[0] * si[0] + 2 * z[1] * si[1]) * np.array(sigma_h(z[0], z[1]))
                term2 = (z[0] ** 2 + z[1] ** 2) * (si[0] * np.array(sigma_hx(z[0], z[1])) + si[1] * np.array(sigma_hy(z[0], z[1])))
                return term1 - term2

            return (1 / 4) * li / (6 * nss) * sum([g(z1) + 4 * g((z1 + z2) / 2) + g(z2)
                    for z1, z2 in zip(zTL2i[1:], zTL2i[:-1])])

        def X3cf(ci, ai):
            def g(theta):
                term1 = 2 * (ai + ci[0] * np.cos(theta) + ci[1] * np.sin(theta)) * np.array(sigma_h(
                    ai * np.cos(theta) + ci[0], ai * np.sin(theta) + ci[1])) * ai
                term2 = (ai ** 2 + 2 * ci[0] * ai * np.cos(theta) + 2 * ci[1] * ai * np.sin(theta) + ci[0] ** 2
                         + ci[1] ** 2) * np.array(sigma_hr(ai, theta, ci[0], ci[1])) * ai
                return np.array(term1) - np.array(term2)

            return 1 / 4 * (2 * math.pi) / (3 * Qtip) * sum([
                g((2 * math.pi) * (2 * x) / (2 * Qtip) ) + 2 * g((2 * math.pi) * (2 * x - 1) / (2 * Qtip) )
                 for x in range(1, 1 + Qtip)])

        X3 = sum([X3lf(zTL2i, li, si) for zTL2i, li, si in zip(zTL2, l, s)]) - sum([X3cf(ci, ai)
            for ci, ai in zip(c, a)])

        first_term = S0 * GBNVi.dot(sigma).dot(GBNVi)
        second_term = v_del_mag0 / (48 * Dgb) * sum(
            ((np.dot(x[0], x[0]) + np.dot(x[1], x[1])) + (x[0][0] * x[1][0] + x[0][1] * x[1][1])) * abs(
                -x[0][1] * x[1][0] + x[0][0] * x[1][1])for x in vtxc0)
        third_term = v_del_mag0 / (4 * Dgb) * sum(
            1 / 2 * np.pi * a_val ** 2 * (2 * np.linalg.norm(c_val) ** 2 + a_val ** 2) for a_val, c_val in zip(a, c))
        y3 = first_term + second_term - third_term

        X3N = np.array(X3) / (S0 * np.dot(GBNVi, np.dot(sigma, GBNVi)))
        y3N = y3 / np.dot(np.dot(S0 * GBNVi, sigma), GBNVi)
        weight3 = [nc + 1.]

        X1 = [[np.array(element) for element in sublist] for sublist in X1]
        # Determination of unknown constants
        X0 = [item for sublist in X1 for item in sublist] + [item for sublist in X21 for item in sublist] + X22 + [X3]
        y0 = [item for sublist in y1 for item in sublist] + [item for sublist in y21 for item in sublist] + y22 + [y3]
        X0N = [item for sublist in X1N for item in sublist] + [item for sublist in X21N for item in sublist] + X22N + [X3N]
        y0N = [item for sublist in y1N for item in sublist] + [item for sublist in y21N for item in sublist] + y22N + [y3N]
        W0 = np.diag(np.concatenate((weight1, weight21, weight22, weight3)))


        eliminaterow = -1
        eliminatecolumn = 1

        X0Ny0Nedit = [
            [x0 - x0[eliminatecolumn-1] / np.array(X0N[eliminaterow][eliminatecolumn-1]) * np.array(X0N[eliminaterow]),
             y0 - x0[eliminatecolumn-1] / np.array(X0N[eliminaterow][eliminatecolumn-1]) * np.array(y0N[eliminaterow])]
            for x0, y0 in zip(X0N, y0N)
                     ]

        X0calc_temp = [element[0] for element in X0Ny0Nedit][:-1]
        X0calc = [sublist[1:] for sublist in X0calc_temp]

        y0calc = [element[1] for element in X0Ny0Nedit][:-1]

        W0calc = [row[:-1] for row in W0[:-1]]

        X0calc = np.array(X0calc)
        W0calc = np.array(W0calc)
        y0calc = np.array(y0calc)
        bbcalc = np.linalg.inv(X0calc.T @ W0calc @ X0calc) @ X0calc.T @ W0calc @ y0calc

        bb = np.insert(bbcalc, eliminatecolumn-1, (y0[eliminaterow]
             - np.dot(np.delete(X0[eliminaterow], eliminatecolumn - 1), bbcalc)) / (
             X0[eliminaterow][eliminatecolumn - 1]))

        def Xgrowthfh(ci, ai):
            def g(theta):
                return np.array(sigma_hr(ai, theta, ci[0], ci[1]))

            return (Dgb * ai * np.pi) / (3 * Qtip) * np.array([
                g((2 * np.pi) / Qtip * (i - 1)) + 4 * g((2 * np.pi) / Qtip * (i - 1/2)) + g((2 * np.pi) / Qtip * i)
                for i in range(1, Qtip + 1)])

        def ygrowthf0(ci, ai):
            def g(theta):
                return sigma_0r(ai, theta, ci[0], ci[1])

            return (Dgb * ai * np.pi) / (3 * Qtip) * np.array([
                g((2 * np.pi) / Qtip * (i - 1)) + 4 * g((2 * np.pi) / Qtip * (i - 1/2)) + g((2 * np.pi) / Qtip * i)
                for i in range(1, Qtip + 1)])

        Xgrowthh = [Xgrowthfh(ci, ai) for ci, ai in zip(c, a)]
        ygrowth0 = [ygrowthf0(ci, ai) for ci, ai in zip(c, a)]

        Vkl = np.dot(Xgrowthh, bb) + ygrowth0

        X_bb_temp = [np.sum(np.dot(Xgrowthh, bb)[i]) for i in range(len(np.dot(Xgrowthh, bb)))]
        y_groth0_temp = [np.sum(ygrowth0[i]) for i in range(len(ygrowth0))]

        Vrate = np.array(X_bb_temp) + np.array(y_groth0_temp) + np.array([np.pi * x ** 2 * v_del_mag0 for x in a])

        V = [(2 * np.pi) / (3 * np.tan(delta_0) ** 2) * item_beta ** 3 * (
             np.sin(item_theta) ** 3 - 3 * np.sin(item_theta) + 2) / np.tan(item_theta) ** 2
             for item_beta, item_theta in zip(beta, theta)]

        temp_V = np.array(V) + np.array(Vrate) * dt * np.array(ntime)
        temp_h = h
        temp_c = c
        temp_theta = theta

        temp_V = list(temp_V)
        temp_h = list(temp_h)
        temp_theta = list(temp_theta)
        temp_c = list(temp_c)
        Vkl = list(Vkl)

        for i in range(len(V)):
            if temp_V[i] < deleteV:
                temp_V[i] = "delete"
                temp_h[i] = "delete"
                temp_theta[i] = "delete"
                temp_c[i] = "delete"
                Vkl[i] = "delete"

            elif temp_V[i] < h[i] ** 3 * smallVCalc:
                temp_h[i] = (temp_V[i] * hCalc) ** (1/3)
                temp_theta[i] = math.pi / 2 - delta_0

            else:
                temp_theta[i] = math.asin(
                    (3 + math.sqrt(9 + 8 * ((3 * math.tan(delta_0) ** 2) / (2 * math.pi) * temp_V[i] / (h[i]) ** 3 - 1))) /
                    (2 * ((3 * math.tan(delta_0) ** 2) / (2 * math.pi) * temp_V[i] / (h[i]) ** 3 - 1)))

        def beta_f(theta_0, h0):
            return h0 / (1 - math.sin(theta_0))

        def alpha_f(beta_0, theta_0):
            return beta_0 / (math.tan(delta_0) * math.tan(theta_0))

        def af(beta_0, theta_0):
            return (beta_0 * math.cos(theta_0)) / (math.tan(delta_0) * math.tan(theta_0))

        temp_beta = ["delete" if theta == "delete" else beta_f(theta, h) for theta, h in zip(temp_theta, temp_h)]
        temp_alpha = ["delete" if beta == "delete" else alpha_f(beta, theta) for beta, theta in zip(temp_beta, temp_theta)]
        temp_a = ["delete" if beta == "delete" else af(beta, theta) for beta, theta in zip(temp_beta, temp_theta)]
        delta_ak = ["delete" if elem1 == "delete" else elem1 - elem2 for elem1, elem2 in zip(temp_a, a)]

        V = [item for item in temp_V if item != "delete"]
        h = [item for item in temp_h if item != "delete"]
        c = [item for item in temp_c if not np.any(item == "delete")] #
        theta = [item for item in temp_theta if item != "delete"]
        beta = [item for item in temp_beta if item != "delete"]
        alpha = [item for item in temp_alpha if item != "delete"]
        a = [item for item in temp_a if item != "delete"]
        delta_ak = [item for item in delta_ak if item != "delete"]
        Vkl = [item for item in Vkl if not np.any(item == "delete")] #

        # Void movements
        midQtip = [[math.cos(math.pi / Qtip * (2 * x - 1)), math.sin(math.pi / Qtip * (2 * x - 1))] for x in range(1, Qtip + 1)]

        if len(V) != 0:
            ratioVkl = Vkl / np.abs(np.sum(Vkl, axis=1, keepdims=True))

            delta_zkl = np.array([[abs(delta_ak1) * x[i] * np.array(midQtip[i]) for i in range(len(midQtip))]
                          for delta_ak1, x in zip(delta_ak, ratioVkl)]) / 2

            delta_z = [sum(item) for item in delta_zkl]

            c = [ci + d for ci, d in zip(c, delta_z)]

        # Surface diffusion
        def Lf(a, b, x):
            return math.pi * a * math.cos(x)

        def Lf1(a, b, x):
            epsilon = 1e-10  # A small value for numerical differentiation
            delta = Lf(a, b, x + epsilon) - Lf(a, b, x)
            derivative = delta / epsilon
            return derivative

        def Sf1(a, b, x):
            result = math.sqrt(a ** 2 * math.sin(x) ** 2 + b ** 2 * math.cos(x) ** 2)
            return result

        def Sf2(a, b, x):
            t = x
            derivative = (Sf1(a, b, t + 1e-10) - Sf1(a, b, t)) / 1e-10
            return derivative

        def mu_f(a, b, x):
            denominator = a * (a ** 2 * math.sin(x) ** 2 + b ** 2 * math.cos(x) ** 2) ** (1 / 2)
            term1 = -gmma_s * CapitalOmega * (b / denominator)
            term2 = -gmma_s * CapitalOmega * (a * b) / (a ** 2 * math.sin(x) ** 2 + b ** 2 * math.cos(x) ** 2) ** (3 / 2)
            result = term1 + term2
            return result

        def mu_f1(a, b, x):
            t = x
            derivative = (mu_f(a, b, t + 1e-5) - mu_f(a, b, t)) / 1e-5
            return derivative

        def mu_f2(a, b, x):
            t = x
            derivative = (mu_f1(a, b, t + 1e-5) - mu_f1(a, b, t)) / 1e-5
            return derivative

        def aratef(a, b, x):
            return -aratefConst * ((mu_f1(a, b, x) * Lf1(a, b, x)) / (Lf(a, b, x) * (Sf1(a, b, x)) ** 2)
                                    - mu_f1(a, b, x) * Sf2(a, b, x) / (Sf1(a, b, x)) ** 3 + mu_f2(a, b, x) / (
                                    Sf1(a, b, x) ** 2))


        arate = [aratef(a1, a2, a3) for a1, a2, a3 in zip(alpha, beta, theta)]

        a = a + (np.array(arate) / math.sin(delta_0)) * dt
        a = [max(a_i, math.pow(np.array(v_i) / aCalc, 1 / 3)) for a_i, v_i in zip(a, V)]

        theta = [math.asin((-(1 - thetaCalc * np.array(v) / a_elem**3) + math.sqrt(1 - thetaCalc * np.array(v) / a_elem**3)) / (1 - thetaCalc * np.array(v) / a_elem**3)) for a_elem, v in zip(a, V)]

        beta = [math.tan(delta_0) * math.tan(elem_theta) * elem_a / math.cos(elem_theta)
                for elem_a, elem_theta in zip(a, theta)]


        h = [elem_beta * (1 - math.sin(elem_theta)) * math.cos(elem_theta)
            for elem_beta, elem_theta in zip(beta, theta)]

        if len(theta) == 0:
            return []
        else:
            return list(map(lambda a, c, beta, theta: [a, c, beta, theta], a, c, beta, theta))

    cavity[is_ + 1] = [growth(elem_c, elem_v, elem_G, elem_Q, elem_vdm, elem_GBNV, GBNum)
                       for elem_c, elem_v, elem_G, elem_Q, elem_vdm, elem_GBNV, GBNum
                       in zip(cavity[is_ + 1], vtxc[is_], GBS[is_], Q[is_], v_del_mag[is_], GBNV[is_], GBNumber)]


    # Extrapolation of Area Fraction
    def extrapolation(cavity0, S0, GBi):

        if calc[GBi] != 2:
            return cavity0

        # find the number of step that the area fraction bigger than 0
        def count0f(x):
            for i in range(len(x)):
                if x[i] != 0:
                    return i - 1
            return len(x)

        # calculate the number of steps where the area fraction is 0 at the grain boundary GBi
        omega_temp1 = [row[GBi] for row in omega_cavity[0 : is_ + 1]]
        ndata0 = count0f(omega_temp1)

        omega_temp2 = [row[GBi] for row in omega_cavity[ndata0 : is_ + 1]]
        data = [[i, j] for i, j in zip(range(1, is_ - ndata0 + 2), omega_temp2)]

        x_values = [point[0] for point in data]
        y_values = [point[1] for point in data]

        from scipy.interpolate import interp1d
        if 0 < len(data) <= 2:
            omega_cavityextra = interp1d(x_values, y_values, kind='linear', fill_value="extrapolate")
        elif len(data) == 3:
            omega_cavityextra = interp1d(x_values[-3 :], y_values[-3 :], kind='quadratic', fill_value="extrapolate")
        else:
            omega_cavityextra = interp1d(x_values[-4 :], y_values[-4 :], kind='cubic', fill_value="extrapolate")

        def omega_cavityf(x):
            return omega_cavityextra(x - ndata0 + 1)

        if omega_cavityf(is_ + 1) >= 1:
            calc[GBi] = 3  # Calculation for the grain boundary is over
            print(f"{GBi} omega = 1.0")

            anew = math.sqrt(S0 / math.pi)
            theta_new = min([row[3] for row in cavity0])
            beta_new = math.tan(delta_0) * math.tan(theta_new) * anew / math.cos(theta_new)

            # Update of next grains' states

            # define the "finding function"
            def find_positions(x, y):
                result = []
                for i in range(len(x)):
                    for j in range(len(x[i])):
                        if x[i][j] == y:
                            result.append([i, j])
                return result

            GBnext_temp = [[sublist[0][1] for sublist in nested_list] for nested_list in GBnextStatus]
            indices = find_positions(GBnext_temp, GBi + 1)

            for idx in indices:
                GBnextStatus[idx[0]][idx[1]][1] = 1
                GBnextStatus[idx[0]][idx[1]][2] = gmma_s * (np.tan(delta_0) * np.tan(theta_new)) / beta_new * np.sin(
                    delta_0) / np.cos(theta_new) * (1 + np.cos(delta_0) ** 2 / np.sin(theta_new) ** 2)

            return [[anew, np.array([0.0, 0.0]), beta_new, theta_new]]

        # Prevent the convexity when the number of data is too small
        compare = omega_cavityf(is_ + 1) - 2 * omega_cavityf(is_) + omega_cavityf(is_ - 1)

        if compare < 0:
            anew = math.sqrt((S0 * (omega_cavityf(is_) + (omega_cavityf(is_) - omega_cavityf(is_ - 1)))) / math.pi)
            theta_new = min([row[3] for row in cavity0])
            beta_new = math.tan(delta_0) * math.tan(theta_new) * anew / math.cos(theta_new)

            return [[anew, np.array([0.0, 0.0]), beta_new, theta_new]]

        anew = math.sqrt((S0 * omega_cavityf(is_ + 1)) / math.pi)
        theta_new = min([row[3] for row in cavity0])
        beta_new = math.tan(delta_0) * math.tan(theta_new) * anew / math.cos(theta_new)

        return [[anew, np.array([0.0, 0.0]), beta_new, theta_new]]

    cavity[is_ + 1] = list(map(extrapolation, cavity[is_ + 1], GBS[is_ + 1], GBNumber))

    # Cavity Separation

    def separation(cavity0, GBi):
        global check
        if calc[GBi] != 1:
            return cavity0
        if len(cavity0) <= 1:
            return cavity0

        temp = cavity0
        cavdist = [[None for _ in range(len(temp))] for _ in range(len(temp))]

        def cavdistf(i, j):
            if i == j:
                cavdist[i][i] = 0
                return
            if i > j:
                cavdist[i][j] = 0
                return

            if np.linalg.norm(np.array(temp[i][1]) - np.array(temp[j][1])) > temp[i][0] + temp[j][0] + aini:
                cavdist[i][j] = 0
            else:
                cavdist[i][j] = 1

            return

        for i in range(len(temp)):
            for j in range(len(temp)):
                cavdistf(i, j)


        if np.sum(cavdist) == 0:
            return temp

        check = 1

        SPiteration = 0

        while np.sum(cavdist) != 0:
            positions = [(row_idx, col_idx) for row_idx, row in enumerate(cavdist)
                         for col_idx, val in enumerate(row) if val == 1]
            i = positions[0][0]
            j = positions[0][1]

            norm_ij = np.linalg.norm(np.array(temp[i][1]) - np.array(temp[j][1]))
            norm_ji = np.linalg.norm(np.array(temp[j][1]) - np.array(temp[i][1]))

            d1 = 0.5 * (temp[i][0] + temp[j][0] + 2 * aini - norm_ij) * (
                        np.array(temp[i][1]) - np.array(temp[j][1])) / norm_ij

            d2 = 0.5 * (temp[i][0] + temp[j][0] + 2 * aini - norm_ij) * (
                        np.array(temp[j][1]) - np.array(temp[i][1])) / norm_ji

            temp[i][1] = np.array(temp[i][1]) + d1
            temp[j][1] = np.array(temp[j][1]) + d2

            for i in range(len(temp)):
                for j in range(len(temp)):
                    cavdistf(i, j)

            SPiteration += 1

            if SPiteration == 500:
                print(f"{GBi} SP")
                calc[GBi] = 2
                break

        return temp

    # Close to the triple line inside

    def growbeyondtl(cavity0, vtxc0, GBi):
        global check
        ncavity0 = len(cavity0)
        nTL0 = len(vtxc0)

        if calc[GBi] != 1:
            return cavity0

        if ncavity0 == 0:
            return cavity0

        p = 0.99  # Tolerance value
        H = [plfoot([0.0, 0.0], x[0], x[1]) for x in vtxc0]
        magH = [np.linalg.norm(h) for h in H]
        pmagH = [p * mag for mag in magH]
        a0 = [sublist[0] for sublist in cavity0]
        tempCavity = cavity0

        cavityTlMatrix = [[[i, j] for j in range(nTL0)] for i in range(ncavity0)]
        f = [[magH[element[1]] - a0[element[0]] for element in row] for row in cavityTlMatrix]
        f2 = [[pmagH[element[1]] - a0[element[0]] for element in row] for row in cavityTlMatrix]

        def mag_production_vector(i0):
            return [np.dot(tempCavity[i0][1], H[idx]) / magH[idx] for idx in range(nTL0)]

        def check_cavity_tl(i0):
            mag_prod_vec = mag_production_vector(i0)
            return [mpv - f_elem for mpv, f_elem in zip(mag_prod_vec, f[i0])]

        def move_cavity(i0):
            TLiteration = 0

            while max(check_cavity_tl(i0)) > 0:
                def find_positions(x, y):
                    result = []
                    for i in range(len(x)):
                        if x[i] == y:
                            result.append([i])
                    return result

                g = find_positions(check_cavity_tl(i0), max(check_cavity_tl(i0)))[0][0]
                mag_prod_vec = mag_production_vector(i0)

                tempCavity[i0][1] = [x * f2[i0][g] / mag_prod_vec[g] for x in tempCavity[i0][1]]

                TLiteration += 1
                if TLiteration == 500:
                    print(GBi, "TL")
                    calc[GBi] = 2
                    break

        for i in range(ncavity0):

            if max(check_cavity_tl(i)) > 0:
                move_cavity(i)
                check = 1


        return tempCavity

    for i in range(len(cavity[is_ + 1])):
        check = 0
        cavity[is_ + 1][i] = separation(cavity[is_ + 1][i], GBNumber[i])
        cavity[is_ + 1][i] = growbeyondtl(cavity[is_ + 1][i], vtxc[is_ + 1][i], GBNumber[i])

        SPTLiteration = 0

        while check != 0:
            check = 0
            cavity[is_ + 1][i] = separation(cavity[is_ + 1][i], GBNumber[i])
            cavity[is_ + 1][i] = growbeyondtl(cavity[is_ + 1][i], vtxc[is_ + 1][i], GBNumber[i])
            SPTLiteration = SPTLiteration + 1

            if SPTLiteration == 500:
                print(i, "SP+TL")
                calc[i] = 2
                break


    # Adjust so that the area fraction is 1.0
    def extrapolation2(cavity0, S0, GBi):
        if calc[GBi] != 3:
            return cavity0

        anew = math.sqrt(S0 / math.pi)
        theta_new = cavity0[0][3]
        beta_new = math.tan(delta_0) * math.tan(theta_new) * anew / math.cos(theta_new)

        return [[anew, np.array([0.0, 0.0]), beta_new, theta_new]]

    cavity[is_ + 1] = [extrapolation2(cav, gbs, gb) for cav, gbs, gb in zip(cavity[is_ + 1], GBS[is_ + 1], GBNumber)]


    # Calculation of number of cavity and area fraction

    # the area fraction of grain boundary i in step is_ + 1
    def omega(i):
        if len(cavity[is_ + 1][i]) == 0:
            return 0.0
        return sum([math.pi * element[0] ** 2  for element in cavity[is_ + 1][i]]) / GBS[is_ + 1][i]

    # the area fraction of whole grain boundary in step is_ + 1
    omega_cavity[is_ + 1] = [omega(i) for i in range(len(cavity[is_ + 1]))]

    # total area of void / total area of grain boundary
    omega_all[is_ + 1] = sum(
        [omega_cavity[is_ + 1][i] * GBS[is_ + 1][i] for i in range(len(omega_cavity[is_ + 1]))]) / sum(GBS[is_ + 1])

    # the Length of omega_cavity and GBS is equal
    def n(i):
        if len(cavity[is_ + 1][i]) == 0:
            return 0
        return len(cavity[is_ + 1][i])

    # the number of voids in step is_ + 1
    ncavity[is_ + 1] = [n(i) for i in range(len(cavity[is_ + 1]))]


    calcrecord[is_ + 1] = calc


    pstep = is_
    print(is_, "/", step)


    # The Calculation ended in the special occasion
    if calc.count(3) > 0 and calc.count(0) > 0:
        import sys
        sys.exit()

    # Final Results
    if is_ == step:
        def getcoor(QPp, QPn):
            return QP27[is_][QPp][QPn]

        def GBcoorf(GBn):
            return [[getcoor(element) for element in sublist] for sublist in GBQP[GBn]]

        GBcoor = [GBcoorf(GBn) for GBn in range(nGB[is_])]

        GBS[is_ + 1] = [polygonarea(x) for x in GBcoor]

        GBG[is_ + 1] = [polygonG(x) for x in GBcoor]

        GBNV[is_ + 1] = [polygonNV(x) for x in GBcoor]

        def vtxcf(x, g, n):
            re, im = base(n)
            return [[complex(item, g, [re, im, n]) for item in sublist] for sublist in x]

        vtxc[is_ + 1] = [vtxcf(term1, term2, term3) for term1, term2, term3 in zip(GBcoor, GBG[is_], GBNV[is_])]


#execution(0)


for is_ in range(step):
    execution(is_)

import matplotlib.pyplot as plt
plt.plot(omega_all, 'o')
plt.show()

from openpyxl import Workbook
wb = Workbook()
sheet = wb.active

for i, value in enumerate(omega_all):
    sheet.cell(row=i+1, column=1, value=value)
wb.save("omega_all_Python_test18.xlsx")

