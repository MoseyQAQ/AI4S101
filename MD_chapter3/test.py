import numpy as np 

box = [43.08,0,0,0,43.08,0,0,0,43.08]

# calculate det
det = box[0]*(box[4]*box[8] - box[5]*box[7]) + \
        box[1]*(box[5]*box[6] - box[3]*box[8]) + \
        box[2]*(box[3]*box[7] - box[4]*box[6])
print(det)

# calculate inv
inv = np.zeros(9)
inv[0] = (box[4]*box[8] - box[5]*box[7]) / det
inv[1] = (box[2]*box[7] - box[1]*box[8]) / det
inv[2] = (box[1]*box[5] - box[2]*box[4]) / det
inv[3] = (box[5]*box[6] - box[3]*box[8]) / det
inv[4] = (box[0]*box[8] - box[2]*box[6]) / det
inv[5] = (box[3]*box[2] - box[0]*box[5]) / det
inv[6] = (box[3]*box[7] - box[4]*box[6]) / det
inv[7] = (box[1]*box[6] - box[0]*box[7]) / det
inv[8] = (box[0]*box[4] - box[1]*box[3]) / det
print(inv)