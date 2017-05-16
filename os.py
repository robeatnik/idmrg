import os
# for chi in [10,20,30,40,50]:
#     os.system('python3 idmrg.py --chi={}'.format(chi))
for N in [15,25,35,45,55]:
    os.system('python3 idmrg.py --N={} --n_chi=5'.format(N))
