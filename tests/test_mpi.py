# test_mpi.py
# from mpi4py import rc
# rc.initialize = False

from mpi4py import MPI
# MPI.Init()

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

print(f"[Rank {rank}] size={size}")