
class MPIComm(object):
    def __init__(self, start, Ntot, do_mpi):

        if do_mpi:

            from mpi4py import MPI

            comm = MPI.COMM_WORLD
            self.rank = comm.Get_rank()
            self.size = comm.Get_size()
        else:
            self.size = 1
            self.rank = 0


        self.Ntot = Ntot
        
        delta = int(Ntot/self.size)

        self.iMin = self.rank*delta+start
        self.iMax = (self.rank+1)*delta+start

        if self.rank == self.size-1:
            self.iMax = Ntot+start


        self.tasks = range(self.iMin, self.iMax)