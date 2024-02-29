
class MPIComm(object):
    def __init__(self, start, Ntot, do_mpi):

        if do_mpi:

            from mpi4py import MPI

            self.comm = MPI.COMM_WORLD
            self.rank = self.comm.Get_rank()
            self.size = self.comm.Get_size()
        else:
            self.comm = None
            self.size = 1
            self.rank = 0


        self.Ntot = Ntot
        
        delta = int(Ntot/self.size)

        self.iMin = self.rank*delta+start
        self.iMax = (self.rank+1)*delta+start

        if self.rank == self.size-1:
            self.iMax = Ntot+start


        self.tasks = range(self.iMin, self.iMax)