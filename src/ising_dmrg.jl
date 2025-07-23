using HDF5: h5open, read, write
using ITensorMPS: siteinds, OpSum, MPO, MPS, random_mps, dmrg

function ising_dmrg(filename)
    fp = h5open(filename, "r+")

    sites = siteinds("Qubit", read(fp["N"]))
    os = OpSum()

    data_zz = read(fp["ZZ"])
    nzz = length(data_zz) ÷ 2
    for iop in 1:nzz
        os += "Z", data_zz[1, iop] + 1, "Z", data_zz[2, iop] + 1
    end
    for index in read(fp["Z"])
        os += "Z", index + 1
    end
    kval = read(fp["K"])
    for index in read(fp["X"])
        os += kval, "X", index + 1
    end

    H = MPO(os, sites)
    psi0 = random_mps(sites, linkdims=2)

    nsweeps = 5
    maxdim = [10, 20, 100, 100, 200]
    cutoff = [1.e-10]

    energy, psi = dmrg(H, psi0; nsweeps, maxdim, cutoff)

    write(fp, "energy", energy)
    write(fp, "psi", psi)
    close(fp)
end

let
    filename = ARGS[1]
    ising_dmrg(filename)
end
