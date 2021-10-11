"""
    integration of the Trappist-1 system as in Tamayo et al. 2017
    https://ui.adsabs.harvard.edu/abs/2017ApJ...840L..19T/abstract
    But without turbulence
"""

import rebound, reboundx
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
import res_angles as angles

#  Useful constants
pi=np.pi
rad=pi/180
deg=180/pi

yr=2*pi
au=1.
msun=1.
mj=0.000954588
me=3.003e-6
rsun=0.00465047
rj=0.10045*rsun
re=rj/11.209
rs=rsun
sec2year=yr/(3600*24*365.25)
day2year=(1/365.25)*2*pi
#np.random.seed(1)

def min180To180(val):
    while val < -np.pi:
        val += 2*np.pi
    while val > np.pi:
        val -= 2*np.pi
    return (val*180/np.pi)
vmin180To180=np.vectorize(min180To180)

def res_angles(p,q,w1,w2,l1,l2):
    theta1 = vmin180To180( (p+q)*l2 - p*l1 - q*w1 )
    theta2 = vmin180To180( (p+q)*l2 - p*l1 - q*w2 )
    return theta1, theta2

def period(a,ms):
    """ Compute orbital period """
    n2 = ms/np.power(a,3)
    return 2*pi/np.sqrt(n2)

def migration_time(rand):
    """ Compute the migration and e-damping timescales """
    ta=np.random.uniform(4,8)*1e4*yr
    K=np.power(10,rand)
    te=ta/K
    return ta, te

def plot_res(time, angle, plot_name):
    fig, ax = plt.subplots(figsize=(8.35,8.35))
    ax.plot(time, angle, 'k.')
    ax.set_axis_off()
    ax.set_xlim(np.amin(time), np.amax(time))
    ax.set_ylim(-180,180)
    ax.set_box_aspect(1)
    plt.savefig(plot_name+'.png', format='png', dpi=20, bbox_inches='tight', pad_inches = 0)
    #plt.show()

def get_delta(P1,P2,p,q):
    return (P2/P1) - (p+q)/float(p)

def find_res(orb1,orb2,n):

    run_name = 'run'+str(n)
    delta_min=1.
    for p in range(1,10):
        q=1.
        delta = get_delta(orb1[1,-1], orb2[1,-1], p, q)
        if np.abs(delta)<delta_min:
            delta_min=np.abs(delta)
            pres=p
            qres=q
    print("The system is near a ", int(pres+qres), ":", int(pres), "resonance")

    theta1, theta2 = res_angles(pres,qres,orb1[6],orb2[6],orb1[7],orb2[7])

    plot_res(orb1[0],theta1, run_name+'_1')
    plot_res(orb1[0],theta2, run_name+'_2')

    # Generate more data by looking at the second nearest resonance
    delta_res = get_delta(orb1[1,-1], orb2[1,-1], pres, qres)
    if pres==1:
        theta1, theta2 = res_angles(pres+1,qres,orb1[6],orb2[6],orb1[7],orb2[7])
        print("The second nearest resonance is the ", int(pres+1+qres), ":", int(pres+1), "resonance")
    else:
        print(delta_res, orb2[1,-1]/orb1[1,-1])
        if delta_res<0:
            print("The second nearest resonance is the ", int(pres+1+qres), ":", int(pres+1), "resonance")
            theta1, theta2 = res_angles(pres+1,qres,orb1[6],orb2[6],orb1[7],orb2[7])
        if delta_res>0:
            print("The second nearest resonance is the ", int(pres-1+qres), ":", int(pres-1), "resonance")
            theta1, theta2 = res_angles(pres-1,qres,orb1[6],orb2[6],orb1[7],orb2[7])
    plot_res(orb1[0], theta1, run_name+'_1n')
    plot_res(orb1[0], theta2, run_name+'_2n')

def mig_sim(n, verbose=False):
    """ Main Function which creates the simulation and run the integration """

    run_name = 'run'+str(n)
    if os.path.exists(run_name + '.bin'):
        os.remove(run_name + '.bin')

    NP=2
    e_seed=1e-4
    i_seed=1e-4

    # Create initial conditions
    P1 = 50.*day2year
    P2 = np.random.uniform(1.25,2.5)*P1
    P = np.array([P1, P2])
    m1 = np.random.uniform(1,20)*me
    m2 = np.random.uniform(0.5*m1,10*m1)
    mass = np.array([m1,m2])
    ecc = np.array([e_seed, e_seed, e_seed, e_seed, e_seed, e_seed, e_seed])
    inc = np.array([i_seed, i_seed, i_seed, i_seed, i_seed, i_seed, i_seed])
    Omega = np.array([np.random.uniform(0,2*pi) for i in range(0,NP)])
    pomega = np.array([np.random.uniform(0,2*pi) for i in range(0,NP)])
    longitude = np.array([np.random.uniform(0,2*pi) for i in range(0,NP)])

    spacing=1.05 # 2% wide of observed spacing
    dist=np.random.uniform(1.,3.)
    Pinit = np.zeros(NP)
    for ip in range(0,NP):
        Pinit[ip] = np.power(spacing,ip)*dist*P[ip]

    # Star parameters
    ms=1.

    # Create the simulation
    sim = rebound.Simulation()
    sim.integrator = "WHFast"
    rebx = reboundx.Extras(sim)
    #tf=np.random.uniform(1e2,1e4*pi)*yr
    tf = np.random.uniform(0,4)
    tf = np.power(10,tf)*yr
    n_out=101
    times = np.linspace(0, tf, n_out)

    # Add particles
    sim.add(m=ms, r=rs)
    for k in range(0,NP):
        sim.add(m=mass[k], P=P[k], e=ecc[k], inc=inc[k], Omega=Omega[k], pomega=pomega[k], l=longitude[k])

    sim.move_to_com()
    ps = sim.particles
    sim.dt = 0.05*period(0.1,ms)

    # Add extra forces if needed
    flag_gr=True
    flag_mig=True

    if flag_gr==True:
        gr = rebx.load_force("gr_potential")
        rebx.add_force(gr)
        from reboundx import constants
        gr.params["c"] = constants.C

    if flag_mig==True:
        mig = rebx.load_force("modify_orbits_forces")
        rebx.add_force(mig)
        ta, te = migration_time(np.random.uniform(3.,4.))
        ps[2].params["tau_a"] = -ta
        for k in range(0,NP):
            ps[k+1].params["tau_e"] = -te
        f = open(run_name+'.in', 'w')
        f.write("%f %f %f\n" % (ta, te, dist))
        f.close()

    orb1=np.zeros((8,n_out))
    orb2=np.zeros((8,n_out))
    flag_first=True

    # Star the integration loop
    for i,time in enumerate(times):

        ps = sim.particles
        sim.integrate(time)
        orb = sim.calculate_orbits()
        sim.simulationarchive_snapshot(run_name+'.bin')

        orb1[:,i]=np.array([sim.t, orb[0].P, orb[0].a, orb[0].e, orb[0].inc, orb[0].Omega, orb[0].pomega, orb[0].l])
        orb2[:,i]=np.array([sim.t, orb[1].P, orb[1].a, orb[1].e, orb[1].inc, orb[1].Omega, orb[1].pomega, orb[1].l])

        if verbose:
            print(("{:d} {:.2f} -- {:.4f} {:.4f} -- {:.4f} ").format(i, time/yr,
                                orb[0].a, orb[1].a, np.power(orb[1].a/orb[0].a, 1.5)))

        if (orb[0].a > 1. or orb[0].e>0.3):
            print("System went instable")
            break
        if (orb[0].a < 0.2) and (orb[0].a >= 0.1):
            if flag_first==True:
                t0=sim.t
                flag_first=False
            fd = np.exp((sim.t-t0)/t0)
            ps[2].params["tau_a"] = -ta*fd
            for k in range(0,NP):
                ps[k+1].params["tau_e"] = -te*fd
        if orb[0].a < 0.1:
            #print("Inner planet reached its current location at t=", sim.t/(2*np.pi))
            ps[2].params["tau_a"] = -np.inf
            for k in range(0,NP):
                ps[k+1].params["tau_e"] = -np.inf


def res_sim(n, verbose=False):
    run_name = 'run'+str(n)
    sa = rebound.SimulationArchive(run_name+".bin")
    sim = sa[-1]
    ps = sim.particles
    tf= np.random.uniform(0.1,3)*1e4*2*yr # 1e4*2*yr - modified after 800 runs
    n_out = np.random.uniform(2.5,4.5)
    n_out = np.power(10, n_out)
    n_out = int(n_out)
    print("nout",n_out)
    times = np.linspace(sim.t, sim.t+tf, n_out)
    rebx = reboundx.Extras(sim)
    # Add extra forces if needed
    flag_gr=True
    if flag_gr==True:
        gr = rebx.load_force("gr_potential")
        rebx.add_force(gr)
        from reboundx import constants
        gr.params["c"] = constants.C

    orb1=np.zeros((8,n_out))
    orb2=np.zeros((8,n_out))

    for i,time in enumerate(times):
        sim.integrate(time)
        orb = sim.calculate_orbits()
        orb1[:,i]=np.array([sim.t, orb[0].P, orb[0].a, orb[0].e, orb[0].inc, orb[0].Omega, orb[0].pomega, orb[0].l])
        orb2[:,i]=np.array([sim.t, orb[1].P, orb[1].a, orb[1].e, orb[1].inc, orb[1].Omega, orb[1].pomega, orb[1].l])
        if verbose:
            print(("{:d} {:.2f} -- {:.4f} {:.4f} -- {:.4f} ").format(i, time/yr,
                                orb[0].a, orb[1].a, np.power(orb[1].a/orb[0].a, 1.5)))

    find_res(orb1,orb2,n)

if __name__ == '__main__':

    description = 'Rebound integration'

    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('-n,', '--njob', dest='job_number', type=int, default=1,
                        help='Job number', required=False)
    parser.add_argument('-v,', '--verbose', dest='verbose',
                        help='Verbose mode', action='store_true')

    args = parser.parse_args()

    mig_sim(args.job_number, verbose=args.verbose)
    res_sim(args.job_number, verbose=args.verbose)
