import numpy as np
import scipy.linalg as la
import scipy.sparse as sparse
import scipy.special as spesh
from mpmath import mp

mp.dps = 24

def jn(n,zf):
    z = mp.mpmathify(zf)
    return mp.mpmathify(mp.sqrt(mp.pi/(2*z))*mp.besselj(mp.mpf(n+0.5),z))
def myj0(zc): z = mp.mpmathify(zc); return mp.sin(z)/z
def myj1(zc): z = mp.mpmathify(zc); return mp.sin(z)/(z*z)-mp.cos(z)/z
def myj2(zc): z = mp.mpmathify(zc); return mp.sin(z)*(mp.mpf(3.0)/(z*z)-mp.mpf(1))/z -mp.mpf(3)*mp.cos(z)/(z*z)

# theory for homogeneous media. returns J/sin(theta)
def funiform(r,muref,condref,om,B0,R,mu0):
    k2ref = 1j*muref*condref*om
    kref = np.sqrt(k2ref)
    A0 = 9*B0*k2ref/(2*kref*((muref+2*mu0)*myj0(kref*R)+(muref-mu0)*myj2(kref*R)))
    f = np.zeros(r.shape,complex)
    for i in range(len(r)):
        f[i] = np.complex(A0*myj1(kref*r[i]))
    return f

def Puniform(mu,cond,om,B0,R,mu0): # Ohmic power losss
    k2 = 1j*mu*cond*om
    k = np.sqrt(k2)
    kstar = np.conj(k)
    j0kR = myj0(k*R); j0kRstar = myj0(kstar*R);
    j1kR = myj1(k*R); j1kRstar = myj1(kstar*R);
    j2kR = myj2(k*R)
    A = 9*B0*k2/(2*k*((mu+2*mu0)*j0kR+(mu-mu0)*j2kR))
    P = 1/2*np.abs(A)**2/cond*8*np.pi/3*(1j/2*R**2)*(j0kR*j1kRstar/kstar-j0kRstar*j1kR/k)
    return P.real

def fheter(r,mu,cond,om,B0,R,mu0): # J/sin(theta)
    dr = r[1]-r[0]
    nr = len(r)
    dr2 = dr**2
    fR = 1.0 # A0*j1(k*R) # exact j1(k*R)
    fsmallr = 0
    k2 = 1j*mu*cond*om
    deld2x = np.zeros(nr)
    d2 = 1/(mu*cond)
    rhom,rhoe = 1/mu,1/cond
    drhom, drhoe, d2rhoe = np.zeros(nr), np.zeros(nr), np.zeros(nr)
    drhom[1:-1] = (rhom[2:]-rhom[:-2])/(2*dr)
    drhoe[1:-1] = (rhoe[2:]-rhoe[:-2])/(2*dr)
    d2rhoe[1:-1] = (rhoe[2:]-2*rhoe[1:-1]+rhoe[:-2])/dr**2
    rhofacf = 2*cond/r*drhoe + mu/r*drhom + cond*d2rhoe + mu*cond*drhom*drhoe
    rhofacdf = 2*cond*drhoe + mu*drhom
    #deld2x[1:-1]  = -(cond[2:]-cond[:-2])/(2*dr)/cond[1:-1]
    #deld2x[1:-1] += -(mu[2:]-mu[:-2])/(2*dr)/mu[1:-1]
    deld2x *= 0

    # build del2r op:
    onec = np.ones((nr),complex)
    diag = np.copy(onec)
    offplus, offminus = np.zeros(nr-1,complex), np.zeros(nr-1,complex)
    offplus[1:]   = onec[1:-1]*(1/dr2+2/(r[1:-1]*2*dr))  # 
    offminus[:-1] = onec[1:-1]*(1/dr2-2/(r[1:-1]*2*dr))  # 
    diag[1:-1] = onec[1:-1]*(-2/dr2 + (k2[1:-1] - 2/r[1:-1]**2))   # diag part
    # now build in for heterogenous material, radial varying mu, cond....
    offplus[1:]   += rhofacdf[1:-1]*(1/(2*dr))  # 
    offminus[:-1] += rhofacdf[1:-1]*(-1/(2*dr))  # 
    diag[1:-1]    += rhofacf[1:-1]
    b = np.zeros(nr,complex)
    b[-1] = fR
    b[0] = fsmallr
    ab = np.zeros((3,nr),complex)
    ab[0,1:]=offplus
    ab[1,:] = diag
    ab[2,:-1]=offminus
    f = la.solve_banded((1,1),ab,b)

    # now need to scale the solution to target B field:
    # just inside of the sphere...equatorial plane
    BeqR = -(f[-2]/r[-2] + (f[-1]-f[-3])/(2*dr))/(1j*om*cond[-2])
    BzR = 2*f[-2]/(1j*om*cond[-2]*r[-2])
    C  = 3*1J*mu[-2]*cond[-2]*om*B0*r[-2]/((mu[-2] + mu0)*2*f[-2]+2*mu0*r[-2]*(f[-1]-f[-3])/(2*dr)) 
    #B0this = 1/3*(BzR-2*BeqR*mu0/mu[-1])
    #fx = f*np.abs(B0)/np.abs(B0this)
    f *= C
    return f #,fx

def Poweravg(r,f,mu,cond,om,B0,R,mu0):
    dr = r[1:]-r[:-1]
    Pavg = 0.5*2*np.pi*4/3*np.sum(dr*r[1:]**2 * np.abs(f[1:])**2/cond[1:])
    return Pavg

def MagMo(r,f,mu,cond,om,B0,R,mu0):
    dr = r[1:]-r[:-1]
    mmofree =2*np.pi*(1/2)*np.sum(dr*r[1:]**3*f[1:])*4/3
    chi = mu/mu0-1
    dr = dr[1:]
    # Bz = 2*cos_h**2 * 1/r[1:]*f[1:] + sin_h^2/r[1:]*(r[2:]*f[2:]-r[:-2]*f[:-2])
    # integrated over theta (*sin_h from volume element):
    Bz = 2 * 2/3 * 1/r[1:-1]*f[1:-1] + 4/3/r[1:-1]*(r[2:]*f[2:]-r[:-2]*f[:-2])/(2*dr)
    Bz *= 1/(1j*om*cond[1:-1])
    
    mmobound = 2*np.pi*np.sum(r[1:-1]**2 * Bz/mu0*chi[1:-1]/(1+chi[1:-1])*dr)
    # print('mbound',format(complex(mmobound),'g'),chi[1])
    # Bz:= (cos(h)*1/r/sin(h)*diff(sin(h)*Jr(r),h) + sin(h)/r*diff(r*Jr(r),r))/(I*k*v*sigma):
    # mmobound 2*np.pi*(2/2)*int(int(Bz*chi/(1+chi)/mu0*sin(h)*r^2,h=0..Pi),r=0..R):
    return mmofree + mmobound

def MagMotheory(mu,cond,om,B0,R,mu0):
    kk = np.sqrt(1j*mu*cond*om); j0kR=myj0(kk*R); j2kR = myj2(kk*R)
    return 2*np.pi*R**3*B0/mu0*((2*mu+mu0)*j2kR+2*(mu-mu0)*j0kR)/((mu+2*mu0)*j0kR+(mu-mu0)*j2kR)

from scipy.special import comb
def smoothstep(x, x_min=0, x_max=1, N=1):    
        x = np.clip((x - x_min) / (x_max - x_min), 0, 1)
        res = 0
        for n in range(0, N + 1):
            res += comb(N + n, n) * comb(2 * N + 1, N - n) * (-x) ** n
        res *= x ** (N + 1)
        return res
        
def smoothxition(r,r1,r2,a,b):
    return a + smoothstep(r,r1,r2,4)*(b-a)

