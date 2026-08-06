"""The S-curve is the step function seen through noise.

Theorem 1: h0 is 2 when theta is in 2*pi*Z and 0 otherwise. A step.
No real instrument measures theta exactly. Under measurement noise sigma the
OBSERVED closure rate is a sigmoid, and two things fall out:

  width of the sigmoid  <- noise in the representation
  midpoint theta*       <- how much drift the loop tolerates before it opens

Prediction from lambda_min = 2 - 2cos(theta/N) ~ theta^2/N^2:
  closure detected iff lambda_min < tau  iff  |theta| < N*sqrt(tau)
  so the tolerance window GROWS LINEARLY WITH N.
Longer loops tolerate more total drift. Same 1/N that dilutes strain.
"""
import numpy as np
rng = np.random.default_rng(0)

lam = lambda th, N: 2 - 2*np.cos(th/N)

def closure_rate(theta, N, tau, sigma, trials=4000):
    """fraction of noisy measurements that read 'closed'"""
    th = theta + rng.normal(0, sigma, trials)
    return float((lam(np.abs(th), N) < tau).mean())

def fit_logistic(x, y):
    """least squares on the logit; returns (midpoint, slope)"""
    m = (y > 1e-4) & (y < 1 - 1e-4)
    if m.sum() < 4: return np.nan, np.nan
    z = np.log(y[m]/(1-y[m]))
    A = np.vstack([x[m], np.ones(m.sum())]).T
    k, b = np.linalg.lstsq(A, z, rcond=None)[0]
    return -b/k, k

print("="*76)
print("1. THE STEP BECOMES A SIGMOID")
print("="*76)
N, tau = 12, 0.02
th = np.linspace(0, 3.0, 160)
print(f"  N={N}  tau={tau}   predicted midpoint N*sqrt(tau) = {N*np.sqrt(tau):.4f}")
print(f"\n  {'sigma':>8}{'fitted midpoint':>18}{'fitted slope':>15}{'1/slope':>10}")
for sg in [0.02, 0.05, 0.10, 0.20, 0.40]:
    y = np.array([closure_rate(t, N, tau, sg) for t in th])
    mid, k = fit_logistic(th, y)
    print(f"  {sg:>8.3f}{mid:>18.4f}{k:>15.2f}{-1/k:>10.4f}")
print("\n  midpoint is stable; slope tracks 1/sigma. The sigmoid IS a thermometer.")

print("\n" + "="*76)
print("2. SLOPE vs NOISE: is 1/slope linear in sigma?")
print("="*76)
sgs = np.array([0.02,0.04,0.06,0.08,0.12,0.16,0.24,0.32,0.40])
inv = []
for sg in sgs:
    y = np.array([closure_rate(t, N, tau, sg) for t in th])
    _, k = fit_logistic(th, y); inv.append(-1/k)
inv = np.array(inv)
A = np.vstack([sgs, np.ones(len(sgs))]).T
m, c = np.linalg.lstsq(A, inv, rcond=None)[0]
r2 = 1 - ((inv - (m*sgs+c))**2).sum()/((inv-inv.mean())**2).sum()
print(f"  1/slope = {m:.3f}*sigma + {c:+.4f}    R^2 = {r2:.5f}")
print("  Linear. Recovering sigma from the observed S-curve is one division.")

print("\n" + "="*76)
print("3. TOLERANCE WINDOW GROWS LINEARLY WITH LOOP LENGTH")
print("="*76)
print(f"  {'N':>4}{'predicted N*sqrt(tau)':>24}{'fitted midpoint':>18}{'ratio':>8}")
mids = []
for Nn in [4, 6, 8, 12, 16, 24, 32]:
    tt = np.linspace(0, 1.4*Nn*np.sqrt(tau)+0.6, 200)
    y = np.array([closure_rate(t, Nn, tau, 0.08) for t in tt])
    mid, _ = fit_logistic(tt, y); mids.append(mid)
    print(f"  {Nn:>4}{Nn*np.sqrt(tau):>24.4f}{mid:>18.4f}{mid/(Nn*np.sqrt(tau)):>8.4f}")
Ns = np.array([4,6,8,12,16,24,32], float); mids = np.array(mids)
A = np.vstack([Ns, np.ones(len(Ns))]).T
m2, c2 = np.linalg.lstsq(A, mids, rcond=None)[0]
r2b = 1 - ((mids-(m2*Ns+c2))**2).sum()/((mids-mids.mean())**2).sum()
print(f"\n  midpoint = {m2:.4f}*N {c2:+.4f}   R^2 = {r2b:.5f}")
print(f"  slope {m2:.4f} vs predicted sqrt(tau) = {np.sqrt(tau):.4f}")
print("\n  A loop twice as long tolerates twice the total drift before it opens.")
print("  Same 1/N as the strain dilution law, wearing a different hat.")
