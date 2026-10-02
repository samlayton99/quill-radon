from decimal import Decimal as D, getcontext
import json
from pathlib import Path
getcontext().prec=70
getcontext().Emin=-9999999
getcontext().Emax=9999999
pi=D('3.141592653589793238462643383279502884197169399375105820974944592307816406286')
def sinh(x):return (x.exp()-(-x).exp())/2
def cosh(x):return (x.exp()+(-x).exp())/2
H={'tanh':lambda x:(pi*x/2)/sinh(pi*x/2) if x else D(1),'gelu':lambda x:(1+x*x)*(-x*x/2).exp(),'swish':lambda x:(pi*x)**2*cosh(pi*x)/sinh(pi*x)**2 if x else D(1),'gaussian':lambda x:(-x*x/4).exp()}
orders={'tanh':1,'gelu':2,'swish':2,'gaussian':0}
count=rho_ratio_checks=0
for name,h in H.items():
 for lam in map(D,['0.12','0.25','0.7','1.5']):
  for th in [D(0),D('.1'),D('.8'),D('2.5'),pi-D('.01')]:
   vals=[(k,h(abs(th+2*pi*k)/lam)/h(th/lam)) for k in list(range(-50,0))+list(range(1,51))]
   rho=h((4*pi-th)/lam)/h((2*pi-th)/lam)
   assert 0<rho<1
   for sign in (-1,1):
    for m in range(1,6):
     successive=h((2*pi*(m+1)+sign*th)/lam)/h((2*pi*m+sign*th)/lam)
     assert successive<=rho*(1+D('1e-60')),(name,lam,th,sign,m)
     rho_ratio_checks+=1
   for r in set([0,orders[name]]):
    exact=sum(a*((abs(th/(th+2*pi*k)))**r if r else 1) for k,a in vals)
    pair=sum((th/(2*pi+sign*th))**r*h((2*pi+sign*th)/lam) for sign in [-1,1]) if r else h((2*pi-th)/lam)+h((2*pi+th)/lam)
    bound=pair/((1-rho)*h(th/lam))
    assert exact<=bound*(1+D('1e-60')),(name,lam,th,r)
    count+=1
for lam in map(D,['.25','.5','1.2']):
 for th in map(D,['.03','.3','2.0']):
  h=H['gaussian'];pair=(h((2*pi-th)/lam)+h((2*pi+th)/lam))/h(th/lam)
  zero=2*h(2*pi/lam);rhs=cosh(pi*th/lam**2)
  assert abs(pair/zero/rhs-1)<D('1e-60')
import mpmath as mp
mp.mp.dps=85
def full_tanh(lam,theta):
 s=mp.pi*theta/(2*lam);a=mp.pi**2/lam
 return mp.sinh(s)*mp.fsum(mp.csch(m*a-s)+mp.csch(m*a+s) for m in range(1,81))
def direct_pole(lam,theta,layers,u):
 h=mp.mpf(1)/16;g=lam/h;d=mp.pi/(2*g);x0=-mp.mpf(1);x=x0+h*u;w0=theta/h
 def density(z):return (mp.exp(1j*w0*(z+1j*d))-mp.exp(1j*w0*(z-1j*d)))/(2j*d)
 terms=[]
 for ell in range(layers):
  for sign in [-1,1]:
   for loc,res in [(x,-1/(2*g)),(x0,1/(2*g))]:
    p=loc+sign*1j*(2*ell+1)*d;w=mp.exp(2j*mp.pi*(p-x0)/h)
    rh=w/(w-1) if sign==1 else 1/(w-1)
    terms.append(rh*res*density(p))
 return 2j*mp.pi*mp.fsum(terms)
series_checks=direct_checks=density_checks=0
max_fraction=mp.mpf(0)
for lam in map(mp.mpf,['.12','.25','.7','1']):
 for theta in map(mp.mpf,['.03','.7','2.5']):
  s=mp.pi*theta/(2*lam);a=mp.pi**2/lam
  spectral=full_tanh(lam,theta)
  poles=4*mp.sinh(s)*mp.fsum(mp.cosh((2*ell+1)*s)/mp.expm1((2*ell+1)*a) for ell in range(80))
  assert abs(spectral/poles-1)<mp.mpf('1e-65')
  series_checks+=1
  # Compute the complex-shift density independently of its Fourier multiplier.
  h=mp.mpf(1)/16;g=lam/h;d=mp.pi/(2*g);omega=theta/h;z=mp.mpf('.37')
  density=(mp.exp(1j*omega*(z+1j*d))-mp.exp(1j*omega*(z-1j*d)))/(2j*d)
  desired=1j*omega*mp.exp(1j*omega*z)
  assert abs(density*(s/mp.sinh(s))/desired-1)<mp.mpf('1e-70')
  density_checks+=1
  for layers in [1,2,3]:
   for u in map(mp.mpf,['.17','.5','.83']):
    fraction=abs(direct_pole(lam,theta,layers,u))/(2*spectral)
    assert fraction<=1+mp.mpf('1e-65')
    max_fraction=max(max_fraction,fraction);direct_checks+=1
a=4*mp.pi**2
basic=a/mp.sinh(a)
certified=4*mp.exp(-a)/((1-mp.exp(-a))*(1-mp.exp(-2*a)))
assert certified<mp.mpf(2)**-53 and basic>mp.mpf(2)**-52
root_lambda=mp.findroot(lambda l:mp.pi**2/l/mp.sinh(mp.pi**2/l)-mp.mpf(2)**-52,(mp.mpf('.243'),mp.mpf('.245')))
limit_checks=0
for name,h in H.items():
 for lam in map(D,['.25','.7','1']):
  th=D('1e-10');r=orders[name]
  pair=sum((th/(2*pi+sign*th))**r*h((2*pi+sign*th)/lam) for sign in [-1,1])
  rho=h((4*pi-th)/lam)/h((2*pi-th)/lam)
  ratio=2*pair/((1-rho)*h(th/lam))
  rho0=h(4*pi/lam)/h(2*pi/lam)
  limiting=4*(th/(2*pi))**r*h(2*pi/lam)/(h(D(0))*(1-rho0))
  assert abs(ratio/limiting-1)<D('1e-15'),(name,lam)
  limit_checks+=1
root=Path(__file__).resolve().parent
out={'status':'passed','numeric_tail_checks':count,'successive_alias_ratio_checks':rho_ratio_checks,'precision_decimal_digits':70,
     'bridge_decimal_digits':mp.mp.dps,'replica_residue_series_checks':series_checks,
     'deconvolution_checks':density_checks,'direct_signed_pole_checks':direct_checks,
     'largest_direct_pole_fraction_of_bound':float(max_fraction),
     'basic_score_at_quarter':float(basic),'class_replica_bound_over_B_at_quarter':float(certified),
     'basic_epsilon_root':float(root_lambda),'zero_angle_limit_checks':limit_checks,
     'note':'Tail checks use the sharper exact central denominator and imply 2S<=R. Numerical checks support, but do not replace, the written proofs.'}
# A prefactor moves a bandwidth root according to the logarithmic sensitivity.
sensitivity_checks = 0
for name in ('tanh', 'gelu', 'gaussian'):
 def log_score(lam):
  a = mp.pi**2/lam
  if name == 'tanh':return mp.log(a/mp.sinh(a))
  z=(2*mp.pi/lam)**2
  return mp.log1p(z)-z/2 if name == 'gelu' else -z/4
 def slope(lam):
  a = mp.pi**2/lam
  if name == 'tanh':return a*mp.coth(a)-1
  z=(2*mp.pi/lam)**2
  return z-2*z/(1+z) if name == 'gelu' else z/2
 def bisect(log_budget):
  lo,hi=mp.mpf('.03'),mp.mpf('3' if name == 'gelu' else '1.5')
  for _ in range(280):
   mid=(lo+hi)/2
   if log_score(mid)<log_budget:lo=mid
   else:hi=mid
  return (lo+hi)/2
 for p in (24,53):
  log_eps=(1-p)*mp.log(2)
  basic_root=bisect(log_eps)
  for th in map(mp.mpf,('.005','.02','.04')):
   c=2*(th/(2*mp.pi))**orders[name]
   surrogate_root=bisect(log_eps-mp.log(c))
   lo,hi=sorted((basic_root,surrogate_root))
   change=abs(mp.log(surrogate_root/basic_root))
   assert change<=abs(mp.log(c))/slope(hi)+mp.mpf('1e-75')
   assert change>=abs(mp.log(c))/slope(lo)-mp.mpf('1e-75')
   assert abs(log_score(surrogate_root)+mp.log(c)-log_eps)<mp.mpf('1e-75')
   sensitivity_checks+=1

# The lightweight recipe must reproduce the continuously located figure roots.
from choose_lambda import choose_lambda, log_alias_score
from reproduce_figures import DISPLAY_ACTIVATIONS, DISPLAY_TARGETS, DISPLAY_PRECISIONS
import math
predictions=json.loads((root/'source_data/revised_rule_predictions.json').read_text())
recipe_checks=0
for sweep in ('width','precision'):
 for key,pr in predictions[sweep].items():
  activation,fn,_=key.split('|')
  if activation not in DISPLAY_ACTIVATIONS or fn not in DISPLAY_TARGETS:
   continue
  for arm in ('basic','refined'):
   if pr[arm]['status']!='root':continue
   kwargs={} if arm=='basic' else dict(omega_scale=pr['theta_scale']*pr['N']/2)
   answer=choose_lambda(activation,spacing=2/pr['N'],e_tol=pr['epsilon'],**kwargs)
   assert answer['status']=='threshold'
   assert abs(answer['lambda']-pr[arm]['lambda'])<1e-11
   assert answer['gamma']==answer['lambda']/(2/pr['N'])
   recipe_checks+=1
first_pair_checks=0
max_display_tail_ratio=0.0
for sweep in ('width','precision'):
 for key,pr in predictions[sweep].items():
  activation,fn,_=key.split('|')
  if activation not in DISPLAY_ACTIVATIONS or fn not in DISPLAY_TARGETS:
   continue
  if pr['refined']['status']!='root':continue
  lam=D(str(pr['refined']['lambda']));th=D(str(pr['theta_scale']));h=H[activation];r=orders[activation]
  exact_pair=sum((th/(2*pi+sign*th))**r*h((2*pi+sign*th)/lam) for sign in [-1,1])/h(th/lam)
  stable=log_alias_score(activation,float(lam),float(th))
  assert abs(float(exact_pair.ln())-stable)<1e-11
  first_pair_checks+=1
  if sweep=='width' or pr['p'] in DISPLAY_PRECISIONS:
   rho=h((4*pi-th)/lam)/h((2*pi-th)/lam)
   max_display_tail_ratio=max(max_display_tail_ratio,float(rho))
for activation in ('tanh','gelu','gaussian'):
 answer=choose_lambda(activation,spacing=2/128,omega_scale=30*math.pi/7)
 assert log_alias_score(activation,answer['lambda'],2/128*30*math.pi/7)<math.log(2**-52)
 recipe_checks+=1
assert max_display_tail_ratio<0.0002
# The omitted p=16 GELU choices are search limits, not located thresholds.
for target in DISPLAY_TARGETS:
 pr=predictions['precision'][f'gelu|{target}|16']
 assert pr['refined']['status']=='upper_search_limit'
 answer=choose_lambda('gelu',spacing=2/128,e_tol=2**-15,omega_scale=pr['theta_scale']*64)
 assert answer['status']=='upper_search_limit'
 recipe_checks+=1
out['practical_first_pair_checks']=first_pair_checks
out['max_displayed_later_pair_ratio']=max_display_tail_ratio
out['log_sensitivity_checks']=sensitivity_checks
out['standalone_recipe_checks']=recipe_checks
out['tanh_log_slope_at_basic_root']=float(mp.pi**2/root_lambda*mp.coth(mp.pi**2/root_lambda)-1)
out['gaussian_log_slope_at_basic_root']=float(104*mp.log(2))
l=mp.mpf(str(predictions['precision']['gelu|mix_2_6_10|53']['basic']['lambda']))
z=(2*mp.pi/l)**2
out['gelu_log_slope_at_basic_root']=float(z-2*z/(1+z))

(root/'figures/mathematical_checks.json').write_text(json.dumps(out,indent=2))
print(json.dumps(out))
