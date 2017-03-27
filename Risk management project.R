allreturn = read.csv("~/Desktop/Risk Project/log returns of indices.csv",header = TRUE)
spx.return =as.numeric(as.character(allreturn[,2][1:3500]))
vix.or = allreturn[,3]
vix.return = as.numeric(as.character(allreturn[,3][1:3500]))
ndx.return = as.numeric(as.character(allreturn[,4][1:3500]))
vxn.return = as.numeric(as.character(allreturn[,5][1:3500]))
spx.return = diff(log(option_price), lag=1)
#invariants test
first_half = vix.or[1:2019]
second_half = vix.or[2020:4037]
hist(first_half,freq = TRUE,xlab = 'Log Returen',ylab = 'Count',main = 'Distribution of first-half example')
hist(second_half,xlab = 'Log Returen',ylab = 'Count',main = 'Distribution of second-half example')
plot(first_half,second_half,xlab = 'Log Return in T1',ylab ='Log Return in T2')
#volatility graph showing
vol = c()
for (i in 1:4000){
  vol<-c(vol,sqrt(var(vix.or[i:(i+37)])))
}
plot(vol,type = 'l', xlab = 'Time In Date')
#Motne Carlo

##QQ plot
qqnorm(vix.return)
qqline(vix.return)
#long tail on both side
#Optimization
#1.Mean- variance theorem
require(tseries)
risk.free = 0.01
#porflio.set = data.frame(spx.return,vix.return,ndx.return,vxn.return)
porflio.set = diff(log(option_price), lag=1)
averet = matrix(colMeans(porflio.set),nrow=1)
rcov = cov(porflio.set)
port.sol = portfolio.optim(x= averet,covmat = rcov,rf = risk.free)
w = port.sol$pw 
r = port.sol$pm
table.sol = portfolio.optim(x= averet,covmat = rcov,shorts = TRUE,reslow = rep(0.1,4))
effFrontier = function (averet, rcov, nports = 20, shorts=T, wmax=1)
{
  mxret = max(abs(averet))
  mnret = -mxret
  n.assets = ncol(averet)
  reshigh = rep(wmax,n.assets)
  if( shorts )
  {
    reslow = rep(-wmax,n.assets)
  } else {
    reslow = rep(0,n.assets)
  }
  min.rets = seq(mnret, mxret, len = nports)
  vol = rep(NA, nports)
  ret = rep(NA, nports)
  for (k in 1:nports)
  {
    port.sol = NULL
    try(port.sol <- portfolio.optim(x=averet, pm=min.rets[k], covmat=rcov,
                                    reshigh=reshigh, reslow=reslow,shorts=shorts),silent=T)
    if ( !is.null(port.sol) )
    {
      vol[k] = sqrt(as.vector(port.sol$pw %*% rcov %*% port.sol$pw))
      ret[k] = averet %*% port.sol$pw
    }
  }
  return(list(vol = vol, ret = ret))
}
ef_frontier = effFrontier(averet, rcov, nports = 30, shorts=T, wmax=1)
plot(ef_frontier$vol,ef_frontier$ret,type = 'l',col = 'red',main = 'Efficient Frontier',xlab = 'Anually Volatitliy',ylab = 'Anually Return')
#Max sharp ratio weight allowed shorting or not
maxSharpe = function (averet, rcov, shorts=F, wmax = 1)
{
  optim.callback = function(param,averet,rcov,reshigh,reslow,shorts)
  {
    port.sol = NULL
    try(port.sol <- portfolio.optim(x=averet, pm=param, covmat=rcov,
                                    reshigh=reshigh, reslow=reslow, shorts=shorts), silent = T)
    if (is.null(port.sol)) {
      ratio = 10^9
    } else {
      m.return = averet %*% port.sol$pw
      m.risk = sqrt(as.vector(port.sol$pw %*% rcov %*% port.sol$pw))
      ratio = -m.return/m.risk
      assign("w",port.sol$pw,inherits=T)
    }
    return(ratio)
  }
  ef = effFrontier(averet=averet, rcov=rcov, shorts=shorts, wmax=wmax, nports = 100)
  n = ncol(averet)
  reshigh = rep(wmax,n)
  if( shorts ) {
    reslow = -reshigh
  } else {
    reslow = rep(0,n)
  }
  max.sh = which.max(ef$ret/ef$vol)
  w = rep(0,ncol(averet))
  xmin = optimize(f=optim.callback, interval=c(ef$ret[max.sh-1], upper=ef$ret[max.sh+1]),
                  averet=averet,rcov=rcov,reshigh=reshigh,reslow=reslow,shorts=shorts)
  return(w)
}
maxSharpe(averet, rcov, shorts=T, wmax = 1)
maxSharpe(averet, rcov, shorts=F, wmax = 1)
# mean-CVaR optimization
install.packages('Rglpk')
rmat = unname(porflio.set)
cvarOpt = function(averet, rmat,alpha=0.05, rmin=0, wmin=0, wmax=1, weight.sum=1)
{
  require(Rglpk)
  n = ncol(rmat) # number of assets
  s = nrow(rmat) # number of scenarios i.e. periods
  #averet = colMeans(rmat)
  # creat objective vector, constraint matrix, constraint rhs
  cmd = as.data.frame(cbind(rbind(1,averet),matrix(data=0,nrow=2,ncol=s+1)))
  cmd2 = setNames(cbind(rmat,diag(s),1),colnames(cmd))
  Amat = rbind(cmd,cmd2)
  objL = c(rep(0,n), rep(-1/(alpha*s), s), -1)
  bvec = c(weight.sum,rmin,rep(0,s))
  # direction vector
  dir.vec = c("==",">=",rep(">=",s))
  # bounds on weights
  bounds = list(lower = list(ind = 1:n, val = rep(wmin,n)),
                upper = list(ind = 1:n, val = rep(wmax,n)))
  res = Rglpk_solve_LP(obj=objL, mat=Amat, dir=dir.vec, rhs=bvec,
                       types=rep("C",length(objL)), max=T, bounds=bounds)
  w = as.numeric(res$solution[1:n])
  cvar = as.numeric(res$solution[length(res$solution)])
  return(list(w=w,status=res$status,min_cvar=cvar))
}
cvarOpt (colMeans(rmat),rmat, alpha=0.01, rmin=0, wmin=0, wmax=1, weight.sum=1)
#Advanced CVaR optimization
library(VineCopula)
u = pobs(porflio.set)[,1]
v = pobs(porflio.set)[,2]
w = pobs(porflio.set)[,3]
k = pobs(porflio.set)[,4]
selectedCopula = BiCopSelect(u,w,familyset=NA)#t for spx and ndx
selectedAnother = BiCopSelect(v,k,familyset=NA)
selectedAnother$family #t for vix and vxn
#under t copula
t.cop = tCopula(dim=2)
fit.copula =  fitCopula(t.cop,cbind(u,w),method='ml')
rho.1 = coef(fit.copula)[1]
df.1 = coef(fit.copula)[2]
persp(tCopula(dim=2,rho.1,df=df.1),dCopula)
sim_uw = rCopula(3500,tCopula(dim=2,rho.1,df=df.1))
plot(sim_uw[,1],sim_uw[,2],pch='.',col='blue')
fit.copula2 =  fitCopula(t.cop,cbind(v,k),method='ml')
rho.2 = coef(fit.copula2)[1]
df.2 = coef(fit.copula2)[2]
persp(tCopula(dim=2,rho.2,df=df.2),dCopula)
sim_vk = rCopula(3500,tCopula(dim=2,rho.2,df=df.2))
t_copula = data.frame(cbind(sim_uw,sim_uw))
cvarOpt (colMeans(t_copula),t_copula, alpha=0.05, rmin=0, wmin=0, wmax=1, weight.sum=1)
#Gaussian copula
copula_cov = cor(porflio.set)
A = t(chol(copula_cov))
z = rnorm(4,0,1)
x = A%*%z
gaussian_copula = sapply(x,pnorm)
library(MASS)
g_n = mvrnorm(n = 3500, colMeans(porflio.set), cov(porflio.set), tol = 1e-6)
cvarOpt (colMeans(g_n),g_n, alpha=0.01, rmin=0, wmin=0, wmax=1, weight.sum=1)
#Student-t copula 
library(mvtnorm)
t_n = rmvt(n=3500, sigma = cov(porflio.set), df = 4)
df = 4
zt = rnorm(4,0,1)
s = rchisq(1,4)
y = sqrt(3/s)*x
t_copula =as.vector(dt(y,4))
cvarOpt (colMeans(t_n),t_n, alpha=0.05, rmin=0, wmin=0, wmax=1, weight.sum=1)
#equally weighted
cvarOpt (colMeans(porflio.set),porflio.set, alpha=0.05, rmin=0, wmin=0.25, wmax=0.25, weight.sum=1)
#dependence test
pairs(porflio.set)
volatility = apply(as.data.frame(cbind(spx.return,vix.return,ndx.return,vxn.return)),2,var)
volatility = sqrt(volatility)
price.of.stocks <- read.table("~/Desktop/Risk Project/price of stocks.xlsx", header=TRUE, quote="\"")
volitility = c(0.01219400,0.06806857,0.01401617,0.05768033)
price = as.matrix(price.of.stocks)
option_price = matrix(0,3500,4)
for(i in 1:4){
  for(j in 1:3500){
    model = GBSOption(TypeFlag = "c", S = price[j,i], X = price[j,i], Time = 1/6, r = 0.01,
                      b = 0, sigma = volatility[i])
    option_price[j,i] = model@price
  }
}
