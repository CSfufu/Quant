import datetime
from datetime import datetime as dt
import numpy as np
from numpy import exp, log, sqrt
import scipy.stats as ss
from scipy.stats import norm
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.formula.api as sm

class OptionAnalytics:
    def __init__(self, option_chain, expiry, today):
        self.expiry = expiry
        self.today = today
        if not type(today) == datetime.datetime:
            self.today = dt.strptime(today, '%Y-%m-%d')
        self.calls, self.puts = option_chain
        self.ks_c = self.calls['strike']
        self.cs = (self.calls['bid'] + self.calls['ask'])/2 
        self.ks_p = self.puts['strike'] 
        self.ps = (self.puts['bid'] + self.puts['ask'])/2         
        # strikes that are traded for both calls and puts
        self.ks = np.array([])
        for k in self.ks_c:
            if k in np.array(self.ks_p):
                self.ks = np.concatenate([self.ks, [k]])

        self.mids_call = np.array([])
        for k in self.ks:
            self.calls[self.calls['strike'] == k]['bid']
            bid = self.calls[self.calls['strike'] == k]['bid']
            ask = self.calls[self.calls['strike'] == k]['ask']
            self.mids_call = np.concatenate([self.mids_call, np.array((bid + ask)/2)])

        self.mids_put = np.array([])
        for k in self.ks:
            bid = self.puts[self.puts['strike'] == k]['bid']
            ask = self.puts[self.puts['strike'] == k]['ask']
            self.mids_put = np.concatenate([self.mids_put, np.array((bid + ask)/2)])
            
        tmp = self.imp_vols()
        self.ivs, self.s_adj = tmp['imp_vols'], tmp['s_adj']
        
    # put-call parity plot 
    def plot_parity(self):
        plt.figure(figsize=(9, 5))
        plt.plot(self.ks, self.mids_call - self.mids_put, 'r.--')
        plt.ylabel(r'$C - P$', fontsize=12)
        plt.xlabel(r'$K$', fontsize=12)
        plt.title(f'Expiry: {self.expiry}', fontsize=15);
        return None
            
    def plot_arb(self):
        # monotonicity and convexity for option premia vs strikes
        fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
        axes[0].plot(self.ks_c, self.cs, 'bo--')
        axes[0].set_ylabel('Option mid price', fontsize=12)
        axes[0].set_xlabel('Strike', fontsize=12)
        axes[1].plot(self.ks_p, self.ps, 'ro--')
        axes[1].set_xlabel('Strike', fontsize=12);
        return None
        
    # plot implied vol
    def plot_imp_vols1(self):
        ivs_c = self.calls['impliedVolatility']
        ivs_p = self.puts['impliedVolatility']
        # plot 
        plt.figure(figsize=(10, 6))
        plt.plot(self.ks_p, ivs_p, 'ro--', label='implied vol from puts')
        plt.plot(self.ks_c, ivs_c, 'bo--', label='implied vol from calls')
        plt.title(f'Expiration date: {self.expiry}', fontsize=15)
        plt.xlabel('Strikes', fontsize=12)
        plt.ylabel('Impliied Volatilities', fontsize=12)
        plt.legend();
        return None
        
    # Black-Scholes formula for call
    def bs_call(self, s, K, t, sigma, r=0):
        d1 = (log(s/K) + r*t)/(sigma*sqrt(t)) + sigma*sqrt(t)/2
        d2 = d1 - sigma*sqrt(t)
        return s*norm.cdf(d1) - K*exp(-r*t)*norm.cdf(d2)
    
    # function calculating implied vol by the bisection method
    def bs_impvol_call(self, s0, K, T, C, r=0):
        K = np.array([K])
        n = len(K)
        sigmaL, sigmaH = 1e-10*np.ones(n), 10*np.ones(n)
        CL, CH = self.bs_call(s0, K, T, sigmaL, r), self.bs_call(s0, K, T, sigmaH, r)
        while np.mean(sigmaH - sigmaL) > 1e-10:
            sigma = (sigmaL + sigmaH)/2
            CM = self.bs_call(s0, K, T, sigma, r)
            CL = CL + (CM < C)*(CM - CL)
            sigmaL = sigmaL + (CM < C)*(sigma - sigmaL)
            CH = CH + (CM >= C)*(CM - CH)
            sigmaH = sigmaH + (CM >= C)*(sigma - sigmaH)    
        return sigma[0]
    
    # calculate implied vols
    def imp_vols(self):
        # regress call - put over strike K 
        # apply put-call parity 
        df = {'CP': self.mids_call - self.mids_put, 'Strike': self.ks}
        result = sm.ols(formula='CP ~ Strike', data=df).fit()
        s_adj, pv = result.params[0], -result.params[1]
        ks_pv = self.ks*pv
        days_to_expiry = (dt.strptime(self.expiry, '%Y-%m-%d') - self.today).days
        imp_vols = self.bs_impvol_call(s_adj, ks_pv, days_to_expiry/365, self.mids_call, r=0)        
        return {'imp_vols': imp_vols, 'pv': pv, 's_adj': s_adj}
    
    # plot implied vol
    def plot_imp_vols2(self):
        plt.figure(figsize=(10, 6))
        y = self.ivs[self.ivs>0.001]
        x = self.ks[self.ivs>0.001]
        plt.plot(log(x/self.s_adj), y, 'b.--')
        plt.plot(log(x/self.s_adj), y, 'r.')
        plt.xlabel('logmoneyness', fontsize=12)
        plt.ylabel('implied volatilities', fontsize=12)
        plt.title('Implied volatilities vs Logmoneyness', fontsize=15);
        return None
    
    def __call__(self):
        pass