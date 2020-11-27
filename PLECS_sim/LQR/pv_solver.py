import numpy as np
import numpy.polynomial.polynomial as poly
import cvxpy as cp
import random
import csv
import mosek

class sos_tracker():
    def __init__(self, buffer_size=24, tolerance=12, eps=1e-9, deg=6, Vbase=250, Ibase=75, freq=10):
        self.buffer_size = buffer_size
        self.buffer_I = np.array([])
        self.buffer_V = np.array([])
        self.tolerance = tolerance
        self.eps = eps
        self.Vbase = Vbase
        self.Ibase = Ibase
        self.deg = deg
        self.f = None
        self.g = None
        self.h = None
        self.count = 0
        self.freq = freq
        self.momentum = 0.5
        self.sos_prob = None

    def update_curve(self, I, V):
        I = np.array(I)
        V = np.array(V)
        # print(I)
        # print(V)
        # print(self.Vbase)
        if self.buffer_size == None:
            self.buffer_I = I
            self.buffer_V = V
            self.solve_sos()
        elif len(self.buffer_I)<self.buffer_size:
            self.buffer_I = np.append(self.buffer_I,I)
            self.buffer_V = np.append(self.buffer_V,V)
            start = len(self.buffer_I)-self.buffer_size
            if start<0:
                start = 0
            self.buffer_I = self.buffer_I[start:]
            self.buffer_V = self.buffer_V[start:]
#            print(self.buffer_I.shape)
#            print(self.buffer_V.shape)
#            print(self.Vbase)
            self.solve_sos()
        else:
            err = self.estimate(I,V/self.Vbase)
            if err>3*self.tolerance or self.count==0:
                self.buffer_I = np.append(self.buffer_I,I)
                self.buffer_V = np.append(self.buffer_V,V)
                start = len(self.buffer_I)-self.buffer_size
                if start<0:
                    start = 0
                self.buffer_I = self.buffer_I[start:]
                self.buffer_V = self.buffer_V[start:]
#                print(self.buffer_I)
#                print(self.buffer_V)
#                print(self.Vbase)
                self.solve_sos()
                self.count = 1
            elif err>self.tolerance:
                self.count = (self.count+1)%self.freq
                self.buffer_I = np.append(self.buffer_I,I)
                self.buffer_V = np.append(self.buffer_V,V)
                start = len(self.buffer_I)-self.buffer_size
                if start<0:
                    start = 0
                self.buffer_I = self.buffer_I[start:]
                self.buffer_V = self.buffer_V[start:]
                self.solve_rlms()
            else:
                idx = np.random.permutation(len(self.buffer_I))
                idx_min = np.argmin(V)
                idx_max = np.argmax(V)
                self.buffer_I[idx[0]] = I[idx_min]
                self.buffer_I[idx[1]] = I[idx_max]
                self.buffer_V[idx[0]] = V[idx_min]
                self.buffer_V[idx[1]] = V[idx_max]
    
    def find_vref(self, p_ref, v_ref, clipping = True):
        i_ref = p_ref/v_ref
        if self.estimate(i_ref, v_ref/self.Vbase)<(p_ref*self.eps*100) and self.count!=1:
            return p_ref, v_ref
        else:
            p_est, v_est = self.track(p_ref, v_ref/self.Vbase)
            if clipping:
                max_v = max(self.buffer_V)
                min_v = min(self.buffer_V)
                mid_v = np.mean(self.buffer_V)
                v_est = np.clip(v_est, min_v*2-mid_v, max_v*2-min_v)
                v_est = v_est*self.momentum+(1-self.momentum)*v_ref/self.Vbase
            return p_est*self.Vbase, v_est*self.Vbase

    def estimate(self, I, V):
        # p_est = poly.polyval(V/self.Vbase, -self.f)
        vect = self.vectorize1d(self.deg,V/self.Vbase)
        p_est = np.dot(vect,-self.f)
        p_true = np.multiply(I,V)
        print(p_est*self.Ibase*self.Vbase,p_true)
        return np.linalg.norm(p_est*self.Ibase*self.Vbase-p_true)
    
    def solve_sos(self):
        if len(self.buffer_I)>self.buffer_size:
            self.buffer_I = self.buffer_I[-self.buffer_size:]
            self.buffer_V = self.buffer_V[-self.buffer_size:]
        d = int(self.deg/2+1)
        self.x_sos = cp.Variable(int(d*(d+1)/2))
        self.A_sos = cp.Parameter((self.buffer_size, int(d*(d+1)/2)))
        self.b_sos = cp.Parameter(self.buffer_size)
        obj = cp.Minimize(cp.sum_squares(self.A_sos@self.x_sos + self.b_sos))
        H = cp.Variable((d-1, d-1), PSD = True)
        con = []
        k = 0
        for i in range(1,d):
            for j in range(1,i+1):
                deg = self.deg+2-i-j
                if i!=j :
                    con += [H[i-1,j-1]+H[j-1,i-1] == deg*(deg-1)*self.x_sos[k]]
                else:
                    con += [H[j-1,i-1] == deg*(deg-1)*self.x_sos[k]]
                k+= 1
        self.sos_prob = cp.Problem(obj, con)
        self.A_sos.value = self.vectorize2d(d)
        self.b_sos.value = np.multiply(self.buffer_I, self.buffer_V)/self.Vbase/self.Ibase
        # self.sos_prob.solve(solver = cp.CVXOPT,warm_start = False)
        self.sos_prob.solve(solver = cp.MOSEK,mosek_params={mosek.iparam.optimizer:mosek.optimizertype.free}, verbose=False,warm_start = True)
        # try:
        # 	# self.sos_prob.solve(solver = cp.SCS,verbose=False,warm_start = True)
        # 	# self.sos_prob.solve(solver = cp.CVXOPT,warm_start = False)
        #     self.sos_prob.solve(solver = cp.MOSEK,mosek_params={mosek.iparam.optimizer:mosek.optimizertype.free}, verbose=False,warm_start = True)
        # except:
        #     print("Resolve optimization")
        #     obj = cp.Minimize(cp.sum_squares(self.A_sos@self.x_sos + self.b_sos)+1*cp.sum_squares(self.x_sos))
        #     self.sos_prob = cp.Problem(obj, con)
        #     # self.sos_prob.solve(solver = cp.CVXOPT,warm_start = False)
        #     self.sos_prob.solve(solver = cp.MOSEK,mosek_params={mosek.iparam.optimizer:mosek.optimizertype.conic}, verbose=False,warm_start = False)
        print(f"x value is: {self.x_sos.value}")
        self.f = np.zeros(self.deg+1)
        k = 0
        for i in range(1,d+1):
            for j in range(1,i+1):
                self.f[i+j-2] += self.x_sos.value[k]
                k+=1
            self.g = [v*(self.deg-idx) for idx,v in enumerate(self.f[:-1])]
            self.h = [v*(self.deg-idx-1) for idx,v in enumerate(self.g[:-1])]
#            print(self.b_sos.value+np.dot(self.A_sos.value,self.x_sos.value))


#    def solve_sos(self):
#        if len(self.buffer_I)>self.buffer_size:
#            self.buffer_I = self.buffer_I[-24:]
#            self.buffer_V = self.buffer_V[-24:]
#        # self.buffer_size = max(len(self.buffer_I),self.buffer_size)
#        d = int(self.deg/2+1)
#        # if self.sos_prob is None and len(self.buffer_I) == self.buffer_size:
#        if self.sos_prob is None:
#            self.x_sos = cp.Variable(int(d*(d+1)/2))
#            self.A_sos = cp.Parameter((self.buffer_size, int(d*(d+1)/2)))
#            self.b_sos = cp.Parameter(self.buffer_size)
#            obj = cp.Minimize(cp.sum_squares(self.A_sos@self.x_sos + self.b_sos))
#            H = cp.Variable((d-1, d-1), PSD = True)
#            con = []
#            k = 0
#            print("adfjasd")
#            for i in range(1,d):
#                for j in range(1,i+1):
#                    deg = self.deg+2-i-j
#                    if i!=j :
#                        con += [H[i-1,j-1]+H[j-1,i-1] == deg*(deg-1)*self.x_sos[k]]
#                    else:
#                        con += [H[j-1,i-1] == deg*(deg-1)*self.x_sos[k]]
#                    k+= 1
#            self.sos_prob = cp.Problem(obj, con)
#            # print(self.buffer_I)
#            # print(self.buffer_V)
#            # self.sos_prob = cp.Problem(obj)
#        elif len(self.buffer_I) < self.buffer_size and len(self.buffer_I) >= self.buffer_size/2:
#            d = int(self.deg/2+1)
#            x = cp.Variable(int(d*(d+1)/2))
#            A_sos = self.vectorize2d(d)
#            b_sos = np.multiply(self.buffer_I, self.buffer_V)
#            obj = cp.Minimize(cp.sum_squares(A_sos@x + b_sos))
#            H = cp.Variable((d-1, d-1), PSD = True)
#            con = []
#            k = 0
#            print("aldkfjkladdklfajdjkldsajfks")
#            for i in range(d-1):
#                for j in range(i+1):
#                    deg = self.deg-i-j
#                    if i!=j :
#                        con += [H[i,j]+H[j,i] == deg*(deg-1)*x[k]]
#                    else:
#                        con += [H[i,j] == deg*(deg-1)*x[k]]
#                    k+= 1
#            self.sos_prob = cp.Problem(obj, con)
#            self.sos_prob.solve()
#            # self.sos_prob.solve(solver=cp.SCS,verbose=True)
#            self.f = np.zeros(self.deg+1)
#            k = 0
#            for i in range(d):
#                for j in range(i+1):
#                    self.f[i+j] += x.value[k]
#                    k+=1
#            self.g = [v*(self.deg-idx) for idx,v in enumerate(self.f[:-1])]
#            self.h = [v*(self.deg-idx-1) for idx,v in enumerate(self.g[:-1])]
#            return
#        elif len(self.buffer_I) < self.buffer_size/2:
#            raise RuntimeError('Not enough data to estimate the PV model!')
#        self.A_sos.value = self.vectorize2d(d)
#        self.b_sos.value = np.multiply(self.buffer_I, self.buffer_V)*self.Vbase
##        self.sos_prob.solve(verbose=True,warm_start = True,alpha=1)
#        print("Matrix A:")
#        print(self.A_sos.value.shape)
#        print("Matrix b:")
#        print(self.b_sos.value.shape)
#        self.sos_prob.solve(solver = cp.MOSEK, verbose=True,warm_start = True)
##        self.sos_prob.solve(solver = cp.CVXOPT, max_iters=1000,verbose=True,warm_start = True, kktsolver='robust',reltol=1e-5,abstol=1e-5,feastol=1e-5)
#        print(f"x value is: {self.x_sos.value}")
#        self.f = np.zeros(self.deg+1)
#        k = 0
#        for i in range(1,d+1):
#            for j in range(1,i+1):
#                self.f[i+j-2] += self.x_sos.value[k]
#                k+=1
#        self.g = [v*(self.deg-idx) for idx,v in enumerate(self.f[:-1])]
#        self.h = [v*(self.deg-idx-1) for idx,v in enumerate(self.g[:-1])]
#        print(self.b_sos.value+np.dot(self.A_sos.value,self.x_sos.value))

    def solve_rlms(self):
        if self.P_rlms is None:
            self.P_rlms = np.eye(self.deg+1)*5
        u = self.vectorize1d(self.deg, self.buffer_V)
        Pu = np.matmul(self.P_rlms,u)
        k = 1.25*np.matmul( Pu, np.linalg.inv(np.eye(len(self.buffer_I)) + 1.25* np.matmul(u.T, Pu)))
        err = np.multiply(self.buffer_I, self.buffer_V) - poly.polyval(self.buffer_V, self.f)
        self.f = self.f + np.matmul(k, err)
        self.g = [v*(self.deg-idx) for idx,v in enumerate(self.f[:-1])]
        self.h = [v*(self.deg-idx-1) for idx,v in enumerate(self.g[:-1])]
        self.P_rlms = 1.25*self.P_rlms - 1.25* np.matmul(np.matmul(k, u.T),self.P_rlms)

    def solve_ekf(self):
        self.f = self.f
        self.g = [v*(self.deg-idx) for idx,v in enumerate(self.f[:-1])]
        self.h = [v*(self.deg-idx-1) for idx,v in enumerate(self.g[:-1])]
        return
    
    def track(self, p, v):
        p = p/self.Vbase/self.Ibase
        flag = False
        v_est = v/self.Vbase
        count = 0
        err = p + poly.polyval(v_est, self.f)
        while abs(err)>self.eps and count<40:
            count += 1
            if not flag:
                if err<0:
                    flag = True
                    count = 0
                else:
                    g = poly.polyval(v_est, self.g)
                    h = poly.polyval(v_est, self.h)
                    v_est -= np.clip(g/h, -1.0,1.0)
            else:
                g = poly.polyval(v_est, self.g)
                h = poly.polyval(v_est, self.h)
                if g>0:
                    v_est -= np.clip(err/g, -1.0,1.0)
                else:
                    v_est -= np.clip(g/h, -1.0,1.0) +0.0001
            err = p + poly.polyval(v_est, self.f)
        return -poly.polyval(v_est, self.f)*self.Vbase*self.Ibase, v_est*self.Vbase
    
    def find_opt(self, v_est=100):
        count = 0
        err = poly.polyval(v_est, self.g)
        while abs(err)>self.eps and count<40:
            count += 1
            h = poly.polyval(v_est, self.h)
            v_est -= np.clip(err/h, -1.0,1.0)
            err = poly.polyval(v_est, self.g)
        return -poly.polyval(v_est, self.f)*self.Vbase, v_est*self.Vbase
    
    def vectorize1d(self, deg, v):
        v = np.asarray(v)
        p = np.array(range(deg,-1,-1))
        p = np.tile(p, (len(v),1)).T
        return np.power(v,p).T

    def vectorize2d(self, deg):
        A1 = self.vectorize1d(deg-1, self.buffer_V/self.Vbase)
        l1 = int(deg*(deg+1)/2)
        l2 = int(len(self.buffer_V/self.Vbase))
        A = np.zeros((l2,l1))
        k = 0
        for i in range(1,deg+1):
            for j in range(1,i+1):
                A[:,k] = np.multiply(A1[:,i-1],A1[:,j-1])
                k+=1
        return A

class pv_panel():
    def __init__(self, Np=10, Ns=10):
        K=1.38065e-23
        q=1.602e-19
        self.Iscn=8.21
        Vocn=32.9
        Kv=-0.123
        self.Ki=0.0032
        Nc=72
        self.Tn=30+273
        self.Gn=1000
        self.a=2.0
        Eg=1.2
        self.Rs=0.221
        self.Rp=415.405
        Vtn=Nc*((K*self.Tn)/q)
        Irr=self.Iscn/((np.exp(Vocn/(self.a*Vtn)))-1)
        self.qegak = (q*Eg/(self.a*K))
        self.Idt=Irr*(self.Tn**3)*np.exp(self.qegak/self.Tn)# /((T**3)*np.exp(self.qegak/T))
        self.nckq = Nc*K/q
        self.Np = Np
        self.Ns = Ns
    
    def solve_v(self, G, T, I, V):
        Id = self.Idt/((T**3)*np.exp(self.qegak/T))
        Iph = (G/self.Gn)*(self.Iscn+self.Ki*(T-self.Tn))
        Vt = T*self.nckq
        V = V/self.Ns
        I = I/self.Np
        t1 = np.exp((V+I*self.Rs)/(Vt*self.a))
        t2 = (V + I*self.Rs)/self.Rp
        err = Iph-(Id*(t1-1)+t2)-I
        count = 0
        while abs(err)>1e-10 and count<100:
            dv = -Id*t1/(Vt*self.a)-1/self.Rp
            V = max(0, V-0.8*err/dv)
            t1 = np.exp((V+I*self.Rs)/(Vt*self.a))
            t2 = (V + I*self.Rs)/self.Rp
            err = Iph-(Id*(t1-1)+t2)-I
            count += 1
        dI=-Id*self.Rs*t1/(Vt*self.a)-self.Rs/self.Rp-1
        dV=-Id*t1/(Vt*self.a)-1/self.Rp
        d = -dI*self.Np/(dV*self.Ns)
        return V*self.Ns, d

    def solve_i(self, G, T, I, V):
        Id = self.Idt/((T**3)*np.exp(self.qegak/T))
        Iph = (G/self.Gn)*(self.Iscn+self.Ki*(T-self.Tn))
        Vt = T*self.nckq
        V = V/self.Ns
        I = I/self.Np
        t1 = np.exp((V+I*self.Rs)/(Vt*self.a))
        t2 = (V + I*self.Rs)/self.Rp
        err = Iph-(Id*(t1-1)+t2)-I
        count = 0
        while abs(err)>1e-10 and count<1000:
            dI=-Id*self.Rs*t1/(Vt*self.a)-self.Rs/self.Rp-1
            I = max(0, I-0.8*err/dI)
            t1 = np.exp((V+I*self.Rs)/(Vt*self.a))
            t2 = (V + I*self.Rs)/self.Rp
            err = Iph-(Id*(t1-1)+t2)-I
            count += 1

        dfdI=-Id*self.Rs*t1/(Vt*self.a)-self.Rs/self.Rp-1
        dfdV=-Id*t1/(Vt*self.a)-1/self.Rp
        dIdV = -dfdV*self.Ns/(dfdI*self.Np)
        return I*self.Np, dIdV
    
    def find_opt(self, G, T):
        V = 100
        I = 10
        I, d = self.solve_i(G,T,I,V)
        err = V/I+1/d
        while abs(err)>1e-8:
            V = V - 0.5*err
            I, dIdV = self.solve_i(G,T,I,V)
            err = V/I+1/dIdV
            # print(V,I,dIdV,err)
        return I*V, V

    def find_vref(self, G, T, p):
        V = 100
        I = p/V
        I, d = self.solve_i(G,T,I,V)
        err = p - I*V
        count = 0
        while abs(err)>1e-8 and count<100:
            count += 1
            d = I + V*d
            if d > 0:
                V = V + 1.5*d
            else:
                count = 0
                V = V + np.clip(err/d,-5,5)
            I, d = self.solve_i(G,T,I,V)
            err = V/I+1/d
        return I*V, V
