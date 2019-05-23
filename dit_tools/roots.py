def bisection(func,x1,x2,tol):
  
    JMAX = 100
    fmid=func(x2)
    f=func(x1)
    
    if f*fmid >= 0.: 
        print('root must be bracketed')
        
    if f < 0.:
        root = x1
        dx = x2-x1
    else:
        root = x2
        dx = x1-x2
        
    for j in range(1,JMAX+1):
        dx = dx*.5
        xmid = root + dx
        fmid = func(xmid)
        
        if fmid <= 0.:
            root = xmid
        # Convergence is tested using relative error between
        # consecutive approximations.
        if abs(dx) < abs(root)*tol or fmid == 0.: 
            return root
        
    print('too many bisections')
  
def newton(f, df, x, tol):
    
    MAXIT = 20;
    root = x;
    
    for j in range(1,MAXIT+1):
        dx = f(root) / df(root);
        root = root - dx;
        # Convergence is tested using relative error between
        # consecutive approximations.
        if (abs(dx) <= abs(root)*tol):
            return root
        
    print('Max number of iterations reached.')
