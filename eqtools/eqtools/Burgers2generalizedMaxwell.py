import numpy as np

def Burger2GMB(miuM, etaM, miuK, etaK):
    '''
    Burgers Body:
        Maxwell: miuM, etaM
        Kelvin : miuK, etaK
    Generalized Maxwell Body with two Maxwell elements:
        m1, m2, eta1, eta2

    Ratio of the long term to the initial shear modulus:
        alpha = miuK/(miuM + miuK)
    
    Reference   :
        * Muller, 1986
    
    Added by kfhe at 05/31/2023
    '''

    a = miuK/miuM
    b = etaK/etaM
    
    X = 1./(2.*a)*(1 + a + b + np.sqrt((1 + a + b)**2 - 4*a*b))
    
    x1 = (a*X - b)/(a*X**2 - b)
    y1 = x1*X
    
    x2 = 1 - x1
    y2 = 1 - y1 
    # get absolute modulus
    m1 = x1*miuM 
    m2 = x2*miuM
    
    eta1 = y1*etaM 
    eta2 = y2*etaM
    # get fractional modulus
    mtot = m1 + m2
    nm1 = m1/mtot
    nm2 = m2/mtot
    
    return m1, eta1, m2, eta2, nm1, nm2


def Burgers2generalizedMaxwellInSympy():
    '''
    
    '''
    import sympy
    import numpy as np
    
    sympy.init_printing()
    
    # eta1k = sympy.Symbol("eta1k");
    # eta2k = sympy.Symbol('eta2k');
    # e1k = sympy.Symbol('e1k');
    # e2k = sympy.Symbol('e2k');
    # eta1m = sympy.Symbol('eta1m');
    # eta2m = sympy.Symbol('eta2m');
    # e1m = sympy.Symbol('e1m');
    # e2m = sympy.Symbol('e2m');
    
    # result = sympy.solve([eta2k-eta1m-eta2m, e2k-e1m-e2m, (eta1k*eta2k)/(e1k*e2k)-(eta1m*eta2m)/(e1m*e2m), eta1k/e1k+eta2k/e1k+eta2k/e2k-eta1m/e1m-eta2m/e2m], [eta1m, eta2m, e1m, e2m]);
    # 用于计算各变量的值
    # res = list(result)
    # res[0][0].evalf(subs={eta1m:3.410425763384412e+18, eta2m:8.957423661558213e+18, e1m:1.71011e+10, e2m:1.88989e+10})

    e1k = 3.0e10 # 3.6e10
    e2k = 3.0e10
    eta1k = 1.e17
    eta2k = 1.e18

    eta1m = [
    (e2k**2*eta1k*eta2k - e1k*e2k*eta2k**2 + (e1k*eta2k**2*(e2k**3*eta1k**2 + e2k**3*eta2k**2 + 2*e1k*e2k**2*eta2k**2 + e1k**2*e2k*eta2k**2 + e2k**2*eta1k*(e1k**2*eta2k**2 - 2*e1k*e2k*eta1k*eta2k + 2*e1k*e2k*eta2k**2 + e2k**2*eta1k**2 + 2*e2k**2*eta1k*eta2k + e2k**2*eta2k**2)**(1/2) + e2k**2*eta2k*(e1k**2*eta2k**2 - 2*e1k*e2k*eta1k*eta2k + 2*e1k*e2k*eta2k**2 + e2k**2*eta1k**2 + 2*e2k**2*eta1k*eta2k + e2k**2*eta2k**2)**(1/2) + 2*e2k**3*eta1k*eta2k - 2*e1k*e2k**2*eta1k*eta2k - e1k*e2k*eta2k*(e1k**2*eta2k**2 - 2*e1k*e2k*eta1k*eta2k + 2*e1k*e2k*eta2k**2 + e2k**2*eta1k**2 + 2*e2k**2*eta1k*eta2k + e2k**2*eta2k**2)**(1/2)))/(2*(e1k**2*eta2k**2 - 2*e1k*e2k*eta1k*eta2k + 2*e1k*e2k*eta2k**2 + e2k**2*eta1k**2 + 2*e2k**2*eta1k*eta2k + e2k**2*eta2k**2)) + (e2k*eta2k**2*(e2k**3*eta1k**2 + e2k**3*eta2k**2 + 2*e1k*e2k**2*eta2k**2 + e1k**2*e2k*eta2k**2 + e2k**2*eta1k*(e1k**2*eta2k**2 - 2*e1k*e2k*eta1k*eta2k + 2*e1k*e2k*eta2k**2 + e2k**2*eta1k**2 + 2*e2k**2*eta1k*eta2k + e2k**2*eta2k**2)**(1/2) + e2k**2*eta2k*(e1k**2*eta2k**2 - 2*e1k*e2k*eta1k*eta2k + 2*e1k*e2k*eta2k**2 + e2k**2*eta1k**2 + 2*e2k**2*eta1k*eta2k + e2k**2*eta2k**2)**(1/2) + 2*e2k**3*eta1k*eta2k - 2*e1k*e2k**2*eta1k*eta2k - e1k*e2k*eta2k*(e1k**2*eta2k**2 - 2*e1k*e2k*eta1k*eta2k + 2*e1k*e2k*eta2k**2 + e2k**2*eta1k**2 + 2*e2k**2*eta1k*eta2k + e2k**2*eta2k**2)**(1/2)))/(2*(e1k**2*eta2k**2 - 2*e1k*e2k*eta1k*eta2k + 2*e1k*e2k*eta2k**2 + e2k**2*eta1k**2 + 2*e2k**2*eta1k*eta2k + e2k**2*eta2k**2)) - (e2k*eta1k*eta2k*(e2k**3*eta1k**2 + e2k**3*eta2k**2 + 2*e1k*e2k**2*eta2k**2 + e1k**2*e2k*eta2k**2 + e2k**2*eta1k*(e1k**2*eta2k**2 - 2*e1k*e2k*eta1k*eta2k + 2*e1k*e2k*eta2k**2 + e2k**2*eta1k**2 + 2*e2k**2*eta1k*eta2k + e2k**2*eta2k**2)**(1/2) + e2k**2*eta2k*(e1k**2*eta2k**2 - 2*e1k*e2k*eta1k*eta2k + 2*e1k*e2k*eta2k**2 + e2k**2*eta1k**2 + 2*e2k**2*eta1k*eta2k + e2k**2*eta2k**2)**(1/2) + 2*e2k**3*eta1k*eta2k - 2*e1k*e2k**2*eta1k*eta2k - e1k*e2k*eta2k*(e1k**2*eta2k**2 - 2*e1k*e2k*eta1k*eta2k + 2*e1k*e2k*eta2k**2 + e2k**2*eta1k**2 + 2*e2k**2*eta1k*eta2k + e2k**2*eta2k**2)**(1/2)))/(2*(e1k**2*eta2k**2 - 2*e1k*e2k*eta1k*eta2k + 2*e1k*e2k*eta2k**2 + e2k**2*eta1k**2 + 2*e2k**2*eta1k*eta2k + e2k**2*eta2k**2)))/(e2k**2*eta1k + e2k**2*eta2k - e1k*e2k*eta2k)
    , (e2k**2*eta1k*eta2k - e1k*e2k*eta2k**2 + (e1k*eta2k**2*(e2k**3*eta1k**2 + e2k**3*eta2k**2 + 2*e1k*e2k**2*eta2k**2 + e1k**2*e2k*eta2k**2 - e2k**2*eta1k*(e1k**2*eta2k**2 - 2*e1k*e2k*eta1k*eta2k + 2*e1k*e2k*eta2k**2 + e2k**2*eta1k**2 + 2*e2k**2*eta1k*eta2k + e2k**2*eta2k**2)**(1/2) - e2k**2*eta2k*(e1k**2*eta2k**2 - 2*e1k*e2k*eta1k*eta2k + 2*e1k*e2k*eta2k**2 + e2k**2*eta1k**2 + 2*e2k**2*eta1k*eta2k + e2k**2*eta2k**2)**(1/2) + 2*e2k**3*eta1k*eta2k - 2*e1k*e2k**2*eta1k*eta2k + e1k*e2k*eta2k*(e1k**2*eta2k**2 - 2*e1k*e2k*eta1k*eta2k + 2*e1k*e2k*eta2k**2 + e2k**2*eta1k**2 + 2*e2k**2*eta1k*eta2k + e2k**2*eta2k**2)**(1/2)))/(2*(e1k**2*eta2k**2 - 2*e1k*e2k*eta1k*eta2k + 2*e1k*e2k*eta2k**2 + e2k**2*eta1k**2 + 2*e2k**2*eta1k*eta2k + e2k**2*eta2k**2)) + (e2k*eta2k**2*(e2k**3*eta1k**2 + e2k**3*eta2k**2 + 2*e1k*e2k**2*eta2k**2 + e1k**2*e2k*eta2k**2 - e2k**2*eta1k*(e1k**2*eta2k**2 - 2*e1k*e2k*eta1k*eta2k + 2*e1k*e2k*eta2k**2 + e2k**2*eta1k**2 + 2*e2k**2*eta1k*eta2k + e2k**2*eta2k**2)**(1/2) - e2k**2*eta2k*(e1k**2*eta2k**2 - 2*e1k*e2k*eta1k*eta2k + 2*e1k*e2k*eta2k**2 + e2k**2*eta1k**2 + 2*e2k**2*eta1k*eta2k + e2k**2*eta2k**2)**(1/2) + 2*e2k**3*eta1k*eta2k - 2*e1k*e2k**2*eta1k*eta2k + e1k*e2k*eta2k*(e1k**2*eta2k**2 - 2*e1k*e2k*eta1k*eta2k + 2*e1k*e2k*eta2k**2 + e2k**2*eta1k**2 + 2*e2k**2*eta1k*eta2k + e2k**2*eta2k**2)**(1/2)))/(2*(e1k**2*eta2k**2 - 2*e1k*e2k*eta1k*eta2k + 2*e1k*e2k*eta2k**2 + e2k**2*eta1k**2 + 2*e2k**2*eta1k*eta2k + e2k**2*eta2k**2)) - (e2k*eta1k*eta2k*(e2k**3*eta1k**2 + e2k**3*eta2k**2 + 2*e1k*e2k**2*eta2k**2 + e1k**2*e2k*eta2k**2 - e2k**2*eta1k*(e1k**2*eta2k**2 - 2*e1k*e2k*eta1k*eta2k + 2*e1k*e2k*eta2k**2 + e2k**2*eta1k**2 + 2*e2k**2*eta1k*eta2k + e2k**2*eta2k**2)**(1/2) - e2k**2*eta2k*(e1k**2*eta2k**2 - 2*e1k*e2k*eta1k*eta2k + 2*e1k*e2k*eta2k**2 + e2k**2*eta1k**2 + 2*e2k**2*eta1k*eta2k + e2k**2*eta2k**2)**(1/2) + 2*e2k**3*eta1k*eta2k - 2*e1k*e2k**2*eta1k*eta2k + e1k*e2k*eta2k*(e1k**2*eta2k**2 - 2*e1k*e2k*eta1k*eta2k + 2*e1k*e2k*eta2k**2 + e2k**2*eta1k**2 + 2*e2k**2*eta1k*eta2k + e2k**2*eta2k**2)**(1/2)))/(2*(e1k**2*eta2k**2 - 2*e1k*e2k*eta1k*eta2k + 2*e1k*e2k*eta2k**2 + e2k**2*eta1k**2 + 2*e2k**2*eta1k*eta2k + e2k**2*eta2k**2)))/(e2k**2*eta1k + e2k**2*eta2k - e1k*e2k*eta2k)
    ]
    
    eta2m = [
    
    (e2k**2*eta2k**2 - (e1k*eta2k**2*(e2k**3*eta1k**2 + e2k**3*eta2k**2 + 2*e1k*e2k**2*eta2k**2 + e1k**2*e2k*eta2k**2 + e2k**2*eta1k*(e1k**2*eta2k**2 - 2*e1k*e2k*eta1k*eta2k + 2*e1k*e2k*eta2k**2 + e2k**2*eta1k**2 + 2*e2k**2*eta1k*eta2k + e2k**2*eta2k**2)**(1/2) + e2k**2*eta2k*(e1k**2*eta2k**2 - 2*e1k*e2k*eta1k*eta2k + 2*e1k*e2k*eta2k**2 + e2k**2*eta1k**2 + 2*e2k**2*eta1k*eta2k + e2k**2*eta2k**2)**(1/2) + 2*e2k**3*eta1k*eta2k - 2*e1k*e2k**2*eta1k*eta2k - e1k*e2k*eta2k*(e1k**2*eta2k**2 - 2*e1k*e2k*eta1k*eta2k + 2*e1k*e2k*eta2k**2 + e2k**2*eta1k**2 + 2*e2k**2*eta1k*eta2k + e2k**2*eta2k**2)**(1/2)))/(2*(e1k**2*eta2k**2 - 2*e1k*e2k*eta1k*eta2k + 2*e1k*e2k*eta2k**2 + e2k**2*eta1k**2 + 2*e2k**2*eta1k*eta2k + e2k**2*eta2k**2)) - (e2k*eta2k**2*(e2k**3*eta1k**2 + e2k**3*eta2k**2 + 2*e1k*e2k**2*eta2k**2 + e1k**2*e2k*eta2k**2 + e2k**2*eta1k*(e1k**2*eta2k**2 - 2*e1k*e2k*eta1k*eta2k + 2*e1k*e2k*eta2k**2 + e2k**2*eta1k**2 + 2*e2k**2*eta1k*eta2k + e2k**2*eta2k**2)**(1/2) + e2k**2*eta2k*(e1k**2*eta2k**2 - 2*e1k*e2k*eta1k*eta2k + 2*e1k*e2k*eta2k**2 + e2k**2*eta1k**2 + 2*e2k**2*eta1k*eta2k + e2k**2*eta2k**2)**(1/2) + 2*e2k**3*eta1k*eta2k - 2*e1k*e2k**2*eta1k*eta2k - e1k*e2k*eta2k*(e1k**2*eta2k**2 - 2*e1k*e2k*eta1k*eta2k + 2*e1k*e2k*eta2k**2 + e2k**2*eta1k**2 + 2*e2k**2*eta1k*eta2k + e2k**2*eta2k**2)**(1/2)))/(2*(e1k**2*eta2k**2 - 2*e1k*e2k*eta1k*eta2k + 2*e1k*e2k*eta2k**2 + e2k**2*eta1k**2 + 2*e2k**2*eta1k*eta2k + e2k**2*eta2k**2)) + (e2k*eta1k*eta2k*(e2k**3*eta1k**2 + e2k**3*eta2k**2 + 2*e1k*e2k**2*eta2k**2 + e1k**2*e2k*eta2k**2 + e2k**2*eta1k*(e1k**2*eta2k**2 - 2*e1k*e2k*eta1k*eta2k + 2*e1k*e2k*eta2k**2 + e2k**2*eta1k**2 + 2*e2k**2*eta1k*eta2k + e2k**2*eta2k**2)**(1/2) + e2k**2*eta2k*(e1k**2*eta2k**2 - 2*e1k*e2k*eta1k*eta2k + 2*e1k*e2k*eta2k**2 + e2k**2*eta1k**2 + 2*e2k**2*eta1k*eta2k + e2k**2*eta2k**2)**(1/2) + 2*e2k**3*eta1k*eta2k - 2*e1k*e2k**2*eta1k*eta2k - e1k*e2k*eta2k*(e1k**2*eta2k**2 - 2*e1k*e2k*eta1k*eta2k + 2*e1k*e2k*eta2k**2 + e2k**2*eta1k**2 + 2*e2k**2*eta1k*eta2k + e2k**2*eta2k**2)**(1/2)))/(2*(e1k**2*eta2k**2 - 2*e1k*e2k*eta1k*eta2k + 2*e1k*e2k*eta2k**2 + e2k**2*eta1k**2 + 2*e2k**2*eta1k*eta2k + e2k**2*eta2k**2)))/(e2k**2*eta1k + e2k**2*eta2k - e1k*e2k*eta2k)
    , (e2k**2*eta2k**2 - (e1k*eta2k**2*(e2k**3*eta1k**2 + e2k**3*eta2k**2 + 2*e1k*e2k**2*eta2k**2 + e1k**2*e2k*eta2k**2 - e2k**2*eta1k*(e1k**2*eta2k**2 - 2*e1k*e2k*eta1k*eta2k + 2*e1k*e2k*eta2k**2 + e2k**2*eta1k**2 + 2*e2k**2*eta1k*eta2k + e2k**2*eta2k**2)**(1/2) - e2k**2*eta2k*(e1k**2*eta2k**2 - 2*e1k*e2k*eta1k*eta2k + 2*e1k*e2k*eta2k**2 + e2k**2*eta1k**2 + 2*e2k**2*eta1k*eta2k + e2k**2*eta2k**2)**(1/2) + 2*e2k**3*eta1k*eta2k - 2*e1k*e2k**2*eta1k*eta2k + e1k*e2k*eta2k*(e1k**2*eta2k**2 - 2*e1k*e2k*eta1k*eta2k + 2*e1k*e2k*eta2k**2 + e2k**2*eta1k**2 + 2*e2k**2*eta1k*eta2k + e2k**2*eta2k**2)**(1/2)))/(2*(e1k**2*eta2k**2 - 2*e1k*e2k*eta1k*eta2k + 2*e1k*e2k*eta2k**2 + e2k**2*eta1k**2 + 2*e2k**2*eta1k*eta2k + e2k**2*eta2k**2)) - (e2k*eta2k**2*(e2k**3*eta1k**2 + e2k**3*eta2k**2 + 2*e1k*e2k**2*eta2k**2 + e1k**2*e2k*eta2k**2 - e2k**2*eta1k*(e1k**2*eta2k**2 - 2*e1k*e2k*eta1k*eta2k + 2*e1k*e2k*eta2k**2 + e2k**2*eta1k**2 + 2*e2k**2*eta1k*eta2k + e2k**2*eta2k**2)**(1/2) - e2k**2*eta2k*(e1k**2*eta2k**2 - 2*e1k*e2k*eta1k*eta2k + 2*e1k*e2k*eta2k**2 + e2k**2*eta1k**2 + 2*e2k**2*eta1k*eta2k + e2k**2*eta2k**2)**(1/2) + 2*e2k**3*eta1k*eta2k - 2*e1k*e2k**2*eta1k*eta2k + e1k*e2k*eta2k*(e1k**2*eta2k**2 - 2*e1k*e2k*eta1k*eta2k + 2*e1k*e2k*eta2k**2 + e2k**2*eta1k**2 + 2*e2k**2*eta1k*eta2k + e2k**2*eta2k**2)**(1/2)))/(2*(e1k**2*eta2k**2 - 2*e1k*e2k*eta1k*eta2k + 2*e1k*e2k*eta2k**2 + e2k**2*eta1k**2 + 2*e2k**2*eta1k*eta2k + e2k**2*eta2k**2)) + (e2k*eta1k*eta2k*(e2k**3*eta1k**2 + e2k**3*eta2k**2 + 2*e1k*e2k**2*eta2k**2 + e1k**2*e2k*eta2k**2 - e2k**2*eta1k*(e1k**2*eta2k**2 - 2*e1k*e2k*eta1k*eta2k + 2*e1k*e2k*eta2k**2 + e2k**2*eta1k**2 + 2*e2k**2*eta1k*eta2k + e2k**2*eta2k**2)**(1/2) - e2k**2*eta2k*(e1k**2*eta2k**2 - 2*e1k*e2k*eta1k*eta2k + 2*e1k*e2k*eta2k**2 + e2k**2*eta1k**2 + 2*e2k**2*eta1k*eta2k + e2k**2*eta2k**2)**(1/2) + 2*e2k**3*eta1k*eta2k - 2*e1k*e2k**2*eta1k*eta2k + e1k*e2k*eta2k*(e1k**2*eta2k**2 - 2*e1k*e2k*eta1k*eta2k + 2*e1k*e2k*eta2k**2 + e2k**2*eta1k**2 + 2*e2k**2*eta1k*eta2k + e2k**2*eta2k**2)**(1/2)))/(2*(e1k**2*eta2k**2 - 2*e1k*e2k*eta1k*eta2k + 2*e1k*e2k*eta2k**2 + e2k**2*eta1k**2 + 2*e2k**2*eta1k*eta2k + e2k**2*eta2k**2)))/(e2k**2*eta1k + e2k**2*eta2k - e1k*e2k*eta2k)
    ]
    
    e2m = [
    
    (e2k**3*eta1k**2 + e2k**3*eta2k**2 + 2*e1k*e2k**2*eta2k**2 + e1k**2*e2k*eta2k**2 + e2k**2*eta1k*(e1k**2*eta2k**2 - 2*e1k*e2k*eta1k*eta2k + 2*e1k*e2k*eta2k**2 + e2k**2*eta1k**2 + 2*e2k**2*eta1k*eta2k + e2k**2*eta2k**2)**(1/2) + e2k**2*eta2k*(e1k**2*eta2k**2 - 2*e1k*e2k*eta1k*eta2k + 2*e1k*e2k*eta2k**2 + e2k**2*eta1k**2 + 2*e2k**2*eta1k*eta2k + e2k**2*eta2k**2)**(1/2) + 2*e2k**3*eta1k*eta2k - 2*e1k*e2k**2*eta1k*eta2k - e1k*e2k*eta2k*(e1k**2*eta2k**2 - 2*e1k*e2k*eta1k*eta2k + 2*e1k*e2k*eta2k**2 + e2k**2*eta1k**2 + 2*e2k**2*eta1k*eta2k + e2k**2*eta2k**2)**(1/2))/(2*(e1k**2*eta2k**2 - 2*e1k*e2k*eta1k*eta2k + 2*e1k*e2k*eta2k**2 + e2k**2*eta1k**2 + 2*e2k**2*eta1k*eta2k + e2k**2*eta2k**2))
    , (e2k**3*eta1k**2 + e2k**3*eta2k**2 + 2*e1k*e2k**2*eta2k**2 + e1k**2*e2k*eta2k**2 - e2k**2*eta1k*(e1k**2*eta2k**2 - 2*e1k*e2k*eta1k*eta2k + 2*e1k*e2k*eta2k**2 + e2k**2*eta1k**2 + 2*e2k**2*eta1k*eta2k + e2k**2*eta2k**2)**(1/2) - e2k**2*eta2k*(e1k**2*eta2k**2 - 2*e1k*e2k*eta1k*eta2k + 2*e1k*e2k*eta2k**2 + e2k**2*eta1k**2 + 2*e2k**2*eta1k*eta2k + e2k**2*eta2k**2)**(1/2) + 2*e2k**3*eta1k*eta2k - 2*e1k*e2k**2*eta1k*eta2k + e1k*e2k*eta2k*(e1k**2*eta2k**2 - 2*e1k*e2k*eta1k*eta2k + 2*e1k*e2k*eta2k**2 + e2k**2*eta1k**2 + 2*e2k**2*eta1k*eta2k + e2k**2*eta2k**2)**(1/2))/(2*(e1k**2*eta2k**2 - 2*e1k*e2k*eta1k*eta2k + 2*e1k*e2k*eta2k**2 + e2k**2*eta1k**2 + 2*e2k**2*eta1k*eta2k + e2k**2*eta2k**2))
    ]
    
    e1m = [
    
    e2k - (e2k**3*eta1k**2 + e2k**3*eta2k**2 + 2*e1k*e2k**2*eta2k**2 + e1k**2*e2k*eta2k**2 + e2k**2*eta1k*(e1k**2*eta2k**2 - 2*e1k*e2k*eta1k*eta2k + 2*e1k*e2k*eta2k**2 + e2k**2*eta1k**2 + 2*e2k**2*eta1k*eta2k + e2k**2*eta2k**2)**(1/2) + e2k**2*eta2k*(e1k**2*eta2k**2 - 2*e1k*e2k*eta1k*eta2k + 2*e1k*e2k*eta2k**2 + e2k**2*eta1k**2 + 2*e2k**2*eta1k*eta2k + e2k**2*eta2k**2)**(1/2) + 2*e2k**3*eta1k*eta2k - 2*e1k*e2k**2*eta1k*eta2k - e1k*e2k*eta2k*(e1k**2*eta2k**2 - 2*e1k*e2k*eta1k*eta2k + 2*e1k*e2k*eta2k**2 + e2k**2*eta1k**2 + 2*e2k**2*eta1k*eta2k + e2k**2*eta2k**2)**(1/2))/(2*(e1k**2*eta2k**2 - 2*e1k*e2k*eta1k*eta2k + 2*e1k*e2k*eta2k**2 + e2k**2*eta1k**2 + 2*e2k**2*eta1k*eta2k + e2k**2*eta2k**2))
    , e2k - (e2k**3*eta1k**2 + e2k**3*eta2k**2 + 2*e1k*e2k**2*eta2k**2 + e1k**2*e2k*eta2k**2 - e2k**2*eta1k*(e1k**2*eta2k**2 - 2*e1k*e2k*eta1k*eta2k + 2*e1k*e2k*eta2k**2 + e2k**2*eta1k**2 + 2*e2k**2*eta1k*eta2k + e2k**2*eta2k**2)**(1/2) - e2k**2*eta2k*(e1k**2*eta2k**2 - 2*e1k*e2k*eta1k*eta2k + 2*e1k*e2k*eta2k**2 + e2k**2*eta1k**2 + 2*e2k**2*eta1k*eta2k + e2k**2*eta2k**2)**(1/2) + 2*e2k**3*eta1k*eta2k - 2*e1k*e2k**2*eta1k*eta2k + e1k*e2k*eta2k*(e1k**2*eta2k**2 - 2*e1k*e2k*eta1k*eta2k + 2*e1k*e2k*eta2k**2 + e2k**2*eta1k**2 + 2*e2k**2*eta1k*eta2k + e2k**2*eta2k**2)**(1/2))/(2*(e1k**2*eta2k**2 - 2*e1k*e2k*eta1k*eta2k + 2*e1k*e2k*eta2k**2 + e2k**2*eta1k**2 + 2*e2k**2*eta1k*eta2k + e2k**2*eta2k**2))
    ]

    #e1m = [17101122895.009941, 18898877104.99006] # 0.375 0.525 ratio; 总的等于e2k，对应于maxwell中弹性模量
    print('{0[0]:.10e} {0[1]:.10e}'.format(eta1m))
    print('{0[0]:.5e} {0[1]:.5e}'.format(e1m))
    # 1.71011e+10 1.88989e+10

    print('{0[0]:.5f} {0[1]:.5f}'.format(e1m/np.sum(e1m)))

    # 计算等效松弛粘性，发现瞬态粘性的松弛时间的10倍左右就差不多粘性转化为稳态粘性
    # eta_2yr = etam*etak/(etam*np.e**(-3./tauk) + etak)
    
    # All Done
    return 


if __name__ == '__main__':
    miuM = 3.e10
    miuK = 3.e10
    etaK = 3.e17
    etaM = 3.e18

    _, eta1, _, eta2, nm1, nm2 = Burger2GMB(miuM, etaM, miuK, etaK)
    print('{0:.8e} {1:.8e} {2:.8e} {3:.8e}'.format(nm1, eta1, nm2, eta2), file=None)