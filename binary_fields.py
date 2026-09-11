class BinaryField:
    """
        polynomials are represented as binary lists
        with leading coefficient first
    """
    def __init__( self, module ) -> None:
        # TODO: add more checks
        assert module[0]==1, "leading coefficient must be 1"
        # poly is irreducible, this is not checked
        self.module = module
        self.deg = len(module)
        self.elemlen = self.deg-1
        self.size = 2**self.deg
    def __repr__( self ):
        return f"GF(2^{self.deg}) mod {''.join((str(x) for x in self.module ))}"
    def zero( self ):
        return BinaryFieldElement( self, (0,) )
    def one( self ):
        return BinaryFieldElement( self, (1,) )

class BinaryFieldElement:
    def __init__( self, bf: BinaryField, coeffs ) -> None:
        # TODO: add checks
        self.coeffs = _modulo( coeffs, bf.module )
        self.field = bf
        self.elemlen = bf.elemlen

    def __repr__( self ):
#        return f"{self.coeffs}_{field.module}"
        return ''.join((str(x) for x in self.coeffs ))

    def __eq__( self, other):
        return self.field == other.field and self.coeffs == other.coeffs

    def __add__( self, other ):
        assert self.field == other.field, "incompatible fields"
        return BinaryFieldElement( self.field, _add( self.coeffs, other.coeffs ) )

    def __sub__( self, other ):
        assert self.field == other.field, "incompatible fields"
        return BinaryFieldElement( self.field, _add( self.coeffs, other.coeffs ) )

    def __mul__( self, other ):
        assert self.field == other.field, "incompatible fields"
        return BinaryFieldElement( self.field, _mult( self.coeffs, other.coeffs ) )

    def __truediv__( self, other ):
        assert self.field == other.field, "incompatible fields"
        assert other.coeffs != (0,), "division by zero"
        return BinaryFieldElement( self.field, _mult( self.coeffs, _inverse_mod( other.coeffs, other.field.module ) ) )

    def __pow__(  self, exponent ):
        # generic (stupid) version for small positive exponents
        assert exponent >= 0, "negative exponents not implemented"
        power = BinaryFieldElement( self.field, (1,) )
        for _ in range( exponent ):
            power = power * self
        return power

def _equal_len( a, b ):
    if len(a)>len(b):
        b = (len(a)-len(b))*(0,) + b
    elif len(a)<len(b):
        a = (len(b)-len(a))*(0,) + a
    return a, b

def _add( a, b ):
    a, b = _equal_len( a, b )
    return _remove_leading_zeros( tuple( a[i]^b[i] for i in range(len(a))) )

def _mult( a, b ):
    a, b = _remove_leading_zeros( a ), _remove_leading_zeros( b )
    prod = (0,)
    for d in reversed(b):
        if d==1:
            prod = _add( prod, a )
        a = a + (0,)
    return prod

def _remove_leading_zeros( a ):
    if len(a)==1:
        return a
    elif a[0]==1:
        return a
    else:
        return _remove_leading_zeros( a[1:] )

def _modulo( a, m ):
    _ , r = _quotient_remainder( a, m )
    return r

def _quotient_remainder( a, m ):
    return _quotient_remainder_rec( (0,), a, m )

def _quotient_remainder_rec( q, r, m ):
    r = _remove_leading_zeros( r )
    m = _remove_leading_zeros( m )
    if r==(0,) or len(r) < len(m):
        return q, r
    shift = len(r)-len(m)
    sh_m =    m + shift*(0,) # shift m
    sh_q = (1,) + shift*(0,) # shift q
    r = _add( r, sh_m )
    q = _add( q, sh_q )
    return _quotient_remainder_rec( q, r, m )

def _extended_gcd( a, b ):
    if len(a)<len(b):
        ( g, x, y ) = _extended_gcd( b, a )
        return (g,y,x)
    x,y, u,v = (0,),(1,), (1,),(0,)
    while a!=(0,):
        q,r = _quotient_remainder( b, a )
        m,n = _add( x, _mult( u, q ) ), _add( y, _mult( v, q ) )
        b,a, x,y, u,v = a,r, u,v, m,n
        # if q==(0,):
        #     print( b,x,y )
        # else:
        #     print( b,x,y,q )
    return b,x,y

def _inverse_mod( a, m ):
    a = _modulo( a, m )
    g, _, y = _extended_gcd( m, a )
    assert g == (1,), "no inverse"
    return _modulo( y, m )

class BinaryFieldPolynomial:
    def __init__( self, coeffs ):
        self.deg = if coeffs
### Examples ###

a = (1,1,0,1,0,0,1,0,0,1)
b = (1,1,0,1,0,1,1,1,0,1)
m =           (1,0,1,0,1)

field = BinaryField( m )
print( field )

ba = BinaryFieldElement( field, a )
bb = BinaryFieldElement( field, b )

print( f"{ba} + {bb} = {ba+bb}")
print( f"{ba} - {bb} = {ba-bb}")
print( f"{ba} * {bb} = {ba*bb}")
print( f"{ba} / {bb} = {ba/bb}")
print( f"{ba**3}")
