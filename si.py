"""
Useful functions for DML1, GDK2, and FTK3
Author: Jürgen Fuß
Date: 2025-07-10

Functions have doctest strings.
Use 'python -m doctest si.py' to test after code changes
or simply run this file (use -v for verbose test output)

See docstrings for functions and classes.

Dependencies:
Make sure to
- pip install sympy
- pip install numpy
- pip install alive-progress
or
- pip install -r requirements.txt
"""

import math, random
import sympy
from numpy import array, copy, zeros, eye, concatenate, ndarray, asarray
from alive_progress import alive_bar
# typing
from typing import List, Tuple, Iterator, Sequence

def _printv( v: int, threshold: int, string: str, end="\n" ) -> None:
    """
    Print depending on verbose level.

        Parameters:
            v (int): an integer
            threshold (int): an integer
            string (string): a string
            end="\n" (string): ending string for print() function

        Returns:
            Prints string, if v>threshold.
            Indentation decreases with verbose level v
    """
    if v>threshold:
        print( "'"+3*(4-v)*' ' + string, end=end )

### primes and factorisation

def list_primes( a: int, b: int ) -> List[int]:
    """
    Returns a list of all primes in an interval.

        Parameters:
            a (int): an integer
            b (int): an integer

        Returns:
            list_primes( a, b ): The list of all primes p, st. a <= p < b
    Example:
    >>> list_primes(7,19)
    [7, 11, 13, 17]
    """
    return list( sympy.sieve.primerange( a, b ) )

def prime_factors( n: int, unique: bool = False ) -> List[int]:
    """
    Returns all prime factors of n (with multiplicity).

        Parameters:
            n (int): a positive integer

        Returns:
            prime_factors( n ): A list of prime numbers whose product is n.
    Examples:
    >>> prime_factors( 57480192000 )
    [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 5, 5, 5, 7, 11]
    >>> prime_factors( 57480192000, unique=True )
    [2, 3, 5, 7, 11]
    """
    if unique:
        return list( sympy.ntheory.factorint(n).keys() )
    else:
        return sympy.ntheory.factorint( n, multiple=True )

def euler_phi( n: int ) -> int:
    """
    Returns the value of the Euler totient funtion at n.

        Parameters:
            n (int): a positive integer

        Returns:
            euler_phi( n ): the value of the Euler totient funtion at n.
    Example:
    >>> euler_phi( 3*25*343 )
    11760
    """
    factors = sympy.ntheory.factorint( n )
    return math.prod( (p-1) * p**(e-1) for (p,e) in factors.items() )

def extended_gcd( a: int, b: int, verbose: int = 0 ) -> Tuple[int,int,int]:
    """
    Computes the extended GCD, i.e. gcd, x, y, st. gcd = ax + by.

        Parameters:
            a (int): an integer
            b (int): an integer
            verbose (int): a non-negative integer

        Returns:
            extended_gcd( a, b, verbose ): A triple (d,a,b), st. d=gcd(a,b) and d=ax+by.
                If verbose>0, the extended Euclidean algorithm is shown step-by-step.
    Examples:
    >>> extended_gcd(1035,336)
    (3, 25, -77)
    >>> extended_gcd(1035,336, verbose=1 )
    '                     1035      336
    '            1035        1        0
    '             336        0        1      3
    '              27        1       -3     12
    '              12      -12       37      2
    '               3       25      -77      4
    (3, 25, -77)
    """
    if a<b:
        (g,x,y) = extended_gcd( b, a, verbose=verbose )
        return (g,y,x)
    # compute tab width for verbose output
    width = 3 + max( len(str(a)), len(str(b)) )
    width_plus = 2*width - 5

    x,y, u,v = 0,1, 1,0
    _printv( verbose, 0, f"{a:{width+width_plus}d}{b:{width_plus}d}" )
    while a != 0:
        q, r = b//a, b%a
        m, n = x-u*q, y-v*q
        b,a, x,y, u,v = a,r, u,v, m,n
        if q==0:
            _printv( verbose, 0, f"{b:{width}d}{x:{width_plus}d}{y:{width_plus}d}" )
        else:
            _printv( verbose, 0, f"{b:{width}d}{x:{width_plus}d}{y:{width_plus}d}{q:{width}d}" )
    return b, x, y

def inverse_mod( a: int, m: int, verbose: int = 0 ) -> int:
    """
    Computes the modular inverse of a mod m, or None, if it does not exist.

        Parameters:
            a (int): an integer
            m (int): a positive integer
            verbose (flag): a non-negative integer

        Returns:
            inverse_mod( a, m ): An integer b, st. ab=1 mod m.
    (pow(a,-1,m) does the same, but does not have a verbose option)
    Example:
    >>> inverse_mod(17,33,verbose=1)
    '                 33   17
    '            33    1    0
    '            17    0    1    1
    '            16    1   -1    1
    '             1   -1    2   16
    2
    """
    a = a % m
    gcd, x, y = extended_gcd( m, a, verbose )
    if gcd != 1:
        raise ValueError(f"{a} does not have an inverse mod {m}.")
    return y % m

def continued_fraction( a: int, b: int ) -> Iterator[int]:
    """
    Computes a continued fraction for a/b.

        Parameters:
            a (int): a positive integer
            b (int): a positive integer

        Returns:
            continued_fraction( a, b ): A generator for the list which represents the continued fraction of a/b.
    Example:
    >>> list( continued_fraction( 123456789, 654321 ) )
    [188, 1, 2, 8, 1, 1, 66, 1, 14, 4]
    """
    while b!=0:
        yield a//b
        a, b = b, a%b

def cf_approx_from_cf( cf: Iterator[int] ) -> Iterator[Tuple[int,int]]:
    """
    Returns approximations for a continued fraction a
       approximations are pairs representing fractions

        Parameters:
            a ([int]): a generator or list representing a continued fraction

        Returns:
            cf_approx( a ): a generator for the list of pairs (x,y) st. the i-th x/y is the i-th approximation of a.
    Example:
    >>> list( cf_approx_from_cf( [188, 1, 2, 8, 1, 1, 66, 1, 14, 4] ) )
    [(188, 1), (189, 1), (566, 3), (4717, 25), (5283, 28), (10000, 53), (665283, 3526), (675283, 3579), (10119245, 53632), (41152263, 218107)]
    >>> list( cf_approx_from_cf( continued_fraction( 166,13 ) ) )
    [(12, 1), (13, 1), (51, 4), (166, 13)]
    """
    if type(cf)==list:
        cf = (x for x in cf) # make the list a generator
    a0 = next(cf)
    p2, q2 = a0, 1
    yield ( p2, q2 )

    a1 = next(cf)
    p1, q1 = a0*a1+1, a1
    yield ( p1, q1 )

    for a in cf:
        p2, q2, p1, q1 = p1, q1, a*p1+p2, a*q1+q2
        yield ( p1, q1 )

def cf_approx( a: int, b: int ) -> Iterator[Tuple[int,int]]:
    """
    Returns approximations for a continued fraction a
       approximations are pairs representing fractions

        Parameters:
            a ([int]): a generator or list representing a continued fraction

        Returns:
            cf_approx( a ): a generator for the list of pairs (x,y) st. the i-th x/y is the i-th approximation if a.
    Example:
    >>> list( cf_approx( 166, 13 ) )
    [(12, 1), (13, 1), (51, 4), (166, 13)]
    """
    cf = continued_fraction( a, b )
    a0 = next(cf)
    p2, q2 = a0, 1
    yield ( p2, q2 )

    a1 = next(cf)
    p1, q1 = a0*a1+1, a1
    yield ( p1, q1 )

    for a in cf:
        p2, q2, p1, q1 = p1, q1, a*p1+p2, a*q1+q2
        yield ( p1, q1 )

def chinese_remainder( ns: List[int], xs: List[int] ) -> int:
    """
    Compute the result of the Chinese Remainder Theorem
        for a list of moduli ns and a list of remainders xs

        Parameters:
            ns ([int]): a list of positive integers
            xs ([int]): a list of integers

        Returns:
            chinese_remainder( ns, xs ): An integer x, st. x = xs[i] mod ns[i] for all i.
                A ZeroDivisionError is thrown, if the moduli are not pairswise relative prime.
    >>> chinese_remainder( [3,4,5], [1,2,3] )
    58
    """
    prod = 1
    for ni in ns:
        if math.gcd( ni, prod ) > 1:
            raise ZeroDivisionError("Moduli have a common factor.")
        prod *= ni
    res = 0
    for ni, xi in zip( ns, xs ):
        qi = prod // ni
        res += xi * pow( qi, -1, ni ) * qi
        res = res % prod
    return res

def miller_rabin( n: int, max_rounds: int = 40, verbose: int = 0, candidates: Sequence[int] = [] ) -> bool:
    """
    Use the Miller-Rabin primality test to check if n is a pseudoprime.

        Parameters:
            n (int): a positive integer
            max_rounds (int): a positive integer
            verbose=0 (flag): show more information
            candidates ([int]): a list of bases to test
                If candidates=[], bases are randomly selected.

        Returns:
            miller_rabin( n, max_rounds ): False, if a witness against primality is found
                in the first max_rounds test rounds, True otherwise
    Examples:
    >>> random.seed(0); miller_rabin( 113, max_rounds=4, verbose=2 )
    '      Testing with basis a=110 '      a**7 == 73'      a**14 == 98'      a**28 == 112'      ... success!
    '      Testing with basis a=51 '      a**7 == 69'      a**14 == 112'      ... success!
    '      Testing with basis a=99 '      a**7 == 15'      ... success!
    '      Testing with basis a=55 '      a**7 == 35'      a**14 == 98'      a**28 == 112'      ... success!
    True
    >>> miller_rabin( 221, candidates=[47,8,15], verbose=1 )
    '         Testing with basis a=47 '         ... success!
    '         Testing with basis a=8 '         Witness found!
    False
    """
    if n%2 == 0 and n!=2: # Miler-Rabin is not made for even numbers
        return False
    if candidates == []:
        if max_rounds > n-3:
            _printv( verbose, 0, f"There are only {n-3} bases to test. Testing all bases." )
            candidates = range( 2, n-2 )
        else:
            candidates = [ random.randrange( 2, n-2 ) for _ in range( max_rounds ) ]

    # factor n-1 into 2**s * k with k odd
    k = n-1
    s = 0
    while( k%2 == 0 ):
        k //= 2
        s += 1
    # test for at most max_rounds rounds
    for a in candidates:
        # choose random basis a
        _printv( verbose, 0, f"Testing with basis a={a}", end=' ' )
        a = pow( a, k, n )
        _printv( verbose, 1, f"a**{k} == {a}" , end='')
        if a == 1:
            # a**k == 1
            _printv( verbose, 0, '... success!' )
            continue
        for r in range( s ):
            if a == n-1:
                # a**(2**r)*k == -1
                # not a witness
                _printv( verbose, 0, '... success!' )
                break
            a = pow( a, 2, n )
            if r>0:
                _printv( verbose, 1, f"a**{2**r*k} == {a}", end='' )
        else:
            # witness found
            _printv( verbose, 0, 'Witness found!' )
            return False
    # max_rounds rounds survived
    return True

def legendre( a: int, p: int ) -> int:
    """
    Returns the Legendre Symbol
        Parameters:
            a (int): an integer
            p (int): a prime number
        Returns:
            0, id a is 0 mod p
            1, if a is a square mod p
           -1, otherwise
    Examples:
    >>> legendre( 0, 11 )
    0
    >>> legendre( 3, 11 )
    1
    >>> legendre( 8, 11 )
    -1
    """
    assert isprime(p), "p must be prime."
    if a%p == 0:
        return 0
    elif pow(a, (p - 1)//2, p) == 1:
        return 1
    else:
        return -1

def sqrts_mod( a: int, p: int ) -> List[int]:
    """
    Return a list of integers r, st. (r**2)%p = a
        Parameters:
            a (int): an integer
            p (int): a prime number
        Returns:
            sqrts_mod( a, p ): a list of integers b, st. (b**2)%p = a
    Example:
    >>> sqrts_mod( 665820697, 1000000009)
    [378633312, 621366697]
    >>> sqrts_mod(13,31)
    []
    """
    assert isprime(p), "p must be prime."

    a = a%p

    if a==0:
        return [0]
    if legendre( a, p ) == -1:
        # not a square
        return []

    # Tonelli-Shanks from rosettacode.org
    q = p - 1
    s = 0
    while q%2 == 0:
        q //= 2
        s += 1
    if s == 1:
        # case p%4 == 3
        r = pow( a, (p+1)//4, p )
        return [ r, (-r)%p ]
    # find a non-square z
    z = next( filter( lambda z: legendre( z, p ) == -1 , range( 2, p ) ) )

    c = pow( z, q, p )
    r = pow( a, (q+1)//2, p )
    t = pow( a, q, p )
    m = s
    t2 = 0
    while (t-1)%p != 0:
        t2 = pow( t, 2, p )
        for i in range( 1, m ):
            if (t2-1)%p == 0:
                break
            t2 = pow( t2, 2, p )
        b = pow( c, 2**(m-i-1), p )
        r = (r*b) % p
        c = (b*b) % p
        t = (t*c) % p
        m = i
    return [ r, (-r)%p ]

def isprime( n: int ) -> bool:
    """
    Calls miller_rabin(n,max_rounds=64).
    I believe that the Miller Rabin test with 64 rounds is always correct.
    Examples:
    >>> isprime(119)
    False
    >>> isprime(419)
    True
    """
    if n == 0 or n == 1:
        return False
    # trial division
    for p in sympy.sieve.primerange( 2, min( math.isqrt(n)+1, 1_000_000 ) ):
        if n%p == 0:
            return False

    return miller_rabin( n, max_rounds=64 )

def multiplicative_order( a: int, n: int ) -> int:
    """
    Return the smallest positive number x, st. pow(a,x,n)==1.

        Parameters:
            a (int): an integer
            n (int): a positive integer

        Returns:
            multiplicative_order(a,p): the smallest positive number x, st. pow(a,x,n)==1.
    Example:
    >>> multiplicative_order( 19, 48 )
    4
    """
    assert math.gcd( a, n )==1, "a and n must be coprime."

    order = euler_phi( n )
    while True:
        pf = prime_factors( order, unique=True )
        for q in pf[::-1]:
            if pow( a, order//q, n ) == 1:
                order //= q
                break
        else:
            return order

def generator_mod( p: int ) -> int:
    """
    Returns an element of multiplicative oder p-1.

        Parameters:
            p (int): a prime number

        Returns:
            generator_mod( p ): an element of multiplicative oder p-1.
    Example:
    >>> random.seed(0); generator_mod( 37 )
    18
    """
    assert isprime( p ), "p must be prime"
    while True:
        g = random.randint( 2, p-2 )
        if multiplicative_order( g, p ) == p-1:
            return g

def next_prime( n: int ) -> int:
    """
    Returns the smallest prime greater than n.

        Parameters:
            n (int): a positive integer

        Returns:
            next_prime(n): the smallest prime greater than n.
    Example:
    >>> next_prime(2**50+2**30)
    1125900980584453
    """
    return sympy.ntheory.nextprime( n )

### elliptic curves

def hasse_bounds( p: int ) -> Tuple[int,int]:
    """
    Return a pair (a,b) of positive integers
    such that the order of a curve mod p
    is between a and b (including a and b).
    Example:
    >>> hasse_bounds(500)
    (457, 545)
    """
    lower = math.ceil( p + 1 - 2*math.sqrt(p) )
    upper = math.floor( p + 1 + 2*math.sqrt(p) )
    return ( lower, upper )

class EC:
    """
    A class to represent an eliptic curve with equation y^2==x^3+a*x+b mod p.

    Attributes:
        a: int
            Parameter a in the curve equation
        b: int
            Parameter b in the curve equation
        p: int
            Parameter p in the curve equation
        order: int
            named parameter; order of the curve (if known)
        progress_bar: bool
            named parameter; set to True, if a progress bar should be shown
            when computing the order of the curve

    Methods:
        zero():
            Returns the neutral element of the group
        points():
            Returns a generator for the points on the curve as Point's
        list_of_points():
            Returns a list with all points on the curve as Point's
q        random_point():
            Returns a random point on the curve
    """
    def __init__( self, a: int, b: int, p: int, progress_bar: bool = False, order: int = 0 ) -> None:
        """
        Construct a curve object.

            Parameters:
                a (int): an integer
                b (int): an integer
                p (int): a prime >3
                4a^3+27b^2 must be != 0 (mod p) (i.e. non-singular curve)
                progress_bar (bool): named parameter; default=False.
                    For p>2**60 consider setting the parameter to True to see progress
                    when the order of the curve is computed.
                    For p<2**60, the progress bar is never shown.
                order (int): named parameter; order of the curve (if known)

            Returns:
                An object representing the curve y^2=x^3+ax+b mod p.
        """
        assert isprime( p ) and p>3, "argument prime must be a prime number greater than 3"
        assert (4*a**3+27*b**2)%p != 0, "the curve is singular (4a^3+27b^2=0 mod p)... parameters rejected"
        (l,u) = hasse_bounds(p)
        self.a: int = a
        self.b: int = b
        self.prime: int = p
        self._progress_bar: bool = False if p<2**60 else progress_bar # no progress bar, if computing the order is too easy
        # order == 0 means 'order unknown'
        if order!=0 and (order<l or order>u):
            print( "Warning: Given order is not inside the Hasse bounds. Ignoring the parameter." )
            order = 0
        # check if the given order is reasonable
        if order != 0:
            # new curve forcing the order
            test_ec = EC(a,b,p)
            test_ec.order = order
            # test for a sum of random points
            point = Point( test_ec, None )
            for _ in range( 20 ):
                point = point + test_ec.random_point()
            point = order * point
            # if multiplied with the curve order gives zero (should of course)
            if point != test_ec.zero():
                print( "Warning: Given order is wrong (not a multiple of point orders). Ignoring the parameter." )
                order = 0
        self._order: int = order

    def __str__( self ) -> str:
        """
        A string with the curve equation.
        """
        return f"y**2 = x**3 + {self.a}*x + {self.b} mod {self.prime}"

    def __repr__( self ) -> str:
        """
        A string generating the curve.
        """
        return f"EC( {self.a}, {self.b}, {self.prime} )"

    def zero( self ):
        """
        Return the neutral element of the group as a Point.
        """
        return Point( self, None )

    def random_point( self ):
        """
        Return a random point on the curve
        """
        if self._order > 0:
            # order is known, use it
            order = self.order
        else:
            # otherwise estimate the order (Hasse)
            order = self.prime
        if random.randrange( order ) == 0:
            # select identity element with correct probability
            return self.zero()

        while True:
            x = random.randrange( self.prime )
            y = sqrts_mod( x**3 + self.a*x + self.b, self.prime )
            if y != []:
                return ((-1)**random.randrange(2)) * Point( self, (x,y[0]) )

    def list_of_points( self ):
        """
        Compute a list of all points on the curve.

            Returns:
                A list with all points on the curve as Point's.
        Example:
        >>> EC(1,3,5).list_of_points()
        [Point( EC( 1, 3, 5 ), None ), Point( EC( 1, 3, 5 ), ( 1, 0 ) ), Point( EC( 1, 3, 5 ), ( 4, 1 ) ), Point( EC( 1, 3, 5 ), ( 4, 4 ) )]
        """
        prime = self.prime
        list_of_points = []
        with alive_bar( self.order, disable=not(self._progress_bar), force_tty=True ) as bar:
            for pt in self.points():
                list_of_points.append( pt )
                bar()

        return list_of_points

    def points( self ):
        """
        Returns:
            A generator for all points on the curve
        """
        prime = self.prime
        yield( Point( self, None ) )
        for x in range( prime ):
            for point in self.points_with_xcoord(x):
                yield point

    def points_with_xcoord( self, x: int ):
        """
        Returns:
            A list of all points on the curve with the given x-coordinate
        Examples:
        >>> EC( 19, 12, 37 ).points_with_xcoord( 23 )
        [Point( EC( 19, 12, 37 ), ( 23, 6 ) ), Point( EC( 19, 12, 37 ), ( 23, 31 ) )]
        >>> EC( 19, 12, 37 ).points_with_xcoord( 22 )
        []
        """
        prime = self.prime

        y2 = (x**3 + self.a*x + self.b)%prime

        if legendre( y2, prime ) != -1:
            return [ Point( self, ( x, y ) ) for y in sqrts_mod( y2, prime ) ]
        else:
            return []

    @property
    def order( self ) -> int:
        """
        Compute the order of the group.
        (stores its result in the _order attribute, because this can be tedious)
        Example:
        >>> EC(5,6,7).order
        4
        >>> EC(2,3,541,progress_bar=False).order
        528
        >>> EC(123,456,next_prime(78**9),progress_bar=False).order
        106868921208308308
        """
        if self._order > 0:
            return self._order
        prime = self.prime
        progress_bar = self._progress_bar

        if prime > 2**70:
            (l,u) = hasse_bounds(prime)
            print( f"The order of the group is between. {l} and {u}.")
            print( "Computing the exact order of a very large curve is not supported. Returning 0 (order unknown).")
            return 0
        elif prime < 81:
            # try all x coordinates
            order = 1
            with alive_bar( prime, disable = not(progress_bar), force_tty=True ) as bar:
                for x in range( prime ):
                    order += 1 + legendre( x**3 + self.a*x + self.b, prime )
                    bar()

            self._order = order

            return self._order
        else:
            # Elliptic Curves in Crptography (Blake, Seroussi, Smart)
            # Section IV.3.
            q = self.prime
            P = self.random_point()

            m = math.ceil( 2*math.sqrt(math.sqrt(q)))

            if progress_bar:
                print( 'Baby Steps' )
            baby_steps = {}
            resume = False
            jP = Point( self, None )
            with alive_bar( m, disable=not(progress_bar), force_tty=True ) as bar:
                for j in range( m ):
                    baby_steps[jP] = j
                    jP += P
                    if jP == self.zero():
                        resume = True
                        break
                    bar()
            if resume:
                if progress_bar:
                    print( 'Small order point selected ... resuming.' )
                return self.order

            mP = m*P
            Q1 = (q+1+math.floor(2*math.sqrt(q)))*P
            Qi = Q1
            if progress_bar:
                print( 'Giant Steps' )
            with alive_bar( disable=not(progress_bar), force_tty=True ) as bar:
                for i in range( m ):
                    if Qi in baby_steps:
                        o = q+1-(i*m+baby_steps[Qi]-math.floor(2*math.sqrt(q)))
                        """
                        computing the order of a point needs the curve order
                        use the candidate o
                        """
                        self.order = o
                        if P.order > 4*math.floor(math.sqrt(q)):
                            return o
                        else:
                            self.order = 0
                            # restart
                            resume = True
                    Qi -= mP
                    bar()
            if resume:
                if progress_bar:
                    print( 'Small order point selected ... resuming.' )
                return self.order
        return 0 # this is for the type checker, it is never reached

    @order.setter
    def order( self, o ):
        self._order = o
class Point:
    """
    A class to represent points on an elliptic curve.

    Attributes:
        x: int/None
            x coordinate of the point, None for the neutral element
        y: int/None
            y coordinate of the point, None for the neutral element
        iszero: bool
            True iff the point is the neutral element
        ec: EC
            The curve on which the point lies.

    Methods:
        add(q,verbose=0):
            Add the point to the point q, return the result as a Point.
            If verbose>0 more information is printed.
        inverse():
            Return the additive inverse of the point as a Point.
        double(verbose=0):
            Double the point, return the result as a Point.
            If verbose>0 more information is printed.
        mult(k,verbose=0):
            Compute the k fold multiple of the point, return the result as a Point.
            If verbose>0 more information is printed.
        order:
            Compute the order (smallest multiple equal to the neutral element) of the point.

        Infix operators for group operations are available:
            pt1+pt2 = pt1.add(pt2)
            -pt     = pt.inverse()
            pt1-pt2 = pt1.add(pt2.inverse())
            k*pt    = pt.mult(k)
        Equality can be checked with the '==' operator.
    """
    def __init__( self, ec: EC, coord: Tuple[int,int] | None ) -> None:
        """
        Construct an object representing a point on an elliptic curve.

            Parameters:
                ec (EC): An elliptic curve
                coord ((int,int)/None): A pair of integers representing the coordinates of the point,
                    None, if it is the neutral element.

            Returns:
                an object representing the point coord on the elliptic curve ec.
        Examples:
        >>> Point( EC(2,3,541),(4,285) )
        Point( EC( 2, 3, 541 ), ( 4, 285 ) )
        >>> Point( EC(2,3,541), None ) # neutral element
        Point( EC( 2, 3, 541 ), None )
        """

        if coord == None:
            self.iszero = True
#            self.x = None
#            self.y = None
            self._order = 1
        else:
            self.iszero = False
            self.x = coord[0] % ec.prime
            self.y = coord[1] % ec.prime
            self._order = 0 # unknown
            x, y = self.x, self.y
            a, b, p = ec.a, ec.b, ec.prime

            assert y**2%p == ( x**3 + a*x +b )%p, f"Warning: Point {coord} is not on the curve. Continuing ..."
        self.ec = ec

    def __str__( self ) -> str:
        """
        A string for the point as a coordinate pair.
        """
        if self.iszero:
            return "∞"
        else:
            return f"( {self.x}, {self.y} )"

    def __repr__( self ) -> str:
        """
        A string generating the point.
        """
        if self.iszero:
            return f"Point( {self.ec!r}, None )"
        else:
            return f"Point( {self.ec!r}, ( {self.x}, {self.y} ) )"

    def __hash__( self ):
        if self.iszero:
            return hash( None )
        else:
            return hash( (self.x,self.y) )

    def inverse( self ):
        """
        Compute the inverse of a point

            Returns:
                the additive inverse of the point as a Point.
        """
        if self.iszero:
            return self
        else:
            return Point( self.ec, ( self.x, -self.y ) )

    def __neg__( self ):
        """
        Overload unary '-' operator
        Example:
        >>> q = Point( EC(2,3,541), (4,285) )
        >>> -q
        Point( EC( 2, 3, 541 ), ( 4, 256 ) )
        """
        return self.inverse()

    def add( self, q, verbose: int = 0 ):
        """
        Add two arbitrary points (may be the same point twice).

            Parameters:
                q (Point): a point on the same elliptic curve
                verbose=0 (int): If verbose>0 more information is printed.

            Returns:
                The sum of self and q as a Point.
        """
        # compute tab with for verbose printing
        width = 5 + len( f"adding {self} and {q}:" )
        _printv( verbose, 0, width*'_' )
        _printv( verbose, 0, f"adding {self} and {q}:" )
        assert self.ec == q.ec
        # first point is zero
        if self.iszero:
            _printv( verbose, 0, f"{self} + {q} = {q}" )
            _printv( verbose, 0, width*'=' )
            return q
        # second point is zero
        if q.iszero:
            _printv( verbose, 0, f"{self} + {q} = {self}" )
            _printv( verbose, 0, width*'=')
            return self
        p = self.ec.prime
        x1, y1 = self.x, self.y
        x2, y2 = q.x, q.y
        if (x1-x2)%p == 0:
            # P+P
            if (y1-y2)%p == 0:
                return self.double( verbose )
            # P+(-P)
            else:
                _printv( verbose, 0, f"{self.ec.zero()}" )
                _printv( verbose, 0, width*'=' )
                return self.ec.zero()
        else:
            if ( x1-x2 )%p == 0:
                raise ZeroDivisionError( "(modular) division by zero" )
            k = ( (y1-y2) * inverse_mod( x1-x2, p, verbose-2 ) ) % p

            x3 = (k**2 - x1 - x2)%p
            y3 = (k*(x1-x3)-y1)%p
            _printv( verbose, 1, f"     k = ({y1}-{y2}) / ({x1}-{x2})" )
            _printv( verbose, 1, f"       = {(y1-y2)%p} / {(x1-x2)%p}" )
            _printv( verbose, 1, f"       = {(y1-y2)%p} * {pow(x1-x2,-1,p)}" )
            _printv( verbose, 1, f"       = {k}" )
            _printv( verbose, 1, f"    x3 = {k}**2 - {x1} - {x2}" )
            _printv( verbose, 1, f"       = {x3}" )
            _printv( verbose, 1, f"    y3 = -{y1} + {k}*({x1}-{x3})" )
            _printv( verbose, 1, f"       = -{y1}  + {k}*{(x1-x3)%p}" )
            _printv( verbose, 1, f"       = {y3}" )
            _printv( verbose, 0, f"{self} + {q} = {Point( self.ec, (x3, y3) )}" )
            _printv( verbose, 0, width*'=' )
            return Point( self.ec, (x3, y3) )

    def __add__( self, q ):
        """
        Overload '+' operator
        Example:
                Example:
        >>> curve = EC(2,3,541)
        >>> q = Point(curve,(4,285))
        >>> r = Point(curve, (93,151))
        >>> q+r
        Point( EC( 2, 3, 541 ), ( 67, 53 ) )
        """
        return self.add( q )

    def __sub__( self, q ):
        """
        Overload binary '-' operator
        Example:
        >>> curve = EC(2,3,541)
        >>> q = Point(curve,(4,285))
        >>> r = Point(curve, (93,151))
        >>> q-r
        Point( EC( 2, 3, 541 ), ( 429, 95 ) )

        """
        return self.add( q.inverse() )

    def double( self, verbose: int = 0 ):
        """
        Double a point

            Parameters:
                verbose=0 (int): If verbose>0 more information is printed.

            Returns:
                The sum of self and self as an Point.
        Example:
        >>> curve = EC(2,3,541)
        >>> q = Point(curve,(4,285))
        >>> q.double( verbose=2 )
        '      ______________________________
        '      doubling ( 4, 285 ):
        '           k = (3*4**2 + 2) / (2*285)
        '             = 50 / 29
        '             = 50 * 56
        '             = 95
        '          x3 = 95**2 - 2*4
        '             = 361
        '          y3 = -285 + 95*(4-361)
        '             = -285 + 95*184
        '             = 424
        '      2 * ( 4, 285 ) = ( 361, 424 )
        '      ==============================
        Point( EC( 2, 3, 541 ), ( 361, 424 ) )
        """
        # compute tab with for verbose printing
        width = 10+len( f"doubling {self}:" )
        _printv( verbose, 0, width*'_' )
        _printv( verbose, 0, f"doubling {self}:" )
        if self.iszero:
            _printv( verbose, 0, f"2 * {self} = {self}" )
            _printv( verbose, 0, width*'=' )
            return self
        p = self.ec.prime
        a = self.ec.a
        x1, y1 = self.x, self.y
        if y1%p ==0:
            _printv( verbose, 0, f"{self.ec.zero()}" )
            _printv( verbose, 0, width*'=' )
            return self.ec.zero()
        k = ( ( 3* x1**2 + a ) * inverse_mod( 2*y1, p, verbose-2 ) ) % p
        x3 = ( k**2 - 2*x1 )%p
        y3 = ( k*(x1-x3)-y1 )%p
        _printv( verbose, 1, f"     k = (3*{x1}**2 + {a}) / (2*{y1})" )
        _printv( verbose, 1, f"       = {(3* x1**2 + a)%p} / {(2*y1)%p}" )
        _printv( verbose, 1, f"       = {(3* x1**2 + a)%p} * {pow(2*y1,-1,p)}" )
        _printv( verbose, 1, f"       = {k}" )
        _printv( verbose, 1, f"    x3 = {k}**2 - 2*{x1}" )
        _printv( verbose, 1, f"       = {x3}" )
        _printv( verbose, 1, f"    y3 = -{y1} + {k}*({x1}-{x3})" )
        _printv( verbose, 1, f"       = -{y1} + {k}*{(x1-x3)%p}" )
        _printv( verbose, 1, f"       = {y3}" )
        _printv( verbose, 0, f"2 * {self} = {Point( self.ec, (x3, y3) )}" )
        _printv( verbose, 0, width*'=' )
        return Point( self.ec, (x3, y3) )

    def mult( self, k: int, verbose: int = 0 ):
        """
        Computes the result of adding the point k times with itself using the binary method.

            Parameters:
                k (int): an integer
                verbose=0 (int): If verbose>0 more information is printed.

            Returns:
                The point k*self as an Point.
        Example:
        >>> curve = EC(2,3,541)
        >>> q = Point(curve,(4,285))
        >>> q.mult(3)
        Point( EC( 2, 3, 541 ), ( 160, 268 ) )
        """
        _printv( verbose, 0, f"computing {k} * {self}" )
        if k<0:
            _printv( verbose, 0, f"computing {-k} * (- {self})" )
            return self.inverse().mult( -k, verbose )
        s = self.ec.zero()
        base = self.add( s, verbose=0 )
        while k!=0:
            if k%2==1:
                _printv( verbose, 0, "adding ... " )
                s = s.add( base, verbose-1 )
            if k>1:
                _printv( verbose, 0, "doubling ... " )
                base = base.double( verbose-1 )
            k //= 2
        _printv( verbose, 1, '' )
        return s

    def __rmul__( self, k: int ):
        """
        Overload the '*' operator.
        Example:
        >>> curve = EC(2,3,541)
        >>> q = Point(curve,(4,285))
        >>> 13*q
        Point( EC( 2, 3, 541 ), ( 447, 289 ) )
        """
        return self.mult( k )

    def _recursive_order( self, o: int ) -> int:
        """
        Internal function used to determine the order of a point in a recursive fashion.
        """
        plist = prime_factors( o, unique=True )
        for p in plist:
            if self.mult( o//p ).iszero:
                return self._recursive_order( o//p )
        return o

    @property
    def order( self ):
        """
        Compute the order of the point.

            Returns:
                The smallest (positive) k, st. k*self is equal to the neutral element.
        Example:
        >>> curve = EC(2,3,541,progress_bar=False)
        >>> q = Point(curve,(93,151))
        >>> q.order
        528
        """
        if self._order>0:
            return self._order
        curve = self.ec
        if self.ec.order==0:
            print( "Cannot compute the order of the point. Returning 0.")
            return 0
        curve_order = self.ec.order
        self._order = self._recursive_order( curve_order )
        return self._order

    def __eq__( self, other ):
        """
        Overload the '==' operator.
        """
        if self.ec.prime != other.ec.prime:
            return False

        if self.iszero or other.iszero:
            return self.iszero == other.iszero
        else:
            return (self.x, self.y) == (other.x, other.y)

def order2params( p, progress_bar = False ) -> dict[int,set[Tuple[int,int]]]:
    """
    Return a dictionary of all orders
    and sets of corresponding parameter pairs (a,b)
    (all curves mod p)
    Example:
    >>> order2params( 5 )
    {2: {(2, 0)}, 3: {(4, 2), (4, 3)}, 4: {(1, 0), (1, 2), (1, 3)}, 5: {(3, 2), (3, 3)}, 6: {(0, 1), (0, 2), (0, 3), (0, 4)}, 7: {(2, 4), (2, 1)}, 8: {(4, 4), (4, 0), (4, 1)}, 9: {(1, 1), (1, 4)}, 10: {(3, 0)}}
    """
    assert p<=10000, "p is too large (>10000). This would take too long ..."
    assert isprime(p), "p must be a prime number"
    order_param: dict[int,set[Tuple[int,int]]] = {}
    (l,u) = hasse_bounds(p)
    for o in range( l, u+1 ):
        order_param[o] = set()

    with alive_bar( p**2, disable=not(progress_bar), force_tty=True ) as bar:
        for a in range( p ):
            for b in range( p ):
                if (4*a**3 + 27*b**2) % p == 0: # singular curve
                    continue
                o = EC( a, b, p, progress_bar=progress_bar ).order
                order_param[o].add( (a,b) )
                bar()
    return order_param

### vector spaces

def gramschmidt( v ):
    """
    Return an orthonormal basis for the linear hull of v

        Parameters:
            v([np.array]): a list of vectors
                or a matrix with basis vectors as columns

        Returns:
            gramschmidt( v ): a list of orthonormal vectors that form a basis
                for the linear hull of the vectors in v
                (in the same format as the argument)
    Examples:
    >>> v1=array([2,4,4,0])
    >>> v2=array([4,3,4,2])
    >>> v3=array([-1,-3,-1,4])
    >>> gramschmidt([v1,v2,v3])
    [array([0.33333333, 0.66666667, 0.66666667, 0.        ]), array([ 0.66666667, -0.33333333,  0.        ,  0.66666667]), array([-0.66666667,  0.        ,  0.33333333,  0.66666667])]
    >>> b = array([v1,v2,v3]).T
    >>> gramschmidt(b)
    array([[ 0.33333333,  0.66666667, -0.66666667],
           [ 0.66666667, -0.33333333,  0.        ],
           [ 0.66666667,  0.        ,  0.33333333],
           [ 0.        ,  0.66666667,  0.66666667]])
    """
    if type(v) == ndarray:
        # argument is a matrix -> parse as a list of column vectors
        matrixtype = True
        v = [ v.T[i] for i in range( len(v.T) ) ]
    else:
        matrixtype = False

    u = len(v)*[0]
    for i in range( len(v) ):
        u[i] = v[i]
        for j in range( i ):
            u[i] = u[i] - ( v[i]@u[j] ) * u[j]  # project
        u[i] = 1/math.sqrt( u[i]@u[i] ) * u[i]  # normalize

    if matrixtype:
        # argument was a matrix -> return a matrix
        # argument is a list of vectors -> return a list of vectors
        return array( u ).T
    else:
        return u

### binary matrices

class BinaryMatrix(ndarray):
    """
    takes a numpy matrix, checks if all entries are either 0 or 1
    Methods:
        binary_rref(): compute reduced row echelon form modulo 2
        binary_nullspace(): compute a basis of the nullspace of the matrix
    """

    def __new__(cls, input_array):
        assert BinaryMatrix.isbinary(input_array), 'Matrix is not a binary matrix!'
        # Input array is an already formed ndarray instance
        # We first cast to be our class type
        obj = asarray(input_array).view(cls)
        # Finally, we must return the newly created object:
        return obj

    @classmethod
    def isbinary( self, mat ):
        """ a matrix is binary, if every entry is either 0 or 1 """
        return all( [ mat[i,j] == 0 or mat[i,j]==1 for i in range(mat.shape[0]) for j in range(mat.shape[1]) ] )

    def binary_rref( self ):
        """ compute the row echelon form of a binary matrix over GF2
        Example:
        >>> m = array( [[0,0,0,1,1,1,1],[0,1,1,0,0,1,1],[1,0,1,0,1,0,1]] )
        >>> b = BinaryMatrix(m)
        >>> b.binary_rref()
        (array([[1, 0, 1, 0, 1, 0, 1],
               [0, 1, 1, 0, 0, 1, 1],
               [0, 0, 0, 1, 1, 1, 1]]), [0, 1, 3])
        >>> b.binary_nullspace()
        BinaryMatrix([[1., 1., 0., 1.],
                      [1., 0., 1., 1.],
                      [1., 0., 0., 0.],
                      [0., 1., 1., 1.],
                      [0., 1., 0., 0.],
                      [0., 0., 1., 0.],
                      [0., 0., 0., 1.]])
        """
        mat = copy(self)
        m,n = mat.shape
        piv = []
        # subdiagonal
        row = 0
        for col in range( n ):
            if row<m:
                if mat[row,col]==0:
                    i = row+1
                    while i<m and mat[i,col]==0:
                        i += 1
                    if i<m:
                        mat[[row,i]] = mat[[i,row]]
                if mat[row,col]==1:
                    piv.append(col)
                    for i in range(row+1,m):
                        if mat[i,col]==1:
                            mat[i] = (mat[i]+mat[row])%2
                    row += 1
        # superdiagonal
        for col in range( n-1, -1, -1 ):
            if col in piv:
                row = piv.index(col)
                for i in range(row):
                    if mat[i,col]==1:
                        mat[i] = (mat[i]+mat[row])%2
        return mat, piv

    def binary_nullspace( self ):
        """ compute a basis for the null space of a binary matrix over GF2 """
        n = self.shape[1]
        rr, piv = self.binary_rref()
        nonpiv = [i for i in range(n) if not( i in piv ) ]
        basis = []
        for i in range(n):
            if i in piv:
                basis.append( list(rr[:,nonpiv][piv.index(i),:]) )
            else:
                basis.append( list(eye(len(nonpiv))[nonpiv.index(i),:]) )
        return BinaryMatrix(array(basis))

    def __matmul__( self, v ):
        """ inherited matrix multiplication is not mod 2 """
        return ndarray.__matmul__( self, v )%2


if __name__ == "__main__":
    import doctest
    print( 'Running doctests ...' )
    print( '(no output means no errors)' )
    doctest.testmod()
