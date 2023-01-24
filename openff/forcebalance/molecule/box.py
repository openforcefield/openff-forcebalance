from collections import namedtuple

import numpy

# Container for Bravais lattice vector.  Three cell lengths, three angles, three vectors, volume, and TINKER trig functions.
Box = namedtuple("Box", ["a", "b", "c", "alpha", "beta", "gamma", "A", "B", "C", "V"])
radian = 180.0 / numpy.pi


def CubicLattice(a):
    """This function takes in three lattice lengths and three lattice angles, and tries to return a complete box specification."""
    b = a
    c = a
    alpha, beta, gamma = 90, 90, 90
    alph = alpha * numpy.pi / 180
    bet = beta * numpy.pi / 180
    gamm = gamma * numpy.pi / 180
    v = numpy.sqrt(
        1
        - numpy.cos(alph) ** 2
        - numpy.cos(bet) ** 2
        - numpy.cos(gamm) ** 2
        + 2 * numpy.cos(alph) * numpy.cos(bet) * numpy.cos(gamm)
    )
    Mat = numpy.array(
        [
            [a, b * numpy.cos(gamm), c * numpy.cos(bet)],
            [
                0,
                b * numpy.sin(gamm),
                c
                * (
                    (numpy.cos(alph) - numpy.cos(bet) * numpy.cos(gamm))
                    / numpy.sin(gamm)
                ),
            ],
            [0, 0, c * v / numpy.sin(gamm)],
        ]
    )
    L1 = Mat.dot(numpy.array([[1], [0], [0]]))
    L2 = Mat.dot(numpy.array([[0], [1], [0]]))
    L3 = Mat.dot(numpy.array([[0], [0], [1]]))
    return Box(
        a,
        b,
        c,
        alpha,
        beta,
        gamma,
        numpy.array(L1).flatten(),
        numpy.array(L2).flatten(),
        numpy.array(L3).flatten(),
        v * a * b * c,
    )


def BuildLatticeFromLengthsAngles(a, b, c, alpha, beta, gamma):
    """This function takes in three lattice lengths and three lattice angles, and tries to return a complete box specification."""
    alph = alpha * numpy.pi / 180
    bet = beta * numpy.pi / 180
    gamm = gamma * numpy.pi / 180
    v = numpy.sqrt(
        1
        - numpy.cos(alph) ** 2
        - numpy.cos(bet) ** 2
        - numpy.cos(gamm) ** 2
        + 2 * numpy.cos(alph) * numpy.cos(bet) * numpy.cos(gamm)
    )
    Mat = numpy.array(
        [
            [a, b * numpy.cos(gamm), c * numpy.cos(bet)],
            [
                0,
                b * numpy.sin(gamm),
                c
                * (
                    (numpy.cos(alph) - numpy.cos(bet) * numpy.cos(gamm))
                    / numpy.sin(gamm)
                ),
            ],
            [0, 0, c * v / numpy.sin(gamm)],
        ]
    )
    L1 = Mat.dot(numpy.array([[1], [0], [0]]))
    L2 = Mat.dot(numpy.array([[0], [1], [0]]))
    L3 = Mat.dot(numpy.array([[0], [0], [1]]))
    return Box(
        a,
        b,
        c,
        alpha,
        beta,
        gamma,
        numpy.array(L1).flatten(),
        numpy.array(L2).flatten(),
        numpy.array(L3).flatten(),
        v * a * b * c,
    )


def BuildLatticeFromVectors(v1, v2, v3):
    """This function takes in three lattice vectors and tries to return a complete box specification."""
    a = numpy.linalg.norm(v1)
    b = numpy.linalg.norm(v2)
    c = numpy.linalg.norm(v3)
    alpha = (
        numpy.arccos(numpy.dot(v2, v3) / numpy.linalg.norm(v2) / numpy.linalg.norm(v3))
        * radian
    )
    beta = (
        numpy.arccos(numpy.dot(v1, v3) / numpy.linalg.norm(v1) / numpy.linalg.norm(v3))
        * radian
    )
    gamma = (
        numpy.arccos(numpy.dot(v1, v2) / numpy.linalg.norm(v1) / numpy.linalg.norm(v2))
        * radian
    )
    alph = alpha * numpy.pi / 180
    bet = beta * numpy.pi / 180
    gamm = gamma * numpy.pi / 180
    v = numpy.sqrt(
        1
        - numpy.cos(alph) ** 2
        - numpy.cos(bet) ** 2
        - numpy.cos(gamm) ** 2
        + 2 * numpy.cos(alph) * numpy.cos(bet) * numpy.cos(gamm)
    )
    Mat = numpy.array(
        [
            [a, b * numpy.cos(gamm), c * numpy.cos(bet)],
            [
                0,
                b * numpy.sin(gamm),
                c
                * (
                    (numpy.cos(alph) - numpy.cos(bet) * numpy.cos(gamm))
                    / numpy.sin(gamm)
                ),
            ],
            [0, 0, c * v / numpy.sin(gamm)],
        ]
    )
    L1 = Mat.dot(numpy.array([[1], [0], [0]]))
    L2 = Mat.dot(numpy.array([[0], [1], [0]]))
    L3 = Mat.dot(numpy.array([[0], [0], [1]]))
    return Box(
        a,
        b,
        c,
        alpha,
        beta,
        gamma,
        numpy.array(L1).flatten(),
        numpy.array(L2).flatten(),
        numpy.array(L3).flatten(),
        v * a * b * c,
    )
