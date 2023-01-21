""" @package forcebalance.openmmio OpenMM input/output.

@author Lee-Ping Wang
@date 04/2012
"""
import os
from collections import OrderedDict, defaultdict
from copy import deepcopy
from typing import List

import numpy
import openmm
from openmm import _openmm, app, unit

from openff.forcebalance import BaseReader
from openff.forcebalance.abinitio import AbInitio
from openff.forcebalance.binding import BindingEnergy
from openff.forcebalance.engine import Engine
from openff.forcebalance.hydration import Hydration
from openff.forcebalance.interaction import Interaction
from openff.forcebalance.liquid import Liquid
from openff.forcebalance.molecule import Molecule
from openff.forcebalance.moments import Moments
from openff.forcebalance.nifty import error, listfiles, warn_once, warn_press_key
from openff.forcebalance.opt_geo_target import OptGeoTarget
from openff.forcebalance.output import getLogger
from openff.forcebalance.torsion_profile import TorsionProfileTarget
from openff.forcebalance.vibration import Vibration

logger = getLogger(__name__)


def get_mask(grps):
    """Given a list of booleans [1, 0, 1] return the bitmask that sets
    these force groups appropriately in Context.getState(). Any values
    not provided are defaulted to 1."""
    mask = 0
    for i, j in enumerate(grps):
        # print i, j, 2**i
        mask += 2**i * j
    for k in range(i + 1, 31):
        # print k, 1, 2**k
        mask += 2**k
    return mask


def energy_components(Sim, verbose=False):
    # Before using EnergyComponents, make sure each Force is set to a different group.
    EnergyTerms = OrderedDict()
    if type(Sim.integrator) in [openmm.LangevinIntegrator, openmm.VerletIntegrator]:
        for i in range(Sim.system.getNumForces()):
            EnergyTerms[Sim.system.getForce(i).__class__.__name__] = (
                Sim.context.getState(getEnergy=True, groups=2**i).getPotentialEnergy()
                / unit.kilojoules_per_mole
            )
    return EnergyTerms


def get_multipoles(simulation, q=None, mass=None, positions=None, rmcom=True):
    """Return the current multipole moments in Debye and Buckingham units."""
    dx = 0.0
    dy = 0.0
    dz = 0.0
    qxx = 0.0
    qxy = 0.0
    qxz = 0.0
    qyy = 0.0
    qyz = 0.0
    qzz = 0.0
    enm_debye = 48.03204255928332  # Conversion factor from e*nm to Debye
    for i in simulation.system.getForces():
        if isinstance(i, openmm.NonbondedForce):
            # Get array of charges.
            if q is None:
                q = numpy.array(
                    [
                        i.getParticleParameters(j)[0]._value
                        for j in range(i.getNumParticles())
                    ]
                )
            # Get array of positions in nanometers.
            if positions is None:
                positions = simulation.context.getState(
                    getPositions=True
                ).getPositions()
            if mass is None:
                mass = numpy.array(
                    [
                        simulation.context.getSystem()
                        .getParticleMass(k)
                        .value_in_unit(unit.dalton)
                        for k in range(simulation.context.getSystem().getNumParticles())
                    ]
                )
            x = numpy.array(positions.value_in_unit(unit.nanometer))
            if rmcom:
                com = numpy.sum(x * mass.reshape(-1, 1), axis=0) / numpy.sum(mass)
                x -= com
            xx, xy, xz, yy, yz, zz = (
                x[:, k] * x[:, l]
                for k, l in [(0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2)]
            )
            # Multiply charges by positions to get dipole moment.
            dip = enm_debye * numpy.sum(x * q.reshape(-1, 1), axis=0)
            dx += dip[0]
            dy += dip[1]
            dz += dip[2]
            qxx += enm_debye * 15 * numpy.sum(q * xx)
            qxy += enm_debye * 15 * numpy.sum(q * xy)
            qxz += enm_debye * 15 * numpy.sum(q * xz)
            qyy += enm_debye * 15 * numpy.sum(q * yy)
            qyz += enm_debye * 15 * numpy.sum(q * yz)
            qzz += enm_debye * 15 * numpy.sum(q * zz)
            tr = qxx + qyy + qzz
            qxx -= tr / 3
            qyy -= tr / 3
            qzz -= tr / 3
    # This ordering has to do with the way TINKER prints it out.
    return [dx, dy, dz, qxx, qxy, qyy, qxz, qyz, qzz]


def get_dipole(simulation, q=None, mass=None, positions=None):
    """Return the current dipole moment in Debye.
    Note that this quantity is meaningless if the system carries a net charge."""
    return get_multipoles(simulation, q=q, mass=mass, positions=positions, rmcom=False)[
        :3
    ]


def PrepareVirtualSites(system):
    """Prepare a list of function wrappers and vsite parameters from the system."""
    isvsites = []
    vsfuncs = []
    vsidxs = []
    vswts = []
    for i in range(system.getNumParticles()):
        if system.isVirtualSite(i):
            isvsites.append(1)
            vs = system.getVirtualSite(i)
            if isinstance(vs, openmm.TwoParticleAverageSite):
                vsidx = [
                    _openmm.VirtualSite_getParticle(vs, 0),
                    _openmm.VirtualSite_getParticle(vs, 1),
                ]
                vswt = [
                    _openmm.TwoParticleAverageSite_getWeight(vs, 0),
                    _openmm.TwoParticleAverageSite_getWeight(vs, 1),
                ]

                def vsfunc(pos, idx_, wt_):
                    return wt_[0] * pos[idx_[0]] + wt_[1] * pos[idx_[1]]

            elif isinstance(vs, openmm.ThreeParticleAverageSite):
                vsidx = [
                    _openmm.VirtualSite_getParticle(vs, 0),
                    _openmm.VirtualSite_getParticle(vs, 1),
                    _openmm.VirtualSite_getParticle(vs, 2),
                ]
                vswt = [
                    _openmm.ThreeParticleAverageSite_getWeight(vs, 0),
                    _openmm.ThreeParticleAverageSite_getWeight(vs, 1),
                    _openmm.ThreeParticleAverageSite_getWeight(vs, 2),
                ]

                def vsfunc(pos, idx_, wt_):
                    return (
                        wt_[0] * pos[idx_[0]]
                        + wt_[1] * pos[idx_[1]]
                        + wt_[2] * pos[idx_[2]]
                    )

            elif isinstance(vs, openmm.OutOfPlaneSite):
                vsidx = [
                    _openmm.VirtualSite_getParticle(vs, 0),
                    _openmm.VirtualSite_getParticle(vs, 1),
                    _openmm.VirtualSite_getParticle(vs, 2),
                ]
                vswt = [
                    _openmm.OutOfPlaneSite_getWeight12(vs),
                    _openmm.OutOfPlaneSite_getWeight13(vs),
                    _openmm.OutOfPlaneSite_getWeightCross(vs),
                ]

                def vsfunc(pos, idx_, wt_):
                    v1 = pos[idx_[1]] - pos[idx_[0]]
                    v2 = pos[idx_[2]] - pos[idx_[0]]
                    cross = numpy.array(
                        [
                            v1[1] * v2[2] - v1[2] * v2[1],
                            v1[2] * v2[0] - v1[0] * v2[2],
                            v1[0] * v2[1] - v1[1] * v2[0],
                        ]
                    )
                    return pos[idx_[0]] + wt_[0] * v1 + wt_[1] * v2 + wt_[2] * cross

            elif isinstance(vs, openmm.LocalCoordinatesSite):
                vsidx = [
                    _openmm.VirtualSite_getParticle(vs, i)
                    for i in range(_openmm.VirtualSite_getNumParticles(vs))
                ]
                vswt = [
                    numpy.array(_openmm.LocalCoordinatesSite_getOriginWeights(vs)),
                    numpy.array(_openmm.LocalCoordinatesSite_getXWeights(vs)),
                    numpy.array(_openmm.LocalCoordinatesSite_getYWeights(vs)),
                    numpy.array(_openmm.LocalCoordinatesSite_getLocalPosition(vs)),
                ]

                def vsfunc(pos, idx_, wt_):
                    # origin weights
                    ows = wt_[0]
                    # xdir weights
                    xws = wt_[1]
                    # ydir weights
                    yws = wt_[2]
                    # vs position in local coordinates
                    vpos = wt_[3]
                    # dependent atom positions
                    dpos = numpy.array([pos[j] for j in idx_])
                    origin = numpy.array(dpos * ows[:, None]).sum(axis=0)
                    xdir = numpy.array(dpos * xws[:, None]).sum(axis=0)
                    ydir = numpy.array(dpos * yws[:, None]).sum(axis=0)
                    zdir = numpy.cross(xdir, ydir)
                    ydir = numpy.cross(zdir, xdir)
                    xdir /= numpy.linalg.norm(xdir)
                    ydir /= numpy.linalg.norm(ydir)
                    zdir /= numpy.linalg.norm(zdir)
                    return origin + numpy.array(
                        numpy.array([xdir, ydir, zdir]) * vpos[:, None]
                    ).sum(axis=0)

            else:
                raise NotImplementedError(
                    f"The virtual site type {vs.__class__.__name__} is not currently supported."
                )

        else:
            isvsites.append(0)
            vsfunc = None
            vsidx = None
            vswt = None
        vsfuncs.append(deepcopy(vsfunc))
        vsidxs.append(deepcopy(vsidx))
        vswts.append(deepcopy(vswt))
    return (isvsites, vsfuncs, vsidxs, vswts)


def ResetVirtualSites_fast(positions, vsinfo):
    """Given a set of OpenMM-compatible positions and a System object,
    compute the correct virtual site positions according to the System."""
    isvsites, vsfuncs, vsidxs, vswts = vsinfo
    if any(isvsites):
        pos = numpy.array(positions.value_in_unit(unit.nanometer))
        for i in range(len(positions)):
            if isvsites[i]:
                pos[i] = vsfuncs[i](pos, vsidxs[i], vswts[i])
        newpos = [openmm.Vec3(*i) for i in pos] * unit.nanometer
        return newpos
    else:
        return positions


def ResetVirtualSites(positions, system):
    """Given a set of OpenMM-compatible positions and a System object,
    compute the correct virtual site positions according to the System."""
    if any([system.isVirtualSite(j) for j in range(system.getNumParticles())]):
        pos = numpy.array(positions.value_in_unit(unit.nanometer))
        for i in range(system.getNumParticles()):
            if system.isVirtualSite(i):
                vs = system.getVirtualSite(i)
                # Faster code to work around Python API slowness.
                if isinstance(vs, openmm.TwoParticleAverageSite):
                    vspos = (
                        _openmm.TwoParticleAverageSite_getWeight(vs, 0)
                        * pos[_openmm.VirtualSite_getParticle(vs, 0)]
                        + _openmm.TwoParticleAverageSite_getWeight(vs, 1)
                        * pos[_openmm.VirtualSite_getParticle(vs, 1)]
                    )
                elif isinstance(vs, openmm.ThreeParticleAverageSite):
                    vspos = (
                        _openmm.ThreeParticleAverageSite_getWeight(vs, 0)
                        * pos[_openmm.VirtualSite_getParticle(vs, 0)]
                        + _openmm.ThreeParticleAverageSite_getWeight(vs, 1)
                        * pos[_openmm.VirtualSite_getParticle(vs, 1)]
                        + _openmm.ThreeParticleAverageSite_getWeight(vs, 2)
                        * pos[_openmm.VirtualSite_getParticle(vs, 2)]
                    )
                elif isinstance(vs, openmm.OutOfPlaneSite):
                    v1 = (
                        pos[_openmm.VirtualSite_getParticle(vs, 1)]
                        - pos[_openmm.VirtualSite_getParticle(vs, 0)]
                    )
                    v2 = (
                        pos[_openmm.VirtualSite_getParticle(vs, 2)]
                        - pos[_openmm.VirtualSite_getParticle(vs, 0)]
                    )
                    cross = openmm.Vec3(
                        v1[1] * v2[2] - v1[2] * v2[1],
                        v1[2] * v2[0] - v1[0] * v2[2],
                        v1[0] * v2[1] - v1[1] * v2[0],
                    )
                    vspos = (
                        pos[_openmm.VirtualSite_getParticle(vs, 0)]
                        + _openmm.OutOfPlaneSite_getWeight12(vs) * v1
                        + _openmm.OutOfPlaneSite_getWeight13(vs) * v2
                        + _openmm.OutOfPlaneSite_getWeightCross(vs) * cross
                    )
                pos[i] = vspos
        newpos = [openmm.Vec3(*i) for i in pos] * unit.nanometer
        return newpos
    else:
        return positions


def GetVirtualSiteParameters(system):
    """Return an array of all virtual site parameters in the system.  Used for comparison purposes."""
    vsprm = []
    for i in range(system.getNumParticles()):
        if system.isVirtualSite(i):
            vs = system.getVirtualSite(i)
            vstype = vs.__class__.__name__
            if vstype == "TwoParticleAverageSite":
                vsprm.append(_openmm.TwoParticleAverageSite_getWeight(vs, 0))
                vsprm.append(_openmm.TwoParticleAverageSite_getWeight(vs, 1))
            elif vstype == "ThreeParticleAverageSite":
                vsprm.append(_openmm.ThreeParticleAverageSite_getWeight(vs, 0))
                vsprm.append(_openmm.ThreeParticleAverageSite_getWeight(vs, 1))
                vsprm.append(_openmm.ThreeParticleAverageSite_getWeight(vs, 2))
            elif vstype == "OutOfPlaneSite":
                vsprm.append(_openmm.OutOfPlaneSite_getWeight12(vs))
                vsprm.append(_openmm.OutOfPlaneSite_getWeight13(vs))
                vsprm.append(_openmm.OutOfPlaneSite_getWeightCross(vs))
    return numpy.array(vsprm)


def CopyHarmonicBondParameters(src, dest):
    for i in range(src.getNumBonds()):
        dest.setBondParameters(i, *src.getBondParameters(i))


def CopyHarmonicAngleParameters(src, dest):
    for i in range(src.getNumAngles()):
        dest.setAngleParameters(i, *src.getAngleParameters(i))


def CopyPeriodicTorsionParameters(src, dest):
    for i in range(src.getNumTorsions()):
        dest.setTorsionParameters(i, *src.getTorsionParameters(i))


def CopyNonbondedParameters(src, dest):
    dest.setReactionFieldDielectric(src.getReactionFieldDielectric())
    for i in range(src.getNumParticles()):
        dest.setParticleParameters(i, *src.getParticleParameters(i))
    for i in range(src.getNumExceptions()):
        dest.setExceptionParameters(i, *src.getExceptionParameters(i))


def CopyGBSAOBCParameters(src, dest):
    dest.setSolventDielectric(src.getSolventDielectric())
    dest.setSoluteDielectric(src.getSoluteDielectric())
    for i in range(src.getNumParticles()):
        dest.setParticleParameters(i, *src.getParticleParameters(i))


def CopyCustomNonbondedParameters(src, dest):
    """
    copy whatever updateParametersInContext can update:
        per-particle parameters
    """
    for i in range(src.getNumParticles()):
        dest.setParticleParameters(i, list(src.getParticleParameters(i)))


def CopyCustomBondParameters(src, dest):
    """
    copy whatever updateParametersInContext can update:
        PerBondParameters
    """
    for i in range(src.getNumBonds()):
        dest.setBondParameters(i, *src.getBondParameters(i))


def do_nothing(src, dest):
    return


def CopySystemParameters(src, dest):
    """Copy parameters from one system (i.e. that which is created by a new force field)
    sto another system (i.e. the one stored inside the Target object).
    DANGER: These need to be implemented manually!!!"""
    Copiers = {
        "HarmonicBondForce": CopyHarmonicBondParameters,
        "HarmonicAngleForce": CopyHarmonicAngleParameters,
        "PeriodicTorsionForce": CopyPeriodicTorsionParameters,
        "NonbondedForce": CopyNonbondedParameters,
        "CustomBondForce": CopyCustomBondParameters,
        "CustomNonbondedForce": CopyCustomNonbondedParameters,
        "GBSAOBCForce": CopyGBSAOBCParameters,
        "CMMotionRemover": do_nothing,
    }
    for i in range(src.getNumForces()):
        nm = src.getForce(i).__class__.__name__
        if nm in Copiers:
            Copiers[nm](src.getForce(i), dest.getForce(i))
        else:
            warn_press_key(
                "There is no Copier function implemented for the OpenMM force type %s!"
                % nm
            )


def UpdateSimulationParameters(src_system, dest_simulation):
    CopySystemParameters(src_system, dest_simulation.system)
    for i in range(src_system.getNumForces()):
        if hasattr(dest_simulation.system.getForce(i), "updateParametersInContext"):
            dest_simulation.system.getForce(i).updateParametersInContext(
                dest_simulation.context
            )
        if isinstance(dest_simulation.system.getForce(i), openmm.CustomNonbondedForce):
            force = src_system.getForce(i)
            for j in range(force.getNumGlobalParameters()):
                pName = force.getGlobalParameterName(j)
                pValue = force.getGlobalParameterDefaultValue(j)
                dest_simulation.context.setParameter(pName, pValue)


def AddVirtualSiteBonds(mod, ff):
    # print "In AddVirtualSiteBonds"
    for ir, R in enumerate(list(mod.topology.residues())):
        A = list(R.atoms())
        # print "Residue", ir, ":", R.name
        for vs in ff._templates[R.name].virtualSites:
            vi = vs.index
            for ai in vs.atoms:
                bi = sorted([A[ai], A[vi]])
                # print "Adding Bond", ai, vi
                mod.topology.addBond(*bi)


def MTSVVVRIntegrator(temperature, collision_rate, timestep, system, ninnersteps=4):
    """
    Create a multiple timestep velocity verlet with velocity randomization (VVVR) integrator.

    ARGUMENTS

    temperature (Quantity compatible with unit.kelvin) - the temperature
    collision_rate (Quantity compatible with 1/picoseconds) - the collision rate
    timestep (Quantity compatible with femtoseconds) - the integration timestep
    system (openmm.System) - system whose forces will be partitioned
    ninnersteps (int) - number of inner timesteps (default: 4)

    RETURNS

    integrator (openmm.CustomIntegrator) - a VVVR integrator

    NOTES

    This integrator is equivalent to a Langevin integrator in the velocity Verlet discretization with a
    timestep correction to ensure that the field-free diffusion constant is timestep invariant.  The inner
    velocity Verlet discretization is transformed into a multiple timestep algorithm.

    REFERENCES

    VVVR Langevin integrator:
    * http://arxiv.org/abs/1301.3800
    * http://arxiv.org/abs/1107.2967 (to appear in PRX 2013)

    TODO

    Move initialization of 'sigma' to setting the per-particle variables.

    """
    # Multiple timestep Langevin integrator.
    for i in system.getForces():
        if i.__class__.__name__ in [
            "NonbondedForce",
            "CustomNonbondedForce",
        ]:
            # Slow force.
            logger.info(i.__class__.__name__ + " is a Slow Force\n")
            i.setForceGroup(1)
        else:
            logger.info(i.__class__.__name__ + " is a Fast Force\n")
            # Fast force.
            i.setForceGroup(0)

    kB = unit.BOLTZMANN_CONSTANT_kB * unit.AVOGADRO_CONSTANT_NA
    kT = kB * temperature

    integrator = openmm.CustomIntegrator(timestep)

    integrator.addGlobalVariable(
        "dt_fast", timestep / float(ninnersteps)
    )  # fast inner timestep
    integrator.addGlobalVariable("kT", kT)  # thermal energy
    integrator.addGlobalVariable(
        "a", numpy.exp(-collision_rate * timestep)
    )  # velocity mixing parameter
    integrator.addGlobalVariable(
        "b",
        numpy.sqrt(
            (2 / (collision_rate * timestep))
            * numpy.tanh(collision_rate * timestep / 2)
        ),
    )  # timestep correction parameter
    integrator.addPerDofVariable("sigma", 0)
    integrator.addPerDofVariable("x1", 0)  # position before application of constraints

    #
    # Pre-computation.
    # This only needs to be done once, but it needs to be done for each degree of freedom.
    # Could move this to initialization?
    #
    integrator.addComputePerDof("sigma", "sqrt(kT/m)")

    #
    # Velocity perturbation.
    #
    integrator.addComputePerDof("v", "sqrt(a)*v + sqrt(1-a)*sigma*gaussian")
    integrator.addConstrainVelocities()

    #
    # Symplectic inner multiple timestep.
    #
    integrator.addUpdateContextState()
    integrator.addComputePerDof("v", "v + 0.5*b*dt*f1/m")
    for innerstep in range(ninnersteps):
        # Fast inner symplectic timestep.
        integrator.addComputePerDof("v", "v + 0.5*b*dt_fast*f0/m")
        integrator.addComputePerDof("x", "x + v*b*dt_fast")
        integrator.addComputePerDof("x1", "x")
        integrator.addConstrainpositions()
        integrator.addComputePerDof("v", "v + 0.5*b*dt_fast*f0/m + (x-x1)/dt_fast")
    integrator.addComputePerDof(
        "v", "v + 0.5*b*dt*f1/m"
    )  # TODO: Additional velocity constraint correction?
    integrator.addConstrainVelocities()

    #
    # Velocity randomization
    #
    integrator.addComputePerDof("v", "sqrt(a)*v + sqrt(1-a)*sigma*gaussian")
    integrator.addConstrainVelocities()

    return integrator


"""Dictionary for building parameter identifiers.  As usual they go like this:
Bond/length/OW.HW
The dictionary is two-layered because the same interaction type (Bond)
could be under two different parent types (HarmonicBondForce, AmoebaHarmonicBondForce)"""
suffix_dict = {
    "HarmonicBondForce": {"Bond": ["class1", "class2"]},
    "HarmonicAngleForce": {
        "Angle": ["class1", "class2", "class3"],
    },
    "PeriodicTorsionForce": {
        "Proper": ["class1", "class2", "class3", "class4"],
        "Improper": ["class1", "class2", "class3", "class4"],
    },
    "NonbondedForce": {"Atom": ["type"]},
    "CustomBondForce": {
        "Bond": ["class1", "class2"],
    },
    "CustomNonbondedForce": {"Atom": ["class"]},
    "GBSAOBCForce": {"Atom": ["type"]},
    "Residues.Residue": {"VirtualSite": ["index"]},
    # LPW's custom parameter definitions
    "ForceBalance": {"GB": ["type"]},
}

# pdict is a useless variable if the force field is XML.
pdict = "XML_Override"


class OpenMM_Reader(BaseReader):
    """Class for parsing OpenMM force field files."""

    def __init__(self, fnm):
        super().__init__(fnm)
        self.pdict = pdict

    def build_pid(self, element, parameter):
        """Build the parameter identifier (see _link_ for an example)
        @todo Add a link here"""
        # InteractionType = ".".join([i.tag for i in list(element.iterancestors())][::-1][1:] + [element.tag])
        ParentType = ".".join([i.tag for i in list(element.iterancestors())][::-1][1:])
        InteractionType = element.tag
        try:
            if ParentType == "Residues.Residue":
                pfx = list(element.iterancestors())[0].attrib["name"]
                Involved = ".".join(
                    [
                        pfx + "-" + element.attrib[i]
                        for i in suffix_dict[ParentType][InteractionType]
                    ]
                )
            else:
                Involved1 = ".".join(
                    [
                        element.attrib[i]
                        for i in suffix_dict[ParentType][InteractionType]
                        if i in element.attrib
                    ]
                )
                suffix2 = [
                    i.replace("class", "type")
                    for i in suffix_dict[ParentType][InteractionType]
                ]
                suffix3 = [
                    i.replace("type", "class")
                    for i in suffix_dict[ParentType][InteractionType]
                ]
                Involved2 = ".".join(
                    [element.attrib[i] for i in suffix2 if i in element.attrib]
                )
                Involved3 = ".".join(
                    [element.attrib[i] for i in suffix3 if i in element.attrib]
                )
                # Keep the Involved string that is the longest (assuming that is the one that properly matched)
                Involved = [Involved1, Involved2, Involved3][
                    numpy.argmax(
                        numpy.array([len(Involved1), len(Involved2), len(Involved3)])
                    )
                ]
            return "/".join([InteractionType, parameter, Involved])
        except:
            logger.info(
                "Minor warning: Parameter ID %s doesn't contain any atom types, "
                "redundancies are possible\n" % ("/".join([InteractionType, parameter]))
            )
            return "/".join([InteractionType, parameter])


class OpenMM(Engine):

    """Derived from Engine object for carrying out general purpose OpenMM calculations."""

    def __init__(self, name="openmm", **kwargs):
        if not hasattr(self, "valkwd"):
            self.valkwd = [
                "ffxml",
                "pdb",
                "platname",
                "precision",
                "mmopts",
                "vsite_bonds",
                "restrain_k",
                "freeze_atoms",
            ]
        super().__init__(name=name, **kwargs)

    def setopts(self, platname="CUDA", precision="single", **kwargs):

        """Called by __init__ ; Set OpenMM-specific options."""

        # Target settings override.
        if hasattr(self, "target"):
            self.platname = self.target.platname
            self.precision = self.target.precision
        else:
            self.platname = platname
            self.precision = precision

        available_platforms: List[str] = [
            openmm.Platform.getPlatform(i).getName()
            for i in range(openmm.Platform.getNumPlatforms())
        ]
        if self.platname not in available_platforms:
            raise ValueError(self.platname)

        self.precision = self.precision.lower()

        available_precisions = ["single", "mixed", "double"]

        if self.precision not in available_precisions:
            raise ValueError(self.precision)

        if self.verbose:
            logger.info("Setting Platform to %s\n" % self.platname)

        self.platform = openmm.Platform.getPlatformByName(self.platname)

        if self.platname == "CUDA":
            # Set the device to the environment variable or zero otherwise
            device = os.environ.get("CUDA_DEVICE", "0")
            if self.verbose:
                logger.info("Setting CUDA Device to %s\n" % device)
            self.platform.setPropertyDefaultValue("CudaDeviceIndex", device)
            if self.verbose:
                logger.info("Setting CUDA Precision to %s\n" % self.precision)
            self.platform.setPropertyDefaultValue("CudaPrecision", self.precision)
        elif self.platname == "OpenCL":
            device = os.environ.get("OPENCL_DEVICE", "0")
            if self.verbose:
                logger.info("Setting OpenCL Device to %s\n" % device)
            self.platform.setPropertyDefaultValue("OpenCLDeviceIndex", device)
            if self.verbose:
                logger.info("Setting OpenCL Precision to %s\n" % self.precision)
            self.platform.setPropertyDefaultValue("OpenCLPrecision", self.precision)

        self.simkwargs = {}

    def readsrc(self, **kwargs):

        """Called by __init__ ; read files from the source directory.
        Provide a molecule object or a coordinate file.
        Add an optional PDB file for residues, atom names etc."""

        pdbfnm = None
        # Determine the PDB file name.
        if "pdb" in kwargs and os.path.exists(kwargs["pdb"]):
            # Case 1. The PDB file name is provided explicitly
            pdbfnm = kwargs["pdb"]
            if not os.path.exists(pdbfnm):
                logger.error("%s specified but doesn't exist\n" % pdbfnm)
                raise RuntimeError

        if "mol" in kwargs:
            self.mol = kwargs["mol"]
        elif "coords" in kwargs:
            self.mol = Molecule(kwargs["coords"])
            if pdbfnm is None and kwargs["coords"].endswith(".pdb"):
                pdbfnm = kwargs["coords"]
        else:
            logger.error("Must provide either a molecule object or coordinate file.\n")
            raise RuntimeError

        # If the PDB file exists, then it is copied directly to create
        # the OpenMM pdb object rather than being written by the
        # Molecule class.
        if pdbfnm is not None:
            self.abspdb = os.path.abspath(pdbfnm)
            mpdb = app.Molecule(pdbfnm)
            for i in ["chain", "atomname", "resid", "resname", "elem"]:
                self.mol.Data[i] = mpdb.Data[i]

        # Store a separate copy of the molecule for reference restraint positions.
        self.ref_mol = deepcopy(self.mol)

    def prepare(self, pbc=False, mmopts={}, **kwargs):

        """
        Prepare the calculation.  Note that we don't create the
        Simulation object yet, because that may depend on MD
        integrator parameters, thermostat, barostat etc.
        """
        # Introduced to attempt to fix a bug, but didn't work,
        # Might be sensible code anyway.
        # if hasattr(self.mol, 'boxes') and not pbc:
        #     del self.mol.Data['boxes']
        # if pbc and not hasattr(self.mol, 'boxes'):
        #     logger.error('Requested periodic boundary conditions but coordinate file contains no boxes')
        #     raise RuntimeError

        # Create the OpenMM PDB object.
        if hasattr(self, "abspdb"):
            self.pdb = app.PDBFile(self.abspdb)
        else:
            pdb1 = "%s-1.pdb" % os.path.splitext(os.path.basename(self.mol.fnm))[0]
            self.mol[0].write(pdb1)
            self.pdb = app.PDBFile(pdb1)
            os.unlink(pdb1)

        # Create the OpenMM ForceField object.
        if hasattr(self, "FF"):
            self.ffxml = [self.FF.openmmxml]
            self.forcefield = app.ForceField(
                os.path.join(self.root, self.FF.ffdir, self.FF.openmmxml)
            )
        else:
            self.ffxml = listfiles(kwargs.get("ffxml"), "xml", err=True)
            self.forcefield = app.ForceField(*self.ffxml)

        # Create bonds between virtual sites and their host atoms.
        # This is mainly for setting up AMOEBA multipoles.
        self.vbonds = kwargs.get("vsite_bonds", 0)

        # OpenMM options for setting up the System.
        self.mmopts = dict(mmopts)

        # Specify frozen atoms and restraint force constant
        if "restrain_k" in kwargs:
            self.restrain_k = kwargs["restrain_k"]
        if "freeze_atoms" in kwargs:
            self.freeze_atoms = kwargs["freeze_atoms"][:]

        # Set system options from ForceBalance force field options.
        if hasattr(self, "FF"):
            self.mmopts["rigidWater"] = self.FF.rigid_water
            if self.FF.constrain_h is True:
                self.mmopts["constraints"] = app.HBonds
                logger.info("Constraining hydrogen bond lengths (SHAKE)")

        # Set system options from periodic boundary conditions.
        self.pbc = pbc
        if pbc:
            minbox = min(
                [self.mol.boxes[0].a, self.mol.boxes[0].b, self.mol.boxes[0].c]
            )
            # Here we will set the CutoffPeriodic so custom nonbonded forces may be used.
            # However, we will turn PME on for AmoebaMultipoleForce and NonbondedForce after the system is created.
            self.SetPME = True
            # LPW: THIS CAUSES ISSUES! (AMOEBA system refuses to be created)
            # self.mmopts.setdefault('nonbondedMethod', CutoffPeriodic)
            self.mmopts.setdefault("nonbondedMethod", app.PME)
            if False:
                nonbonded_cutoff = kwargs.get("nonbonded_cutoff", 7.0)
                vdw_cutoff = kwargs.get("nonbonded_cutoff", 8.5)
                vdw_cutoff = kwargs.get("vdw_cutoff", vdw_cutoff)
                # Conversion to nanometers
                nonbonded_cutoff /= 10
                vdw_cutoff /= 10
                if "nonbonded_cutoff" in kwargs and "vdw_cutoff" not in kwargs:
                    warn_press_key(
                        "AMOEBA detected and nonbonded_cutoff is set, but not vdw_cutoff; it will be set equal to nonbonded_cutoff"
                    )
                if nonbonded_cutoff > 0.05 * (float(int(minbox - 1))):
                    warn_press_key(
                        "nonbonded_cutoff = {:.1f} should be smaller than half the box size = {:.1f} Angstrom".format(
                            nonbonded_cutoff * 10, minbox
                        )
                    )
                if vdw_cutoff > 0.05 * (float(int(minbox - 1))):
                    warn_press_key(
                        "vdw_cutoff = {:.1f} should be smaller than half the box size = {:.1f} Angstrom".format(
                            vdw_cutoff * 10, minbox
                        )
                    )
                self.mmopts.setdefault(
                    "nonbondedCutoff", nonbonded_cutoff * unit.nanometer
                )
                self.mmopts.setdefault("vdwCutoff", vdw_cutoff * unit.nanometer)
                self.mmopts.setdefault("aEwald", 5.4459052)
                # self.mmopts.setdefault('pmeGridDimensions', [24,24,24])
            else:
                if "vdw_cutoff" in kwargs:
                    warn_press_key(
                        "AMOEBA not detected, your provided vdw_cutoff will not be used"
                    )
                nonbonded_cutoff = kwargs.get("nonbonded_cutoff", 8.5)
                # Conversion to nanometers
                nonbonded_cutoff /= 10
                if nonbonded_cutoff > 0.05 * (float(int(minbox - 1))):
                    warn_press_key(
                        "nonbonded_cutoff = {:.1f} should be smaller than half the box size = {:.1f} Angstrom".format(
                            nonbonded_cutoff * 10, minbox
                        )
                    )

                self.mmopts.setdefault(
                    "nonbondedCutoff", nonbonded_cutoff * unit.nanometer
                )
                self.mmopts.setdefault("useSwitchingFunction", True)
                self.mmopts.setdefault(
                    "switchingDistance", (nonbonded_cutoff - 0.1) * unit.nanometer
                )
            self.mmopts.setdefault("useDispersionCorrection", True)
        else:
            if "nonbonded_cutoff" in kwargs or "vdw_cutoff" in kwargs:
                warn_press_key(
                    "No periodic boundary conditions, your provided nonbonded_cutoff and vdw_cutoff will not be used"
                )
            self.SetPME = False
            self.mmopts.setdefault("nonbondedMethod", app.NoCutoff)
            self.mmopts["removeCMMotion"] = False

        # Generate list of OpenMM-compatible positions
        mod = self.generate_xyz_omm(self.mol)
        # Build a topology and atom lists.
        Top = mod.getTopology()
        Atoms = list(Top.atoms())
        [(a.index, b.index) for a, b in list(Top.bonds())]

        # vss = [(i, [system.getVirtualSite(i).getParticle(j) for j in range(system.getVirtualSite(i).getNumParticles())]) \
        #            for i in range(system.getNumParticles()) if system.isVirtualSite(i)]
        self.AtomLists = defaultdict(list)
        self.AtomLists["Mass"] = [
            a.element.mass.value_in_unit(unit.dalton) if a.element is not None else 0
            for a in Atoms
        ]
        self.AtomLists["ParticleType"] = [
            "A" if m >= 1.0 else "D" for m in self.AtomLists["Mass"]
        ]
        self.AtomLists["ResidueNumber"] = [a.residue.index for a in Atoms]
        self.AtomMask = [a == "A" for a in self.AtomLists["ParticleType"]]
        self.realAtomIdxs = [i for i, a in enumerate(self.AtomMask) if a is True]

    def generate_xyz_omm(self, mol):
        # Generate OpenMM-compatible positions
        self.xyz_omms = []
        for I in range(len(mol)):
            xyz = mol.xyzs[I]
            xyz_omm = [openmm.Vec3(i[0], i[1], i[2]) for i in xyz] * unit.angstrom
            # An extra step with adding virtual particles
            mod = app.Modeller(self.pdb.topology, xyz_omm)
            mod.addExtraParticles(self.forcefield)
            if self.pbc:
                # Obtain the periodic box
                if (
                    mol.boxes[I].alpha != 90.0
                    or mol.boxes[I].beta != 90.0
                    or mol.boxes[I].gamma != 90.0
                ):
                    logger.error("OpenMM cannot handle nonorthogonal boxes.\n")
                    raise RuntimeError
                box_omm = [
                    openmm.Vec3(mol.boxes[I].a, 0, 0) * unit.angstrom,
                    openmm.Vec3(0, mol.boxes[I].b, 0) * unit.angstrom,
                    openmm.Vec3(0, 0, mol.boxes[I].c) * unit.angstrom,
                ]
            else:
                box_omm = None
            # Finally append it to list.
            self.xyz_omms.append((mod.getPositions(), box_omm))
        return mod

    def create_simulation(
        self,
        timestep=1.0,
        faststep=0.25,
        temperature=None,
        pressure=None,
        anisotropic=False,
        mts=False,
        collision=1.0,
        nbarostat=25,
        rpmd_beads=0,
        **kwargs,
    ):

        """
        Create simulation object.  Note that this also takes in some
        options pertinent to system setup, including the type of MD
        integrator and type of pressure control.
        """

        # Divisor for the temperature (RPMD sets it to nonzero.)
        self.tdiv = 1

        # Determine the integrator.
        if temperature:
            # If temperature control is turned on, then run Langevin dynamics.
            if mts:
                if rpmd_beads > 0:
                    logger.error(
                        "No multiple timestep integrator without temperature control.\n"
                    )
                    raise RuntimeError
                integrator = openmm.MTSVVVRIntegrator(
                    temperature * unit.kelvin,
                    collision / unit.picosecond,
                    timestep * unit.femtosecond,
                    self.system,
                    ninnersteps=int(timestep / faststep),
                )
            else:
                if rpmd_beads > 0:
                    logger.info(
                        "Creating RPMD integrator with %i beads.\n" % rpmd_beads
                    )
                    self.tdiv = rpmd_beads
                    integrator = openmm.RPMDIntegrator(
                        rpmd_beads,
                        temperature * unit.kelvin,
                        collision / unit.picosecond,
                        timestep * unit.femtosecond,
                    )
                else:
                    integrator = openmm.LangevinIntegrator(
                        temperature * unit.kelvin,
                        collision / unit.picosecond,
                        timestep * unit.femtosecond,
                    )
        else:
            # If no temperature control, default to the Verlet integrator.
            if rpmd_beads > 0:
                logger.error("No RPMD integrator without temperature control.\n")
                raise RuntimeError
            if mts:
                warn_once(
                    "No multiple timestep integrator without temperature control."
                )
            integrator = openmm.VerletIntegrator(timestep * unit.femtoseconds)

        # Add the barostat.
        if pressure is not None:
            if anisotropic:
                barostat = openmm.MonteCarloAnisotropicBarostat(
                    [pressure, pressure, pressure] * unit.atmospheres,
                    temperature * unit.kelvin,
                    nbarostat,
                )
            else:
                barostat = openmm.MonteCarloBarostat(
                    pressure * unit.atmospheres, temperature * unit.kelvin, nbarostat
                )
        if self.pbc and pressure is not None:
            self.system.addForce(barostat)
        elif pressure is not None:
            warn_once("Pressure is ignored because pbc is set to False.")

        # Add a restraint force if we have one.
        self.restraint_frc_index = None
        if hasattr(self, "restrain_k") and self.restrain_k != 0.0:
            restraint_frc = openmm.CustomExternalForce(
                "0.5*k*((x-x0)^2+(y-y0)^2+(z-z0)^2)"
            )
            restraint_frc.addGlobalParameter(
                "k", self.restrain_k * unit.kilocalorie_per_mole / unit.angstrom**2
            )
            restraint_frc.addPerParticleParameter("x0")
            restraint_frc.addPerParticleParameter("y0")
            restraint_frc.addPerParticleParameter("z0")
            for i, j in enumerate(self.realAtomIdxs):
                restraint_frc.addParticle(j)
                restraint_frc.setParticleParameters(i, j, [0.0, 0.0, 0.0])
            self.restraint_frc_index = self.system.addForce(restraint_frc)

        # Freeze atoms if we have any.
        if hasattr(self, "freeze_atoms"):
            for i in self.freeze_atoms:
                j = self.realAtomIdxs[i]
                self.system.setParticleMass(j, 0.0)

        # Set up for energy component analysis.
        GrpTogether = [
            "CustomNonbondedForce",
            "NonbondedForce",
        ]
        GrpNums = {}
        if not mts:
            for j in self.system.getForces():
                i = -1
                if j.__class__.__name__ in GrpTogether:
                    for k in GrpNums:
                        if k in GrpTogether:
                            i = GrpNums[k]
                            break
                if i == -1:
                    i = len(set(GrpNums.values()))
                GrpNums[j.__class__.__name__] = i
                j.setForceGroup(i)

        # test: exclude all Amoeba Nonbonded Forces within each residue
        # SetAmoebaNonbondedExcludeAll(self.system, self.mod.topology)

        # Finally create the simulation object.
        self.simulation = app.Simulation(
            self.mod.topology, self.system, integrator, self.platform
        )

    def update_simulation(self, **kwargs):

        """
        Create the simulation object, or update the force field
        parameters in the existing simulation object.  This should be
        run when we write a new force field XML file.
        """
        if len(kwargs) > 0:
            self.simkwargs = kwargs
        self.forcefield = app.ForceField(*self.ffxml)

        self.mod = openmm.app.Modeller(self.pdb.topology, self.pdb.positions)
        self.mod.addExtraParticles(self.forcefield)
        # Add bonds for virtual sites. (Experimental)
        if self.vbonds:
            AddVirtualSiteBonds(self.mod, self.forcefield)

        try:
            self.system = self.forcefield.createSystem(self.mod.topology, **self.mmopts)
        # This try/except block catches a failure case introduced by the release of openmm 7.7
        # where a ValueError would be raised if createSystem was given an unused kwarg.
        # Now, when that error occurs, we remove the unused kwargs from mmopts.
        # More info at https://github.com/leeping/forcebalance/issues/246
        except ValueError as e:
            if "useSwitchingFunction" not in str(e):
                raise e
            self.mmopts.pop("useSwitchingFunction")
            self.mmopts.pop("switchingDistance")
            self.system = self.forcefield.createSystem(self.mod.topology, **self.mmopts)
        self.vsinfo = PrepareVirtualSites(self.system)
        self.nbcharges = numpy.zeros(self.system.getNumParticles())

        for i in self.system.getForces():
            if isinstance(i, openmm.NonbondedForce):
                self.nbcharges = numpy.array(
                    [
                        i.getParticleParameters(j)[0]._value
                        for j in range(i.getNumParticles())
                    ]
                )
                if self.SetPME:
                    i.setNonbondedMethod(i.PME)

        # ----
        # If the virtual site parameters have changed,
        # the simulation object must be remade.
        # ----
        vsprm = GetVirtualSiteParameters(self.system)
        if (
            hasattr(self, "vsprm")
            and len(self.vsprm) > 0
            and numpy.max(numpy.abs(vsprm - self.vsprm)) != 0.0
        ):
            if hasattr(self, "simulation"):
                delattr(self, "simulation")
        self.vsprm = vsprm.copy()

        # ----
        # Same with the Drude Particles, if the parameters
        # have changed then the positions must be recomputed and
        # the simulation object must be remade.
        # ----
        if hasattr(self, "simulation"):
            UpdateSimulationParameters(self.system, self.simulation)
        else:
            self.create_simulation(**self.simkwargs)

    def set_restraint_positions(self, shot):
        """
        Set reference positions for energy restraints.  This may be a different set of positions
        from the "current" positions that are stored in self.mol and self.xyz_omm.
        """
        if self.restraint_frc_index is not None:
            # Generate OpenMM-compatible positions in nanometers
            xyz = self.ref_mol.xyzs[shot] / 10.0
            frc = self.simulation.system.getForce(self.restraint_frc_index)
            for i, j in enumerate(self.realAtomIdxs):
                frc.setParticleParameters(i, j, xyz[i])
            frc.updateParametersInContext(self.simulation.context)
        else:
            raise RuntimeError(
                "Asked to set restraint positions, but no restraint force has been added to the system"
            )

    def set_positions(self, shot):

        """
        Set the positions and periodic box vectors to one of the
        stored coordinates.

        *** NOTE: If you run a MD simulation, then the coordinates are
        overwritten by the MD trajectory. ***
        """
        # ----
        # Ideally the virtual site parameters would be copied but they're not.
        # Instead we update the vsite positions manually.
        # ----
        # if self.FF.rigid_water:
        #     simulation.context.applyConstraints(1e-8)
        # else:
        #     simulation.context.computeVirtualSites()
        # ----
        # NOTE: Periodic box vectors must be set FIRST
        if self.pbc:
            self.simulation.context.setPeriodicBoxVectors(*self.xyz_omms[shot][1])
        # self.simulation.context.setPositions(ResetVirtualSites(self.xyz_omms[shot][0], self.system))
        # self.simulation.context.setPositions(ResetVirtualSites_fast(self.xyz_omms[shot][0], self.vsinfo))
        self.simulation.context.setPositions(self.xyz_omms[shot][0])
        self.simulation.context.computeVirtualSites()

    def get_charges(self):
        logger.error(
            "OpenMM engine does not have get_charges (should be trivial to implement however.)"
        )
        raise NotImplementedError

    def compute_volume(self, box_vectors):
        """Compute the total volume of an OpenMM system."""
        [a, b, c] = box_vectors
        A = numpy.array([a / a.unit, b / a.unit, c / a.unit])
        # Compute volume of parallelepiped.
        volume = numpy.linalg.det(A) * a.unit**3
        return volume

    def compute_mass(self, system):
        """Compute the total mass of an OpenMM system."""
        mass = 0.0 * unit.amu
        for i in range(system.getNumParticles()):
            mass += system.getParticleMass(i)
        return mass

    def evaluate_one_(self, force=False, dipole=False):
        """Perform a single point calculation on the current geometry."""

        state = self.simulation.context.getState(
            getPositions=dipole, getEnergy=True, getForces=force
        )
        Result = {}
        Result["Energy"] = state.getPotentialEnergy() / unit.kilojoules_per_mole
        if force:
            Force = state.getForces(asNumpy=True).value_in_unit(
                unit.kilojoule_per_mole / unit.nanometer
            )
            # Extract forces belonging to real atoms only
            Result["Force"] = Force[self.realAtomIdxs].flatten()
        if dipole:
            Result["Dipole"] = get_dipole(
                self.simulation,
                q=self.nbcharges,
                mass=self.AtomLists["Mass"],
                positions=state.getPositions(),
            )
        return Result

    def evaluate_(self, force=False, dipole=False, traj=False):

        """
        Utility function for computing energy, and (optionally) forces and dipoles using OpenMM.

        Inputs:
        force: Switch for calculating the force.
        dipole: Switch for calculating the dipole.
        traj: Trajectory (listing of coordinate and box 2-tuples).  If provide, will loop over these snapshots.
        Otherwise will do a single point evaluation at the current geometry.

        Outputs:
        Result: Dictionary containing energies, forces and/or dipoles.
        """

        self.update_simulation()

        # If trajectory flag set to False, perform a single-point calculation.
        if not traj:
            return evaluate_one_(force, dipole)
        Energies = []
        Forces = []
        Dipoles = []
        for I in range(len(self.xyz_omms)):
            self.set_positions(I)
            R1 = self.evaluate_one_(force, dipole)
            Energies.append(R1["Energy"])
            if force:
                Forces.append(R1["Force"])
            if dipole:
                Dipoles.append(R1["Dipole"])
        # Compile it all into the dictionary object
        Result = OrderedDict()

        Result["Energy"] = numpy.array(Energies)
        if force:
            Result["Force"] = numpy.array(Forces)
        if dipole:
            Result["Dipole"] = numpy.array(Dipoles)
        return Result

    def energy_one(self, shot):
        self.set_positions(shot)
        return self.evaluate_()["Energy"]

    def energy_force_one(self, shot):
        self.set_positions(shot)
        Result = self.evaluate_(force=True)
        return numpy.hstack((Result["Energy"].reshape(-1, 1), Result["Force"]))

    def energy(self):
        return self.evaluate_(traj=True)["Energy"]

    def energy_force(self):
        """Loop through the snapshots and compute the energies and forces using OpenMM."""
        Result = self.evaluate_(force=True, traj=True)
        Result["Energy"].reshape(-1, 1)
        Result["Force"]
        return numpy.hstack((Result["Energy"].reshape(-1, 1), Result["Force"]))

    def energy_dipole(self):
        """Loop through the snapshots and compute the energies and forces using OpenMM."""
        Result = self.evaluate_(dipole=True, traj=True)
        return numpy.hstack((Result["Energy"].reshape(-1, 1), Result["Dipole"]))

    def build_mass_weighted_hessian(
        self, shot=0, optimize=True, mass_weighted_hessian_only=True
    ):
        """OpenMM single frame hessian evaluation
        Since OpenMM doesnot provide a Hessian evaluation method, we used finite difference on forces

        Parameters
        ----------
        shot: int
            The frame number in the trajectory of this target

        Returns
        -------
        hessian: numpy.array with shape 3N x 3N, N = number of "real" atoms
            The result hessian matrix.
            The row indices are fx0, fy0, fz0, fx1, fy1, ...
            The column indices are x0, y0, z0, x1, y1, ..
            The unit is kilojoule / (nanometer^2 * mole * dalton) => 10^24 s^-2
        """
        self.update_simulation()
        if optimize is True:
            _, _, M_opt = self.optimize(shot, crit=1e-8)
        else:
            warn_once("Computing mass-weighted hessian without geometry optimization")
            self.set_positions(shot)
            # it returns M_opt with original position
            M_opt = deepcopy(self.mol[0])
            X0 = (
                self.simulation.context.getState(getPositions=True)
                .getPositions(asNumpy=True)
                .value_in_unit(unit.angstrom)[self.realAtomIdxs]
            )
            M_opt.xyzs = [X0]

        context = self.simulation.context
        pos = context.getState(getPositions=True).getPositions(asNumpy=True)
        # pull real atoms and their mass
        massList = numpy.array(self.AtomLists["Mass"])[
            self.realAtomIdxs
        ]  # unit in dalton
        # initialize an empty hessian matrix
        noa = len(self.realAtomIdxs)
        hessian = numpy.empty((noa * 3, noa * 3), dtype=float)
        mass_weighted_hessian = numpy.empty((noa * 3, noa * 3), dtype=float)
        # finite difference step size
        diff = unit.Quantity(0.0001, unit=unit.nanometer)
        coef = 1.0 / (0.0001 * 2)  # 1/2h
        for i, i_atom in enumerate(self.realAtomIdxs):
            massWeight = 1.0 / numpy.sqrt(massList * massList[i])
            # loop over the x, y, z coordinates
            for j in range(3):
                # plus perturbation
                pos[i_atom][j] += diff
                context.setPositions(pos)
                grad_plus = (
                    context.getState(getForces=True)
                    .getForces(asNumpy=True)
                    .value_in_unit(unit.kilojoule / (unit.nanometer * unit.mole))
                )
                grad_plus = -grad_plus[
                    self.realAtomIdxs
                ]  # gradients are negative forces
                # minus perturbation
                pos[i_atom][j] -= 2 * diff
                context.setPositions(pos)
                grad_minus = (
                    context.getState(getForces=True)
                    .getForces(asNumpy=True)
                    .value_in_unit(unit.kilojoule / (unit.nanometer * unit.mole))
                )
                grad_minus = -grad_minus[
                    self.realAtomIdxs
                ]  # gradients are negative forces
                # set the perturbation back to zero
                pos[i_atom][j] += diff
                # fill one row of the hessian matrix
                hessian[i * 3 + j] = numpy.ravel((grad_plus - grad_minus) * coef)
                mass_weighted_hessian[i * 3 + j] = numpy.ravel(
                    (grad_plus - grad_minus) * coef * massWeight[:, numpy.newaxis]
                )

        # make hessian symmetric by averaging upper right and lower left
        hessian += hessian.T
        hessian *= 0.5

        mass_weighted_hessian += mass_weighted_hessian.T
        mass_weighted_hessian *= 0.5

        # recover the original position
        context.setPositions(pos)
        gradient = (
            context.getState(getForces=True)
            .getForces(asNumpy=True)
            .value_in_unit(unit.kilojoule / (unit.nanometer * unit.mole))
            .flatten()
        )
        if mass_weighted_hessian_only:
            return mass_weighted_hessian
        else:
            return pos, gradient, hessian, mass_weighted_hessian, M_opt

    def normal_modes(self, shot=0, optimize=True, for_hessian_target=False):
        """OpenMM Normal Mode Analysis
        Since OpenMM doesnot provide a Hessian evaluation method, we used finite difference on forces

        Parameters
        ----------
        shot: int
            The frame number in the trajectory of this target
        optimize: bool, default True
            Optimize the geometry before evaluating the normal modes

        Returns
        -------
        freqs: numpy.array with shape (3N - 6) x 1, N = number of "real" atoms
            Harmonic frequencies, sorted from smallest to largest, with the 6 smallest removed, in unit cm^-1
        normal_modes: numpy.array with shape (3N - 6) x (3N), N = number of "real" atoms
            The normal modes corresponding to each of the frequencies, scaled by mass^-1/2.
        """
        if self.precision == "single":
            warn_once(
                "Single-precision OpenMM engine used for normal mode analysis - recommend that you use mix or double precision."
            )
        if not optimize:
            warn_once("Asking for normal modes without geometry optimization?")
        # step 0: check number of real atoms
        noa = len(self.realAtomIdxs)
        if noa < 2:
            error("normal mode analysis not suitable for system with one or less atoms")
        # step 1: build a full hessian matrix
        (
            coords,
            gradient,
            hessian,
            mass_weighted_hessian,
            M_opt,
        ) = self.build_mass_weighted_hessian(
            shot=shot, optimize=optimize, mass_weighted_hessian_only=False
        )

        # step 2: diagonalize the hessian matrix
        eigvals, eigvecs = numpy.linalg.eigh(mass_weighted_hessian)
        # step 3: convert eigenvalues to frequencies
        coef = 0.5 / numpy.pi * 33.3564095  # 10^12 Hz => cm-1
        negatives = (eigvals >= 0).astype(int) * 2 - 1  # record the negative ones
        freqs = numpy.sqrt(numpy.abs(eigvals)) * coef * negatives
        # step 4: convert eigenvectors to normal modes
        # re-arange to row index and shape
        normal_modes = eigvecs.T.reshape(noa * 3, noa, 3)
        # step 5: Remove mass weighting from eigenvectors
        massList = numpy.array(self.AtomLists["Mass"])[
            self.realAtomIdxs
        ]  # unit in dalton
        for i in range(normal_modes.shape[0]):
            mode = normal_modes[i]
            mode /= numpy.sqrt(massList[:, numpy.newaxis])
            mode /= numpy.linalg.norm(mode)
        # step 5: remove the 6 freqs with smallest abs value and corresponding normal modes
        n_remove = 5 if len(self.realAtomIdxs) == 2 else 6
        larger_freq_idxs = numpy.sort(
            numpy.argpartition(numpy.abs(freqs), n_remove)[n_remove:]
        )
        freqs = freqs[larger_freq_idxs]
        normal_modes = normal_modes[larger_freq_idxs]
        if not for_hessian_target:
            return freqs, normal_modes
        else:
            return coords, gradient, hessian, freqs, normal_modes, M_opt

    def _update_positions(self, X1, disable_vsite):
        """A convenience method for updating the positions of the simulation context."""
        # check if we have units
        if isinstance(X1, numpy.ndarray):
            X1 = X1 * unit.angstrom

        if disable_vsite:
            self.simulation.context.setPositions(X1)
        else:
            # Create virtual sites before setting positions
            mod = app.Modeller(self.pdb.topology, X1)
            mod.addExtraParticles(self.forcefield)
            self.simulation.context.setPositions(
                ResetVirtualSites_fast(mod.getPositions(), self.vsinfo)
            )

    def optimize(
        self,
        shot,
        crit=1e-4,
        disable_vsite=False,
        align=True,
        include_restraint_energy=False,
    ):

        """
        Optimize the geometry and align the optimized
        geometry to the starting geometry.

        Parameters
        ----------
        shot : int
            The snapshot number to be minimized
        crit : float
            Convergence criterion in kJ/mol
        disable_vsite : bool
            Disable virtual sites (needed for SMIRNOFF)
        include_restraint_energy : bool
            Include energy component from CustomExternalForce

        Returns
        -------
        E : float
            Potential energy of the system
        rmsd : float
            RMSD of the system (w/r.t. starting geometry) in Angstrom
        """

        steps = int(max(1, -1 * numpy.log10(crit)))
        self.update_simulation()
        self.set_positions(shot)
        if self.restraint_frc_index is not None:
            self.set_restraint_positions(shot)
        # Get the previous geometry.
        X0 = (
            self.simulation.context.getState(getPositions=True)
            .getPositions(asNumpy=True)
            .value_in_unit(unit.angstrom)[self.realAtomIdxs]
        )

        for logc in numpy.linspace(0, numpy.log10(crit), steps):
            self.simulation.minimizeEnergy(
                tolerance=10**logc * unit.kilojoule_per_mole, maxIterations=100000
            )
        # check if energy minimization is successful
        # try 1000 times with 10 steps each as openmm minimizer is not very stable at the tolerance
        for _ in range(1000):
            e_minimized = (
                self.simulation.context.getState(getEnergy=True)
                .getPotentialEnergy()
                .value_in_unit(unit.kilojoule_per_mole)
            )
            self.simulation.minimizeEnergy(
                tolerance=crit * unit.kilojoule_per_mole, maxIterations=10
            )
            e_new = (
                self.simulation.context.getState(getEnergy=True)
                .getPotentialEnergy()
                .value_in_unit(unit.kilojoule_per_mole)
            )
            if abs(e_new - e_minimized) < crit * 10:
                break
        else:
            logger.error("Energy minimization did not converge")
            raise RuntimeError("Energy minimization did not converge")
        # Remove the restraint energy from the total energy if desired.
        groups = set(range(32))
        if self.restraint_frc_index is not None and not include_restraint_energy:
            frc = self.simulation.system.getForce(self.restraint_frc_index)
            groups.remove(frc.getForceGroup())

        S = self.simulation.context.getState(
            getPositions=True, getEnergy=True, groups=groups
        )
        # Get the optimized geometry.
        X1 = S.getPositions(asNumpy=True).value_in_unit(unit.angstrom)[
            self.realAtomIdxs
        ]
        E = S.getPotentialEnergy().value_in_unit(unit.kilocalorie_per_mole)
        # Align to original geometry.
        M = deepcopy(self.mol[0])
        M += deepcopy(M)
        M.xyzs = [X0, X1]
        if not self.pbc and align:
            M.align(center=False)
        X1 = M.xyzs[1]

        self._update_positions(X1, disable_vsite)

        return E, M.ref_rmsd(0)[1], M[1]

    def getContextPosition(self, removeVirtual=False):
        """
        Get current position from simulation context.

        Parameters
        ----------
        removeVirtual: bool
            Remove positions of virtual atoms, result will only have positions of real atoms.

        Returns
        -------
        pos: numpy.ndarray of shape (N x 3)
            Position array in unit of Angstrom. If removeVirtual=True, N = No. real atoms, else N = No. all atoms.

        """
        pos = (
            self.simulation.context.getState(getPositions=True)
            .getPositions(asNumpy=True)
            .value_in_unit(unit.angstrom)
        )
        if removeVirtual:
            pos = pos[self.realAtomIdxs]
        return pos

    def multipole_moments(self, shot=0, optimize=True, polarizability=False):

        """Return the multipole moments of the i-th snapshot in Debye and Buckingham units."""

        self.update_simulation()

        if polarizability:
            logger.error("Polarizability calculation is available in TINKER only.\n")
            raise NotImplementedError

        if self.platname in ["CUDA", "OpenCL"] and self.precision in [
            "single",
            "mixed",
        ]:
            crit = 1e-4
        else:
            crit = 1e-6

        if optimize:
            self.optimize(shot, crit=crit)
        else:
            self.set_positions(shot)

        moments = get_multipoles(self.simulation)

        dipole_dict = OrderedDict(zip(["x", "y", "z"], moments[:3]))
        quadrupole_dict = OrderedDict(
            zip(["xx", "xy", "yy", "xz", "yz", "zz"], moments[3:10])
        )

        calc_moments = OrderedDict(
            [("dipole", dipole_dict), ("quadrupole", quadrupole_dict)]
        )

        return calc_moments

    def energy_rmsd(self, shot=0, optimize=True):

        """Calculate energy of the 1st structure (optionally minimize and return the minimized energy and RMSD). In kcal/mol."""

        self.update_simulation()

        if self.platname in ["CUDA", "OpenCL"] and self.precision in [
            "single",
            "mixed",
        ]:
            crit = 1e-4
        else:
            crit = 1e-6

        rmsd = 0.0
        if optimize:
            E, rmsd, _ = self.optimize(shot, crit=crit)
        else:
            self.set_positions(shot)
            E = (
                self.simulation.context.getState(getEnergy=True)
                .getPotentialEnergy()
                .value_in_unit(openmm.unit.kilocalories_per_mole)
            )

        return E, rmsd

    def interaction_energy(self, fraga, fragb):

        """Calculate the interaction energy for two fragments."""

        self.update_simulation()

        if self.name == "A" or self.name == "B":
            logger.error("Don't name the engine A or B!\n")
            raise RuntimeError

        # Create two subengines.
        if hasattr(self, "target"):
            if not hasattr(self, "A"):
                self.A = OpenMM(
                    name="A", mol=self.mol.atom_select(fraga), target=self.target
                )
            if not hasattr(self, "B"):
                self.B = OpenMM(
                    name="B", mol=self.mol.atom_select(fragb), target=self.target
                )
        else:
            if not hasattr(self, "A"):
                self.A = OpenMM(
                    name="A",
                    mol=self.mol.atom_select(fraga),
                    platname=self.platname,
                    precision=self.precision,
                    ffxml=self.ffxml,
                    mmopts=self.mmopts,
                )
            if not hasattr(self, "B"):
                self.B = OpenMM(
                    name="B",
                    mol=self.mol.atom_select(fragb),
                    platname=self.platname,
                    precision=self.precision,
                    ffxml=self.ffxml,
                    mmopts=self.mmopts,
                )

        # Interaction energy needs to be in kcal/mol.
        D = self.energy()
        A = self.A.energy()
        B = self.B.energy()

        return (D - A - B) / 4.184

    def molecular_dynamics(
        self,
        nsteps,
        timestep,
        temperature=None,
        pressure=None,
        nequil=0,
        nsave=1000,
        minimize=True,
        anisotropic=False,
        save_traj=False,
        verbose=False,
        **kwargs,
    ):

        """
        Method for running a molecular dynamics simulation.

        Required arguments:
        nsteps      = (int)   Number of total time steps
        timestep    = (float) Time step in FEMTOSECONDS
        temperature = (float) Temperature control (Kelvin)
        pressure    = (float) Pressure control (atmospheres)
        nequil      = (int)   Number of additional time steps at the beginning for equilibration
        nsave       = (int)   Step interval for saving and printing data
        minimize    = (bool)  Perform an energy minimization prior to dynamics

        Returns simulation data:
        Rhos        = (array)     Density in kilogram m^-3
        Potentials  = (array)     Potential energies
        Kinetics    = (array)     Kinetic energies
        Volumes     = (array)     Box volumes
        Dips        = (3xN array) Dipole moments
        EComps      = (dict)      Energy components
        """

        if float(int(float(nequil) / float(nsave))) != float(nequil) / float(nsave):
            logger.error("Please set nequil to an integer multiple of nsave\n")
            raise RuntimeError
        iequil = int(nequil / nsave)

        if float(int(float(nsteps) / float(nsave))) != float(nsteps) / float(nsave):
            logger.error("Please set nsteps to an integer multiple of nsave\n")
            raise RuntimeError
        isteps = int(nsteps / nsave)
        nsave = int(nsave)

        if hasattr(self, "simulation"):
            logger.warning("Deleting the simulation object and re-creating for MD\n")
            delattr(self, "simulation")

        self.update_simulation(
            timestep=timestep,
            temperature=temperature,
            pressure=pressure,
            anisotropic=anisotropic,
            **kwargs,
        )
        self.set_positions(0)

        # Minimize the energy.
        if minimize:
            if verbose:
                logger.info(
                    "Minimizing the energy... (starting energy % .3f kJ/mol)"
                    % self.simulation.context.getState(getEnergy=True)
                    .getPotentialEnergy()
                    .value_in_unit(openmm.unit.kilojoule_per_mole)
                )
            self.simulation.minimizeEnergy()
            if verbose:
                logger.info(
                    "Done (final energy % .3f kJ/mol)\n"
                    % self.simulation.context.getState(getEnergy=True)
                    .getPotentialEnergy()
                    .value_in_unit(openmm.unit.kilojoule_per_mole)
                )

        # Serialize the system if we want.
        Serialize = 0
        if Serialize:
            serial = openmm.XmlSerializer.serializeSystem(system)
            with wopen("%s_system.xml" % phase) as f:
                f.write(serial)

        # Determine number of degrees of freedom; the center of mass motion remover is also a constraint.
        kB = unit.BOLTZMANN_CONSTANT_kB * unit.AVOGADRO_CONSTANT_NA

        # Compute total mass.
        self.mass = (
            self.compute_mass(self.system).in_units_of(unit.gram / unit.mole)
            / unit.AVOGADRO_CONSTANT_NA
        )

        # Determine number of degrees of freedom.
        self.ndof = (
            3
            * (
                self.system.getNumParticles()
                - sum(
                    [
                        self.system.isVirtualSite(i)
                        for i in range(self.system.getNumParticles())
                    ]
                )
            )
            - self.system.getNumConstraints()
            - 3 * self.pbc
        )

        # Initialize statistics.
        edecomp = OrderedDict()
        # Stored coordinates, box vectors
        self.xyz_omms = []
        # Densities, potential and kinetic energies, box volumes, dipole moments
        Rhos = []
        Potentials = []
        Kinetics = []
        Volumes = []
        Dips = []
        Temps = []
        # ========================#
        # Now run the simulation #
        # ========================#
        # Initialize velocities.
        self.simulation.context.setVelocitiesToTemperature(temperature * unit.kelvin)
        # Equilibrate.
        if iequil > 0:
            if verbose:
                logger.info("Equilibrating...\n")
            if self.pbc:
                if verbose:
                    logger.info(
                        "%6s %9s %9s %13s %10s %13s\n"
                        % (
                            "Iter.",
                            "Time(ps)",
                            "Temp(K)",
                            "Epot(kJ/mol)",
                            "Vol(nm^3)",
                            "Rho(kg/m^3)",
                        )
                    )
            else:
                if verbose:
                    logger.info(
                        "%6s %9s %9s %13s\n"
                        % ("Iter.", "Time(ps)", "Temp(K)", "Epot(kJ/mol)")
                    )
        for iteration in range(-1, iequil):
            if iteration >= 0:
                self.simulation.step(nsave)
            state = self.simulation.context.getState(
                getEnergy=True, getPositions=True, getVelocities=False, getForces=False
            )
            kinetic = state.getKineticEnergy() / self.tdiv
            potential = state.getPotentialEnergy()
            if self.pbc:
                box_vectors = state.getPeriodicBoxVectors()
                volume = self.compute_volume(box_vectors)
                density = (self.mass / volume).in_units_of(
                    unit.kilogram / unit.meter**3
                )
            else:
                volume = 0.0 * unit.nanometers**3
                density = 0.0 * unit.kilogram / unit.meter**3
            kinetic_temperature = (
                2.0 * kinetic / kB / self.ndof
            )  # (1/2) ndof * kB * T = KE
            if self.pbc:
                if verbose:
                    logger.info(
                        "%6d %9.3f %9.3f % 13.3f %10.4f %13.4f\n"
                        % (
                            iteration + 1,
                            state.getTime() / unit.picoseconds,
                            kinetic_temperature / unit.kelvin,
                            potential / unit.kilojoules_per_mole,
                            volume / unit.nanometers**3,
                            density / (unit.kilogram / unit.meter**3),
                        )
                    )
            else:
                if verbose:
                    logger.info(
                        "%6d %9.3f %9.3f % 13.3f\n"
                        % (
                            iteration + 1,
                            state.getTime() / unit.picoseconds,
                            kinetic_temperature / unit.kelvin,
                            potential / unit.kilojoules_per_mole,
                        )
                    )
        # Collect production data.
        if verbose:
            logger.info("Production...\n")
        if self.pbc:
            if verbose:
                logger.info(
                    "%6s %9s %9s %13s %10s %13s\n"
                    % (
                        "Iter.",
                        "Time(ps)",
                        "Temp(K)",
                        "Epot(kJ/mol)",
                        "Vol(nm^3)",
                        "Rho(kg/m^3)",
                    )
                )
        else:
            if verbose:
                logger.info(
                    "%6s %9s %9s %13s\n"
                    % ("Iter.", "Time(ps)", "Temp(K)", "Epot(kJ/mol)")
                )
        if save_traj:
            self.simulation.reporters.append(
                app.PDBReporter("%s-md.pdb" % self.name, nsteps)
            )
            self.simulation.reporters.append(
                app.DCDReporter("%s-md.dcd" % self.name, nsave)
            )

        for iteration in range(-1, isteps):
            # Propagate dynamics.
            if iteration >= 0:
                self.simulation.step(nsave)
            # Compute properties.
            state = self.simulation.context.getState(
                getEnergy=True, getPositions=True, getVelocities=False, getForces=False
            )
            kinetic = state.getKineticEnergy() / self.tdiv
            potential = state.getPotentialEnergy()
            kinetic_temperature = 2.0 * kinetic / kB / self.ndof
            if self.pbc:
                box_vectors = state.getPeriodicBoxVectors()
                volume = self.compute_volume(box_vectors)
                density = (self.mass / volume).in_units_of(
                    unit.kilogram / unit.meter**3
                )
            else:
                box_vectors = None
                volume = 0.0 * unit.nanometers**3
                density = 0.0 * unit.kilogram / unit.meter**3
            positions = (
                state.getPositions(asNumpy=True).astype(numpy.float32) * unit.nanometer
            )
            self.xyz_omms.append([positions, box_vectors])
            # Perform energy decomposition.
            for comp, val in energy_components(self.simulation).items():
                if comp in edecomp:
                    edecomp[comp].append(val)
                else:
                    edecomp[comp] = [val]
            if self.pbc:
                if verbose:
                    logger.info(
                        "%6d %9.3f %9.3f % 13.3f %10.4f %13.4f\n"
                        % (
                            iteration + 1,
                            state.getTime() / unit.picoseconds,
                            kinetic_temperature / unit.kelvin,
                            potential / unit.kilojoules_per_mole,
                            volume / unit.nanometers**3,
                            density / (unit.kilogram / unit.meter**3),
                        )
                    )
            else:
                if verbose:
                    logger.info(
                        "%6d %9.3f %9.3f % 13.3f\n"
                        % (
                            iteration + 1,
                            state.getTime() / unit.picoseconds,
                            kinetic_temperature / unit.kelvin,
                            potential / openmm.unit.kilojoules_per_mole,
                        )
                    )
            Temps.append(kinetic_temperature / unit.kelvin)
            Rhos.append(density.value_in_unit(unit.kilogram / unit.meter**3))
            Potentials.append(potential / unit.kilojoules_per_mole)
            Kinetics.append(kinetic / unit.kilojoules_per_mole)
            Volumes.append(volume / unit.nanometer**3)
            Dips.append(get_dipole(self.simulation, positions=self.xyz_omms[-1][0]))
        Rhos = numpy.array(Rhos)
        Potentials = numpy.array(Potentials)
        Kinetics = numpy.array(Kinetics)
        Volumes = numpy.array(Volumes)
        Dips = numpy.array(Dips)
        Ecomps = OrderedDict([(key, numpy.array(val)) for key, val in edecomp.items()])
        Ecomps["Potential Energy"] = Potentials
        Ecomps["Kinetic Energy"] = Kinetics
        Ecomps["Temperature"] = Temps
        Ecomps["Total Energy"] = Potentials + Kinetics
        # Initialized property dictionary.
        prop_return = OrderedDict()
        prop_return.update(
            {
                "Rhos": Rhos,
                "Potentials": Potentials,
                "Kinetics": Kinetics,
                "Volumes": Volumes,
                "Dips": Dips,
                "Ecomps": Ecomps,
            }
        )
        return prop_return

    def scale_box(self, x=1.0, y=1.0, z=1.0):
        """Scale the positions of molecules and box vectors. Molecular structures will be kept the same.
        Input: x, y, z :scaling factors (float)
        Output: None
        After this function call, self.xyz_omms will be overwritten with the new positions and box_vectors.
        """
        if not hasattr(self, "xyz_omms"):
            logger.error("molecular_dynamics has not finished correctly!")
            raise RuntimeError
        # record the indices of each residue
        if not hasattr(self, "residues_idxs"):
            self.residues_idxs = numpy.array(
                [
                    [a.index for a in r.atoms()]
                    for r in self.simulation.topology.residues()
                ]
            )
        scale_xyz = numpy.array([x, y, z])
        # loop over each frame and replace items
        for i in range(len(self.xyz_omms)):
            pos, box = self.xyz_omms[i]
            # scale the box vectors
            new_box = numpy.array(box / unit.nanometer) * scale_xyz
            # convert pos to numpy.array
            positions = numpy.array(pos / unit.nanometer)
            # Positions of each residue
            residue_positions = numpy.take(positions, self.residues_idxs, axis=0)
            # Center of each residue
            res_center_positions = numpy.mean(residue_positions, axis=1)
            # Shift of each residue
            center_pos_shift = res_center_positions * (scale_xyz - 1)
            # New positions
            new_pos = (
                residue_positions + center_pos_shift[:, numpy.newaxis, :]
            ).reshape(-1, 3)
            # update this frame
            self.xyz_omms[i] = [
                new_pos.astype(numpy.float32) * unit.nanometer,
                new_box * unit.nanometer,
            ]


class Liquid_OpenMM(Liquid):
    """Condensed phase property matching using OpenMM."""

    def __init__(self, options, tgt_opts, forcefield):
        # Time interval (in ps) for writing coordinates
        self.set_option(tgt_opts, "force_cuda", forceprint=True)
        # Enable multiple timestep integrator
        self.set_option(tgt_opts, "mts_integrator", forceprint=True)
        # Enable ring polymer MD
        self.set_option(options, "rpmd_beads", forceprint=True)
        # OpenMM precision
        self.set_option(tgt_opts, "openmm_precision", "precision", default="mixed")
        # OpenMM platform
        self.set_option(tgt_opts, "openmm_platform", "platname", default="CUDA")
        # Name of the liquid coordinate file.
        self.set_option(
            tgt_opts, "liquid_coords", default="liquid.pdb", forceprint=True
        )
        # Name of the gas coordinate file.
        self.set_option(tgt_opts, "gas_coords", default="gas.pdb", forceprint=True)
        # Name of the surface tension coordinate file. (e.g. an elongated box with a film of water)
        self.set_option(tgt_opts, "nvt_coords", default="surf.pdb", forceprint=True)
        # Set the number of steps between MC barostat adjustments.
        self.set_option(tgt_opts, "mc_nbarostat")
        # Class for creating engine object.
        self.engine_ = OpenMM
        # Name of the engine to pass to npt.py.
        self.engname = "openmm"
        # Command prefix.
        self.nptpfx = "bash runcuda.sh"
        if tgt_opts["remote_backup"]:
            self.nptpfx += " -b"
        # Extra files to be linked into the temp-directory.
        self.nptfiles = []
        self.nvtfiles = []
        # Set some options for the polarization correction calculation.
        self.gas_engine_args = {}
        # Scripts to be copied from the ForceBalance installation directory.
        self.scripts = ["runcuda.sh"]
        # Initialize the base class.
        super().__init__(options, tgt_opts, forcefield)
        # Send back the trajectory file.
        if self.save_traj > 0:
            self.extra_output = ["liquid-md.pdb", "liquid-md.dcd"]
        # These functions need to be called after self.nptfiles is populated
        self.post_init(options)


class AbInitio_OpenMM(AbInitio):
    """Force and energy matching using OpenMM."""

    def __init__(self, options, tgt_opts, forcefield):
        # Default file names for coordinates and key file.
        self.set_option(tgt_opts, "pdb", default="conf.pdb")
        self.set_option(tgt_opts, "coords", default="all.gro")
        self.set_option(
            tgt_opts, "openmm_precision", "precision", default="double", forceprint=True
        )
        self.set_option(
            tgt_opts, "openmm_platform", "platname", default="CUDA", forceprint=True
        )
        self.engine_ = OpenMM
        # Initialize base class.
        super().__init__(options, tgt_opts, forcefield)


class BindingEnergy_OpenMM(BindingEnergy):
    """Binding energy matching using OpenMM."""

    def __init__(self, options, tgt_opts, forcefield):
        self.engine_ = OpenMM
        self.set_option(
            tgt_opts, "openmm_precision", "precision", default="double", forceprint=True
        )
        self.set_option(
            tgt_opts,
            "openmm_platform",
            "platname",
            default="Reference",
            forceprint=True,
        )
        # Initialize base class.
        super().__init__(options, tgt_opts, forcefield)


class Interaction_OpenMM(Interaction):
    """Interaction matching using OpenMM."""

    def __init__(self, options, tgt_opts, forcefield):
        # Default file names for coordinates and key file.
        self.set_option(tgt_opts, "coords", default="all.pdb")
        self.set_option(
            tgt_opts, "openmm_precision", "precision", default="double", forceprint=True
        )
        self.set_option(
            tgt_opts,
            "openmm_platform",
            "platname",
            default="Reference",
            forceprint=True,
        )
        self.engine_ = OpenMM
        # Initialize base class.
        super().__init__(options, tgt_opts, forcefield)


class Moments_OpenMM(Moments):
    """Multipole moment matching using OpenMM."""

    def __init__(self, options, tgt_opts, forcefield):
        # Default file names for coordinates and key file.
        self.set_option(tgt_opts, "coords", default="input.pdb")
        self.set_option(
            tgt_opts, "openmm_precision", "precision", default="double", forceprint=True
        )
        self.set_option(
            tgt_opts,
            "openmm_platform",
            "platname",
            default="Reference",
            forceprint=True,
        )
        self.engine_ = OpenMM
        # Initialize base class.
        super().__init__(options, tgt_opts, forcefield)


class Hydration_OpenMM(Hydration):
    """Single point hydration free energies using OpenMM."""

    def __init__(self, options, tgt_opts, forcefield):
        # Default file names for coordinates and key file.
        # self.set_option(tgt_opts,'coords',default="input.pdb")
        self.set_option(
            tgt_opts, "openmm_precision", "precision", default="double", forceprint=True
        )
        self.set_option(
            tgt_opts, "openmm_platform", "platname", default="CUDA", forceprint=True
        )
        self.engine_ = OpenMM
        self.engname = "openmm"
        # Scripts to be copied from the ForceBalance installation directory.
        self.scripts = ["runcuda.sh"]
        # Suffix for coordinate files.
        self.crdsfx = ".pdb"
        # Command prefix.
        self.prefix = "bash runcuda.sh"
        if tgt_opts["remote_backup"]:
            self.prefix += " -b"
        # Initialize base class.
        super().__init__(options, tgt_opts, forcefield)
        # Send back the trajectory file.
        if self.save_traj > 0:
            self.extra_output = ["openmm-md.dcd"]


class Vibration_OpenMM(Vibration):
    """Vibrational frequency matching using TINKER."""

    def __init__(self, options, tgt_opts, forcefield):
        # Default file names for coordinates and key file.
        self.set_option(tgt_opts, "coords", default="input.pdb")
        self.set_option(
            tgt_opts, "openmm_precision", "precision", default="double", forceprint=True
        )
        self.set_option(
            tgt_opts,
            "openmm_platform",
            "platname",
            default="Reference",
            forceprint=True,
        )
        self.engine_ = OpenMM
        # Initialize base class.
        super().__init__(options, tgt_opts, forcefield)


class OptGeoTarget_OpenMM(OptGeoTarget):
    """Optimized geometry matching using OpenMM."""

    def __init__(self, options, tgt_opts, forcefield):
        self.engine_ = OpenMM
        self.set_option(
            tgt_opts, "openmm_precision", "precision", default="double", forceprint=True
        )
        self.set_option(
            tgt_opts,
            "openmm_platform",
            "platname",
            default="Reference",
            forceprint=True,
        )
        # Initialize base class.
        super().__init__(options, tgt_opts, forcefield)

    def create_engines(self, engine_args):
        """create a dictionary of self.engines = {sysname: Engine}"""
        self.engines = OrderedDict()
        for sysname, sysopt in self.sys_opts.items():
            # path to pdb file
            pdbpath = os.path.join(self.root, self.tgtdir, sysopt["topology"])
            # use the PDB file with topology
            M = Molecule(pdbpath)
            # replace geometry with values from xyz file for higher presision
            M0 = Molecule(os.path.join(self.root, self.tgtdir, sysopt["geometry"]))
            M.xyzs = M0.xyzs
            # here mol=M is given for the purpose of using the topology from the input pdb file
            # if we don't do this, pdb=top.pdb option will only copy some basic information but not the topology into OpenMM.mol (openmmio.py line 615)
            self.engines[sysname] = self.engine_(
                target=self, mol=M, name=sysname, pdb=pdbpath, **engine_args
            )


class TorsionProfileTarget_OpenMM(TorsionProfileTarget):
    """Optimized geometry matching using OpenMM."""

    def __init__(self, options, tgt_opts, forcefield):
        # Default file names for coordinates and key file.
        self.set_option(tgt_opts, "pdb", default="conf.pdb")
        self.set_option(tgt_opts, "coords", default="scan.xyz")
        self.set_option(
            tgt_opts, "openmm_precision", "precision", default="double", forceprint=True
        )
        self.set_option(
            tgt_opts,
            "openmm_platform",
            "platname",
            default="Reference",
            forceprint=True,
        )
        self.engine_ = OpenMM
        # Initialize base class.
        super().__init__(options, tgt_opts, forcefield)
