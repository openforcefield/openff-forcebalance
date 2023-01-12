from collections import OrderedDict

import pytest

from openff.forcebalance.gmxio import GMX
from openff.forcebalance.nifty import *
from openff.forcebalance.openmmio import OpenMM

from .__init__ import ForceBalanceTestCase, check_for_openmm

# Set SAVEDATA to True and run the tests in order to save data
# to a file for future reference. This is easier to use for troubleshooting
# vs. comparing multiple programs against each other, b/c we don't know
# which one changed.
SAVEDATA = False


class TestAmber99SB(ForceBalanceTestCase):
    """Amber99SB unit test consisting of ten structures of
    ACE-ALA-NME interacting with ACE-GLU-NME.  The tests check for
    whether the OpenMM, GMX Engines produce consistent
    results for:

    1) Single-point energies and forces over all ten structures
    2) Minimized energies and RMSD from the initial geometry for a selected structure
    3) Interaction energies between the two molecules over all ten structures
    4) Multipole moments of a selected structure
    5) Multipole moments of a selected structure after geometry optimization
    6) Normal modes of a selected structure
    7) Normal modes of a selected structure after geometry optimization

    If the engines are setting up the calculation correctly, then the
    remaining differences between results are due to differences in
    the parameter files or software implementations.

    The criteria in this unit test are more stringent than normal
    simulations.  In order for the software packages to agree to
    within the criteria, I had to do the following:

    - Remove improper dihedrals from the force field, because there is
    an ambiguity in the atom ordering which leads to force differences
    - Increase the number of decimal points in the "fudgeQQ" parameter
    in the GROMACS .itp file
    - Increase two torsional barriers to ensure optimizer converges
    to the same local minimum consistently
    - Compile GROMACS in double precision

    Residual errors are as follows:
    Potential energies: <0.01 kJ/mol (<1e-4 fractional error)
    Forces: <0.1 kJ/mol/nm (<1e-3 fractional error)
    Energy of optimized geometry: < 0.001 kcal/mol
    RMSD from starting structure: < 0.001 Angstrom
    Interaction energies: < 0.0001 kcal/mol
    Multipole moments: < 0.001 Debye / Debye Angstrom
    Multipole moments (optimized): < 0.01 Debye / Debye Angstrom
    Vibrational frequencies: < 0.5 wavenumber (~ 1e-4 fractional error)
    Vibrational eigenvectors: < 0.05 (on 11/2019, updated these)
    """

    @classmethod
    def setup_class(cls):
        """
        setup any state specific to the execution of the given class (which usually contains tests).
        """
        super().setup_class()
        which("testgrad")
        # try to find mdrun_d or gmx_d
        # gmx should be built with config -DGMX_DOUBLE=ON
        gmxpath = which("mdrun_d") or which("gmx_d")
        gmxsuffix = "_d"
        # Tests will FAIL if use single precision gromacs
        # gmxpath = which('mdrun') or which('gmx')
        # gmxsuffix = ''
        # self.logger.debug("\nBuilding options for target...\n")
        cls.cwd = os.path.dirname(os.path.realpath(__file__))
        os.chdir(os.path.join(cls.cwd, "files", "amber_alaglu"))
        cls.tmpfolder = os.path.join(cls.cwd, "files", "amber_alaglu", "temp")
        if not os.path.exists(cls.tmpfolder):
            os.makedirs(cls.tmpfolder)
        os.chdir(cls.tmpfolder)
        for i in [
            "topol.top",
            "shot.mdp",
            "a99sb.xml",
            "a99sb.prm",
            "all.gro",
            "all.arc",
            "AceGluNme.itp",
            "AceAlaNme.itp",
            "a99sb.itp",
        ]:
            os.system("ln -fs ../%s" % i)
        cls.engines = OrderedDict()
        # Set up GMX engine
        if gmxpath != "":
            cls.engines["GMX"] = GMX(
                coords="all.gro",
                gmx_top="topol.top",
                gmx_mdp="shot.mdp",
                gmxpath=gmxpath,
                gmxsuffix=gmxsuffix,
            )
        else:
            logger.warning("GROMACS cannot be found, skipping GMX tests.")
        # Set up OpenMM engine
        try:
            try:
                import openmm
            except ImportError:
                import simtk.openmm
            cls.engines["OpenMM"] = OpenMM(
                coords="all.gro",
                pdb="conf.pdb",
                ffxml="a99sb.xml",
                platname="Reference",
                precision="double",
            )
        except:
            logger.warning("OpenMM cannot be imported, skipping OpenMM tests.")

    @classmethod
    def teardown_class(cls):
        """
        teardown any state that was previously setup with a call to setup_class.
        """
        os.chdir(cls.cwd)
        # shutil.rmtree(cls.cwd, "files", "amber_alaglu", "temp")

    def setup_method(self):
        os.chdir(self.tmpfolder)

    def test_energy_force(self):
        """Test GMX, OpenMM, energy and forces using AMBER force field"""
        printcool("Test GMX, OpenMM, energy and forces using AMBER force field")
        missing_pkgs = []
        for eng in ["GMX", "OpenMM"]:
            if eng not in self.engines:
                missing_pkgs.append(eng)
        if len(missing_pkgs) > 0:
            pytest.skip("Missing packages: %s" % ", ".join(missing_pkgs))
        Data = OrderedDict()
        for name, eng in self.engines.items():
            Data[name] = eng.energy_force()
        datadir = os.path.join(
            self.cwd, "files", "test_engine", self.__class__.__name__
        )
        if SAVEDATA:
            fout = os.path.join(datadir, "test_energy_force.dat")
            if not os.path.exists(os.path.dirname(fout)):
                os.makedirs(os.path.dirname(fout))
            np.savetxt(fout, Data[list(self.engines.keys())[0]])
        fin = os.path.join(datadir, "test_energy_force.dat")
        RefData = np.loadtxt(fin)
        for n1 in self.engines.keys():
            np.testing.assert_allclose(
                Data[n1][:, 0],
                RefData[:, 0],
                rtol=0,
                atol=0.01,
                err_msg="%s energies do not match the reference" % (n1),
            )
            np.testing.assert_allclose(
                Data[n1][:, 1:].flatten(),
                RefData[:, 1:].flatten(),
                rtol=0,
                atol=0.1,
                err_msg="%s forces do not match the reference" % (n1),
            )

    def test_optimized_geometries(self):
        """Test GMX, OpenMM, optimized geometries and RMSD using AMBER force field"""
        printcool(
            "Test GMX, OpenMM, optimized geometries and RMSD using AMBER force field"
        )
        missing_pkgs = []
        for eng in ["GMX", "OpenMM"]:
            if eng not in self.engines:
                missing_pkgs.append(eng)
        if len(missing_pkgs) > 0:
            pytest.skip("Missing packages: %s" % ", ".join(missing_pkgs))
        Data = OrderedDict()
        for name, eng in self.engines.items():
            Data[name] = eng.energy_rmsd(5)
        datadir = os.path.join(
            self.cwd, "files", "test_engine", self.__class__.__name__
        )
        if SAVEDATA:
            fout = os.path.join(datadir, "test_optimized_geometries.dat")
            if not os.path.exists(os.path.dirname(fout)):
                os.makedirs(os.path.dirname(fout))
            np.savetxt(fout, Data[list(self.engines.keys())[0]])
        fin = os.path.join(datadir, "test_optimized_geometries.dat")
        RefData = np.loadtxt(fin)
        for n1 in self.engines.keys():
            print("%s vs Reference energies:" % n1, Data[n1][0], RefData[0])
        for n1 in self.engines.keys():
            np.testing.assert_allclose(
                Data[n1][0],
                RefData[0],
                rtol=0,
                atol=0.001,
                err_msg="%s optimized energies do not match the reference" % n1,
            )
            np.testing.assert_allclose(
                Data[n1][1],
                RefData[1],
                rtol=0,
                atol=0.001,
                err_msg="%s RMSD from starting structure do not match the reference"
                % n1,
            )

    def test_interaction_energies(self):
        """Test GMX, OpenMM, interaction energies using AMBER force field"""
        printcool("Test GMX, OpenMM, interaction energies using AMBER force field")
        missing_pkgs = []
        for eng in ["GMX", "OpenMM"]:
            if eng not in self.engines:
                missing_pkgs.append(eng)
        if len(missing_pkgs) > 0:
            pytest.skip("Missing packages: %s" % ", ".join(missing_pkgs))
        Data = OrderedDict()
        for name, eng in self.engines.items():
            Data[name] = eng.interaction_energy(
                fraga=list(range(22)), fragb=list(range(22, 49))
            )
        datadir = os.path.join(
            self.cwd, "files", "test_engine", self.__class__.__name__
        )
        if SAVEDATA:
            fout = os.path.join(datadir, "test_interaction_energies.dat")
            if not os.path.exists(os.path.dirname(fout)):
                os.makedirs(os.path.dirname(fout))
            np.savetxt(fout, Data[list(self.engines.keys())[0]])
        fin = os.path.join(datadir, "test_interaction_energies.dat")
        RefData = np.loadtxt(fin)
        for n1 in self.engines.keys():
            np.testing.assert_allclose(
                Data[n1],
                RefData,
                rtol=0,
                atol=0.0001,
                err_msg="%s interaction energies do not match the reference" % n1,
            )

    def test_multipole_moments(self):
        """Test GMX, OpenMM, multipole moments using AMBER force field"""
        printcool("Test GMX, OpenMM, multipole moments using AMBER force field")
        missing_pkgs = []
        for eng in ["GMX", "OpenMM"]:
            if eng not in self.engines:
                missing_pkgs.append(eng)
        if len(missing_pkgs) > 0:
            pytest.skip("Missing packages: %s" % ", ".join(missing_pkgs))
        Data = OrderedDict()
        for name, eng in self.engines.items():
            Data[name] = eng.multipole_moments(shot=5, optimize=False)
        datadir = os.path.join(
            self.cwd, "files", "test_engine", self.__class__.__name__
        )
        if SAVEDATA:
            fout = os.path.join(datadir, "test_multipole_moments.dipole.dat")
            if not os.path.exists(os.path.dirname(fout)):
                os.makedirs(os.path.dirname(fout))
            np.savetxt(
                fout,
                np.array(list(Data[list(self.engines.keys())[0]]["dipole"].values())),
            )
            fout = os.path.join(datadir, "test_multipole_moments.quadrupole.dat")
            np.savetxt(
                fout,
                np.array(
                    list(Data[list(self.engines.keys())[0]]["quadrupole"].values())
                ),
            )
        RefDip = np.loadtxt(os.path.join(datadir, "test_multipole_moments.dipole.dat"))
        RefQuad = np.loadtxt(
            os.path.join(datadir, "test_multipole_moments.quadrupole.dat")
        )
        for n1 in self.engines.keys():
            d1 = np.array(list(Data[n1]["dipole"].values()))
            q1 = np.array(list(Data[n1]["quadrupole"].values()))
            np.testing.assert_allclose(
                d1,
                RefDip,
                rtol=0,
                atol=0.001,
                err_msg="%s dipole moments do not match the reference" % n1,
            )
            np.testing.assert_allclose(
                q1,
                RefQuad,
                rtol=0,
                atol=0.001,
                err_msg="%s quadrupole moments do not match the reference" % n1,
            )

    def test_multipole_moments_optimized(self):
        """Test GMX, OpenMM, multipole moments at optimized geometries"""
        # ==================================================#
        # | Geometry-optimized multipole moments; requires |#
        # | double precision in order to pass!             |#
        # ==================================================#
        printcool("Test GMX, OpenMM, multipole moments at optimized geometries")
        missing_pkgs = []
        for eng in ["GMX", "OpenMM"]:
            if eng not in self.engines:
                missing_pkgs.append(eng)
        if len(missing_pkgs) > 0:
            pytest.skip("Missing packages: %s" % ", ".join(missing_pkgs))
        Data = OrderedDict()
        for name, eng in self.engines.items():
            Data[name] = eng.multipole_moments(shot=5, optimize=True)
        datadir = os.path.join(
            self.cwd, "files", "test_engine", self.__class__.__name__
        )
        if SAVEDATA:
            fout = os.path.join(datadir, "test_multipole_moments_optimized.dipole.dat")
            if not os.path.exists(os.path.dirname(fout)):
                os.makedirs(os.path.dirname(fout))
            np.savetxt(
                fout,
                np.array(list(Data[list(self.engines.keys())[0]]["dipole"].values())),
            )
            fout = os.path.join(
                datadir, "test_multipole_moments_optimized.quadrupole.dat"
            )
            np.savetxt(
                fout,
                np.array(
                    list(Data[list(self.engines.keys())[0]]["quadrupole"].values())
                ),
            )
        RefDip = np.loadtxt(
            os.path.join(datadir, "test_multipole_moments_optimized.dipole.dat")
        )
        RefQuad = np.loadtxt(
            os.path.join(datadir, "test_multipole_moments_optimized.quadrupole.dat")
        )
        for n1 in self.engines.keys():
            d1 = np.array(list(Data[n1]["dipole"].values()))
            q1 = np.array(list(Data[n1]["quadrupole"].values()))
            np.testing.assert_allclose(
                d1,
                RefDip,
                rtol=0,
                atol=0.02,
                err_msg="%s dipole moments at optimized geometry do not match the reference"
                % n1,
            )
            np.testing.assert_allclose(
                q1,
                RefQuad,
                rtol=0,
                atol=0.02,
                err_msg="%s quadrupole moments at optimized geometry do not match the reference"
                % n1,
            )

    def test_normal_modes(self):
        """Test GMX and OpenMM normal modes"""
        printcool("Test GMX, OpenMM normal modes")
        missing_pkgs = []
        for eng in ["GMX", "OpenMM"]:
            if eng not in self.engines:
                missing_pkgs.append(eng)
        if len(missing_pkgs) > 0:
            pytest.skip("Missing packages: %s" % ", ".join(missing_pkgs))
        FreqG, ModeG = self.engines["GMX"].normal_modes(shot=5, optimize=False)
        FreqO, ModeO = self.engines["OpenMM"].normal_modes(shot=5, optimize=False)
        datadir = os.path.join(
            self.cwd, "files", "test_engine", self.__class__.__name__
        )
        if SAVEDATA:
            fout = os.path.join(datadir, "test_normal_modes.freq.dat")
            if not os.path.exists(os.path.dirname(fout)):
                os.makedirs(os.path.dirname(fout))
            np.savetxt(fout, FreqT)
            fout = os.path.join(datadir, "test_normal_modes.mode.dat.npy")
            # Need to save as binary data since it's a multidimensional array
            np.save(fout, ModeT)
        FreqRef = np.loadtxt(os.path.join(datadir, "test_normal_modes.freq.dat"))
        ModeRef = np.load(os.path.join(datadir, "test_normal_modes.mode.dat.npy"))
        for Freq, Mode, Name in [(FreqG, ModeG, "GMX"), (FreqO, ModeO, "OpenMM")]:
            iv = -1
            for v, vr, m, mr in zip(Freq, FreqRef, Mode, ModeRef):
                iv += 1
                # Count vibrational modes. Stochastic issue seems to occur for a mode within the lowest 3.
                if vr < 0:
                    continue  # or iv < 3: continue
                # Frequency tolerance is half a wavenumber.
                np.testing.assert_allclose(
                    v,
                    vr,
                    rtol=0,
                    atol=0.5,
                    err_msg="%s vibrational frequencies do not match the reference"
                    % Name,
                )
                delta = 0.05
                for a in range(len(m)):
                    try:
                        np.testing.assert_allclose(
                            m[a],
                            mr[a],
                            rtol=0,
                            atol=delta,
                            err_msg="%s normal modes do not match the reference" % Name,
                        )
                    except:
                        np.testing.assert_allclose(
                            m[a],
                            -1.0 * mr[a],
                            rtol=0,
                            atol=delta,
                            err_msg="%s normal modes do not match the reference" % Name,
                        )

    def test_normal_modes_optimized(self):
        """Test GMX and OpenMM normal modes at optimized geometry"""
        printcool("Test GMX, OpenMM normal modes at optimized geometry")
        missing_pkgs = []
        for eng in ["GMX", "OpenMM"]:
            if eng not in self.engines:
                missing_pkgs.append(eng)
        if len(missing_pkgs) > 0:
            pytest.skip("Missing packages: %s" % ", ".join(missing_pkgs))
        FreqG, ModeG = self.engines["GMX"].normal_modes(shot=5, optimize=True)
        FreqO, ModeO = self.engines["OpenMM"].normal_modes(shot=5, optimize=True)
        datadir = os.path.join(
            self.cwd, "files", "test_engine", self.__class__.__name__
        )
        if SAVEDATA:
            fout = os.path.join(datadir, "test_normal_modes_optimized.freq.dat")
            if not os.path.exists(os.path.dirname(fout)):
                os.makedirs(os.path.dirname(fout))
            np.savetxt(fout, FreqT)
            fout = os.path.join(datadir, "test_normal_modes_optimized.mode.dat")
            # Need to save as binary data since it's a multidimensional array
            np.save(fout, ModeT)
        FreqRef = np.loadtxt(
            os.path.join(datadir, "test_normal_modes_optimized.freq.dat")
        )
        ModeRef = np.load(
            os.path.join(datadir, "test_normal_modes_optimized.mode.dat.npy")
        )
        for Freq, Mode, Name in [(FreqG, ModeG, "GMX"), (FreqO, ModeO, "OpenMM")]:
            iv = -1
            for v, vr, m, mr in zip(Freq, FreqRef, Mode, ModeRef):
                iv += 1
                # Count vibrational modes. Stochastic issue seems to occur for a mode within the lowest 3.
                if vr < 0:
                    continue  # or iv < 3: continue
                # Frequency tolerance is half a wavenumber.
                np.testing.assert_allclose(
                    v,
                    vr,
                    rtol=0,
                    atol=0.5,
                    err_msg="%s vibrational frequencies do not match the reference"
                    % Name,
                )
                delta = 0.05
                for a in range(len(m)):
                    try:
                        np.testing.assert_allclose(
                            m[a],
                            mr[a],
                            rtol=0,
                            atol=delta,
                            err_msg="%s normal modes do not match the reference" % Name,
                        )
                    except:
                        np.testing.assert_allclose(
                            m[a],
                            -1.0 * mr[a],
                            rtol=0,
                            atol=delta,
                            err_msg="%s normal modes do not match the reference" % Name,
                        )
