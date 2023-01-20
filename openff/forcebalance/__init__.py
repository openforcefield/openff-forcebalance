from collections import OrderedDict
from re import findall

from ._version import get_versions
from .parser import gen_opts_defaults, tgt_opts_defaults

versions = get_versions()
__version__ = versions["version"]
__git_revision__ = versions["full-revisionid"]
del get_versions, versions


class BaseClass:
    """Provides some nifty functions that are common to all ForceBalance classes."""

    def __init__(self, options):
        self.option_dict = dict()
        self.option_keys = set()
        self.verbose_options = options.get("verbose_options", False)

    def __setattr__(self, key, value):
        if not hasattr(self, "option_dict"):
            super().__setattr__("option_dict", dict())
        if not hasattr(self, "option_keys"):
            super().__setattr__("option_keys", set())

        ## These attributes return a list of attribute names defined in this class, that belong in the chosen category.
        ## For example: self.FrameKeys should return set(['xyzs','boxes']) if xyzs and boxes exist in self.Data
        if key in self.option_keys:
            self.option_dict[key] = value
        return super().__setattr__(key, value)

    def set_option(
        self, in_dict, src_key, dest_key=None, val=None, default=None, forceprint=False
    ):
        if not hasattr(self, "print_option_dict"):
            self.print_option_dict = OrderedDict()
        if dest_key is None:
            dest_key = src_key
        if val is None:
            if src_key in in_dict and in_dict[src_key] is not None:
                val = in_dict[src_key]
            elif default is not None:
                val = default
            elif src_key in gen_opts_defaults:
                val = gen_opts_defaults[src_key]
            elif src_key in tgt_opts_defaults:
                val = tgt_opts_defaults[src_key]
        if default is None:
            if src_key in gen_opts_defaults:
                default = gen_opts_defaults[src_key]
            elif src_key in tgt_opts_defaults:
                default = tgt_opts_defaults[src_key]
            else:
                default = None
        if (
            (
                val != default
                or (hasattr(self, "verbose_options") and self.verbose_options)
            )
            and dest_key != "root"
        ) or forceprint:
            self.print_option_dict[dest_key] = val
        self.option_keys.add(dest_key)
        return self.__setattr__(dest_key, val)


class BaseReader:
    """The 'reader' class.  It serves two main functions:

    1) When parsing a text force field file, the 'feed' method is
    called once for every line.  Calling the 'feed' method stores the
    internal variables that are needed for making the unique parameter
    identifier.

    2) The 'reader' also stores the 'pdict' dictionary, which is
    needed for building the matrix of rescaling factors.  This is not
    needed for the XML force fields, so in XML force fields pdict is
    replaced with a string called "XML_Override".

    """

    def __init__(self, fnm):
        self.ln = 0
        self.itype = fnm
        self.suffix = ""
        self.pdict = {}
        ## The mapping of (this residue, atom number) to (atom name) for building atom-specific interactions in [ bonds ], [ angles ] etc.
        self.adict = OrderedDict()
        ## The mapping of (molecule name) to a dictionary of  of atom types for the atoms in that residue.
        # self.moleculedict = OrderedDict()
        ## The listing of 'RES:ATOMNAMES' for atom names in the line
        ## This is obviously a placeholder.
        self.molatom = ("Sniffy", ["Mao", "Schmao"])

        self.Molecules = OrderedDict()
        self.AtomTypes = OrderedDict()

    def Split(self, line):
        return line.split()

    def Whites(self, line):
        return findall("[ ]+", line)

    def feed(self, line):
        self.ln += 1

    def build_pid(self, pfld):
        """Returns the parameter type (e.g. K in BONDSK) based on the
        current interaction type.

        Both the 'pdict' dictionary (see gmxio.pdict) and the
        interaction type 'state' (here, BONDS) are needed to get the
        parameter type.

        If, however, 'pdict' does not contain the ptype value, a suitable
        substitute is simply the field number.

        Note that if the interaction type state is not set, then it
        defaults to the file name, so a generic parameter ID is
        'filename.line_num.field_num'

        """
        # print self.pdict[self.itype][pfld]
        if hasattr(self, "overpfx"):
            return self.overpfx + ":%i:" % pfld + self.oversfx
        ptype = self.pdict.get(self.itype, {}).get(pfld, ":%i.%i" % (self.ln, pfld))
        answer = self.itype
        answer += ptype
        answer += self.suffix
        return answer