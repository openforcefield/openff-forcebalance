def read_mol2(fnm, **kwargs):
    xyz = []
    charge = []
    atomname = []
    atomtype = []
    elem = []
    resname = []
    resid = []
    data = Mol2.mol2_set(fnm)
    if len(data.compounds) > 1:
        sys.stderr.write(
            "Not sure what to do if the MOL2 file contains multiple compounds\n"
        )
    for i, atom in enumerate(list(data.compounds.items())[0][1].atoms):
        xyz.append([atom.x, atom.y, atom.z])
        charge.append(atom.charge)
        atomname.append(atom.atom_name)
        atomtype.append(atom.atom_type)
        resname.append(atom.subst_name)
        resid.append(atom.subst_id)
        thiselem = atom.atom_name
        if len(thiselem) > 1:
            thiselem = thiselem[0] + re.sub("[A-Z0-9]", "", thiselem[1:])
        elem.append(thiselem)

    bonds = []
    for bond in list(data.compounds.items())[0][1].bonds:
        a1 = bond.origin_atom_id - 1
        a2 = bond.target_atom_id - 1
        aL, aH = (a1, a2) if a1 < a2 else (a2, a1)
        bonds.append((aL, aH))

    return {
        "xyzs": [numpy.array(xyz)],
        "partial_charge": charge,
        "atomname": atomname,
        "atomtype": atomtype,
        "elem": elem,
        "resname": resname,
        "resid": resid,
        "bonds": bonds,
    }
