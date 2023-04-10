import numpy as np
import seqfold

def order_of_mag(a):
    """Calculate the order of magnitude of integer elements in an array.

    The order of magnitude represents the number of nucleotides in a
    nucleic acid strand.

    Parameters
    ----------
    a : np.ndarray
        Array of integer nucleic acid sequences

    Returns
    -------
    order : np.ndarray
        Array of integers representing the length of each nucleic acid strand
    """

    order = np.vectorize(len)(np.vectorize(str)(a))

    return order

def num_nucleo(nucleotide_list):
    """Calculate the total number of nucleotides in an array.

    Can be used to check that we are not accidentally deleting any
    nucleotide when we break bonds.

    Parameters
    ----------
    nucleotide_list : np.ndarray
        Array containing all nucleic acid integer sequences

    Returns
    -------
    summ : int
        Total number of nucleotides in all strands
    """

    summ = 0
    for el in nucleotide_list:
        summ += len(str(el))
    return summ

def break_short(short, cleav_prop):
    """Randomly break bonds in short nucleic acid sequences.

    Short strands have fewer nucleotides than a given threshold.

    Parameters
    ----------
    short : np.ndarray
        Array containing all short nucleic acid integer sequences 
    cleave_prop : float
        The probability of breaking each bond

    Returns
    -------
    new_short : np.ndarray
        Array containing all broken and intact nucleic acid strands
    """

    # Calculate the number of nucleotides in each strand
    order = order_of_mag(short).astype(object)
    # Calculate the total number of bonds 
    num_bonds = np.sum(order - 1)
    # Generate an array of random numbers between 0 and 1 (length: #bonds)
    # and check whether each element is lower than a given threshold.
    # If it is lower, then we will break the bond
    cleave = np.random.random(num_bonds) < cleav_prop
    new_short = []
    i = 0 
    # Loop over all strands in the array
    for ns, seq in enumerate(short):
        part = seq 
        n_bond = 1 
        # Loop over all bonds in the strand
        for no in range(1, order[ns]):
            i += 1

            # If bond is not broken, but this is last sub-string of the
            # strand, then save
            if (not cleave[i-1]) and (no == order[ns]-1):
                new_short.append(part)
                continue

            # If bond is not broken, continue
            if not cleave[i-1]:
                n_bond += 1
                continue

            # Break bond. We represent breaking bonds by integer division
            # and modulus operation. This will separate an integer number
            # to two parts.
            part0 = part%10**n_bond
            part = part//10**n_bond
            n_bond = 1 

            # Append one part of the strand, and keep the other part
            # whose remaining bonds may be broken
            new_short.append(part0)

            # Check if the remaining part has no remaining bond, then append
            if no == order[ns]-1:
                new_short.append(part)
    
    return new_short


def structured_regions(structs):
    """Predict structured regions in a nucleic acid strand using seqfold 

    Identifies stacked base pairs, hairpins, and bulges in the sequence.

    Parameters
    ----------
    structs : list of seqfold.fold.Struct objects

    Returns
    -------
    struct_bonds : np.ndarray
        indices of bonds in structured regions
    """

    struct_bonds = []

    # Loop over all structured regions in a given strand
    for struct in structs:
        i, j = struct.ij[0][0], struct.ij[0][1]
        # If we have stacked base pairs, include indices of the bonds
        # between the stacked nucleobases
        if "STACK" in struct.desc:
            length = len(struct.desc.split(":")[1].split("/")[0])
            for bond in range(i, i+length-1):
                struct_bonds.append(bond)
            for bond in range(j-length+1, j):
                struct_bonds.append(bond) 

        # If we have hairpins or bulges, include indices of all the
        # bonds between the beginning and end of the structured region
        elif ("HAIRPIN" in struct.desc) or ("BULGE" in struct.desc):
            for bond in range(i, j):
                struct_bonds.append(bond)

            
    struct_bonds = np.array(sorted(struct_bonds), dtype=np.int64)

    return struct_bonds


def break_long(long, cleav_prop, cleav_prop_struct, mapping):
    """Randomly break bonds in long nucleic acid sequences.

    Long strands have more nucleotides than a given threshold.
    For long strands, we first identify structured regions in the strand.
    If a bond is in a structured region, we break it with probability
    `cleave_prop_struct`. If the bond is in unstructured region,
    we break it with probability `cleav_prop`, which is equal to
    the probability of breaking short strands.

    Parameters
    ----------
    long : np.ndarray
        Array containing all long nucleic acid integer sequences 
    cleave_prop : float
        The probability of breaking each bond in the unstructured region
    cleave_prop_struct : float
        The probability of breaking each bond in the structured region
    mappint : dict
        Map integer numbers (1, 2, 3, 4) with nucleobase names (A, G, C, U)

    Returns
    -------
    new_long : np.ndarray
        Array containing all broken and intact nucleic acid strands
    """

    if long.size == 0:
        return long
    
    new_longs = []

    # Convert integer sequences to base names
    long_s = convert_int_to_str_seq(long, mapping)

    # Loop over all long strands
    for seq, int_seq in zip(long_s, long):
        # Identify structured regions
        structs = seqfold.fold(seq)
        struct_bonds = structured_regions(structs)

        # Calculate the length of the strand and the number of bonds
        order = len(seq)
        num_bonds = order - 1

        # Determine broken bonds for structured and unstructured regions
        cleave = np.random.random(num_bonds) < cleav_prop
        cleave[struct_bonds] = np.random.random(struct_bonds.size) < cleav_prop_struct

        new_long = []

        i = 0
        part = int_seq
        n_bond = 1

        # Loop over all bonds in the strand
        for no in range(1, order):
            i += 1

            # If bond is not broken, but this is last sub-string of the
            # strand, then save
            if (not cleave[i-1]) and (no == order-1):
                new_long.append(part)
                continue

            # If bond is not broken, continue
            if not cleave[i-1]:
                n_bond += 1
                continue

            # Break bond. We represent breaking bonds by integer division
            # and modulus operation. This will separate an integer number
            # to two parts.
            part0 = part%10**n_bond
            part = part//10**n_bond
            n_bond = 1

            # Append one part of the strand, and keep the other part
            # whose remaining bonds may be broken
            new_long.append(part0)

            # Check if the remaining part has no remaining bond, then append
            if (no == order-1):
                new_long.append(part)

        new_longs.extend(new_long)

    new_longs = np.array(new_longs, dtype=object)
                
    return new_longs

def convert_int_to_str_seq(seq_array, mapping):
    """Convert integer nucleic acid seqences to strings with standard names.

    Parameters
    ----------
    seq_array : np.ndarray
        Array of integer nucleic acid strands
    mapping : dict
        Map integer numbers (1, 2, 3, 4) with nucleobase names (A, G, C, U)
    """
    seq_array_s = seq_array.astype(str)

    # Replace each number with the corresponding base name
    for k, v in mapping.items():
        seq_array_s = np.char.replace(seq_array_s, k, v)

    return seq_array_s
