# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

# Purpose: Deterministic seeding of every experiment. Per-condition seeds are
# spawned from one global seed with numpy.random.SeedSequence, and the state of
# a bit generator can be serialised so that a sampler resumes exactly.
# Manuscript: Section 4.2, synthetic data design.
# Inputs: a global seed and condition indices. Outputs: seed sequences,
# generators and serialisable generator states.

import numpy as np  # Random number generation and seed sequence handling.


# Spawn one independent seed sequence per condition, in index order.
# Arguments:
#   global_seed (int): the seed recorded in the configuration file.
#   n_conditions (int): number of conditions of the experimental design.
# Returns:
#   list: one numpy.random.SeedSequence per condition, in index order.
def spawn_seed_sequences(global_seed, n_conditions):
    root = np.random.SeedSequence(int(global_seed))  # Root sequence of the whole experiment.
    return list(root.spawn(int(n_conditions)))  # Independent child sequences, in index order.


# Build a generator from a seed sequence.
# Arguments:
#   sequence (numpy.random.SeedSequence): the seed sequence of one condition.
# Returns:
#   numpy.random.Generator: a generator seeded by that sequence.
def generator_from_sequence(sequence):
    return np.random.Generator(np.random.PCG64(sequence))  # Reproducible bit generator.


# Compact integer label of a seed sequence, recorded in the output metadata.
# Arguments:
#   sequence (numpy.random.SeedSequence): the seed sequence of one condition.
# Returns:
#   int: the first entry of the generated entropy of the sequence.
def sequence_label(sequence):
    state = sequence.generate_state(1, dtype=np.uint64)  # One word of generated entropy.
    return int(state[0])  # Integer label that identifies the sequence in the output.


# Serialise the state of a generator so that a run can be resumed exactly.
# Arguments:
#   generator (numpy.random.Generator): the generator whose state is stored.
# Returns:
#   dict: a mapping that can be written to a checkpoint file.
def serialise_generator(generator):
    state = generator.bit_generator.state  # Internal state of the bit generator.
    # Assemble the fields needed to rebuild the generator exactly.
    return {
        "bit_generator": state["bit_generator"],  # Name of the bit generator class.
        "state": int(state["state"]["state"]),  # Counter of the permuted congruential generator.
        "inc": int(state["state"]["inc"]),  # Increment of the permuted congruential generator.
        "has_uint32": int(state["has_uint32"]),  # Whether a cached word is available.
        "uinteger": int(state["uinteger"]),  # The cached word itself.
    }


# Restore a generator from a serialised state.
# Arguments:
#   record (dict): a mapping produced by serialise_generator.
# Returns:
#   numpy.random.Generator: a generator in exactly the recorded state.
def deserialise_generator(record):
    bit_generator = np.random.PCG64()  # Fresh bit generator whose state is overwritten.
    # Overwrite the internal state with the values recorded in the checkpoint.
    bit_generator.state = {
        "bit_generator": record["bit_generator"],  # Name of the bit generator class.
        "state": {"state": int(record["state"]), "inc": int(record["inc"])},  # Counter and increment.
        "has_uint32": int(record["has_uint32"]),  # Whether a cached word is available.
        "uinteger": int(record["uinteger"]),  # The cached word itself.
    }
    return np.random.Generator(bit_generator)  # Generator that continues the recorded stream.
