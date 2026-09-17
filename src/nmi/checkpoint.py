# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

# Purpose: Atomic checkpoints for the samplers of Section 4.4. A checkpoint is
# written to a temporary file and then moved into place with os.replace, so
# that an interrupted job never leaves a partially written file behind. The
# state of the NumPy bit generator is stored so that a resumed chain continues
# the same random stream.
# Manuscript: Section 4.6, computational resources; Experiment 0.3.
# Inputs: sampler state. Outputs: compressed NumPy archives on disk.

import json  # Serialisation of the generator state and of the metadata.
import os  # Atomic replacement of the checkpoint file.
from pathlib import Path  # Portable filesystem paths.

import numpy as np  # Numerical arrays and compressed archives.


# Write a checkpoint atomically.
# Arguments:
#   path (str or pathlib.Path): the checkpoint file to write.
#   arrays (dict): named arrays holding the state of the sampler.
#   state (dict): the serialised state of the NumPy bit generator.
#   metadata (dict): further entries such as the iteration counter.
# Returns:
#   pathlib.Path: the checkpoint that was written.
def write_checkpoint(path, arrays, state, metadata):
    location = Path(path)  # Normalise the argument into a path object.
    location.parent.mkdir(parents=True, exist_ok=True)  # Create the directory when it is absent.
    temporary = location.with_suffix(location.suffix + ".tmp")  # Temporary file in the same directory.
    payload = dict(arrays)  # Copy the named arrays so that the caller is unaffected.
    # Generator state and metadata are stored as raw bytes of a JSON document.
    payload["_generator_state"] = np.frombuffer(json.dumps(state).encode("utf-8"), dtype=np.uint8)
    # Metadata of the run, stored in the same way as the generator state.
    payload["_metadata"] = np.frombuffer(json.dumps(metadata).encode("utf-8"), dtype=np.uint8)
    with temporary.open("wb") as handle:  # Open the temporary file for writing.
        np.savez(handle, **payload)  # Write the whole payload into the temporary file.
        handle.flush()  # Flush the buffered data to the operating system.
        os.fsync(handle.fileno())  # Force the data to reach the storage device.
    os.replace(temporary, location)  # Atomically move the temporary file into place.
    return location  # Return the written checkpoint for convenience.


# Read a checkpoint written by write_checkpoint.
# Arguments:
#   path (str or pathlib.Path): the checkpoint file to read.
# Returns:
#   tuple: the named arrays, the generator state and the metadata.
def read_checkpoint(path):
    location = Path(path)  # Normalise the argument into a path object.
    with np.load(location, allow_pickle=False) as archive:  # Open the compressed archive.
        # Entries whose name starts with an underscore hold serialised metadata.
        arrays = {name: archive[name] for name in archive.files if not name.startswith("_")}
        state = json.loads(bytes(archive["_generator_state"]).decode("utf-8"))  # Generator state.
        metadata = json.loads(bytes(archive["_metadata"]).decode("utf-8"))  # Run metadata.
    return arrays, state, metadata  # Sampler state, generator state and metadata.


# Test whether a checkpoint exists and can be read.
# Arguments:
#   path (str or pathlib.Path): the checkpoint file to test.
# Returns:
#   bool: True when the checkpoint exists and is readable.
def checkpoint_exists(path):
    location = Path(path)  # Normalise the argument into a path object.
    if not location.is_file():  # No checkpoint has been written yet.
        return False  # The caller must start the chain from its initial state.
    try:  # A checkpoint may be truncated if the job was killed during the write.
        read_checkpoint(location)  # Attempt a full read of the checkpoint.
    except (OSError, ValueError, KeyError):  # The checkpoint cannot be interpreted.
        return False  # Treat an unreadable checkpoint as absent.
    return True  # The checkpoint exists and can be resumed from.
