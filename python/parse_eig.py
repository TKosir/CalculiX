"""Utility for parsing CalculiX frequency-step ``.eig`` files and validating
results against reference data.

Format overview
---------------
For a standard (non-cyclic, Hermitian) frequency extraction performed by the
ARPACK driver, CalculiX writes the following binary records (using the C
``ITG`` integer type and ``double`` precision floats)【F:src/arpack.c†L921-L995】:

1. ``ITG`` cyclic-symmetry flag (``0`` for the common non-cyclic case).
2. ``ITG`` Hermitian indicator (``1`` for real-valued modal problems).
3. ``ITG`` perturbation flag ``iperturb[0]`` (``0`` or ``1``).
4. Optional ``double`` block of reference displacements of length
   ``mt * nk`` if the perturbation flag equals ``1``.
5. ``ITG`` number of stored eigenpairs ``nev``.
6. ``nev`` ``double`` values containing the generalized eigenvalues of the
   ``[K] u = λ [M] u`` problem (i.e. angular frequency squared).  Downstream
   routines take the square root to obtain angular frequencies.【F:src/dyna.c†L375-L397】
7. ``neq[1]`` ``double`` entries with the diagonal of the stiffness matrix.
8. ``nzs[2]`` ``double`` entries with the upper triangular off-diagonals of the
   stiffness matrix.
9. ``neq[1]`` ``double`` entries with the diagonal of the mass matrix.
10. ``nzs[1]`` ``double`` entries with the upper triangular off-diagonals of the
    mass matrix.
11. ``nev`` eigenvectors, each stored as ``neq[1]`` consecutive ``double``
    values (real part only for non-cyclic analyses).

Cyclic-symmetry and complex-frequency analyses append extra metadata (nodal
diameters, complex-valued eigenpairs, orthogonality matrices, etc.) that are not
handled by this parser; those variants emit different record layouts in
``complexfreq.c`` and ``arpackcs.c``.【F:src/complexfreq.c†L970-L1186】【F:src/arpackcs.c†L1066-L1148】

This module detects integer size and endianness, reconstructs the data layout,
and exposes :func:`read_eig` for programmatic use together with a CLI that can
validate parsed eigenpairs against reference NumPy arrays using numerical error
checks and MAC analysis.
"""

from __future__ import annotations

import argparse
import json
import math
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Tuple

import numpy as np

try:  # Optional dependency for Hungarian assignment
    from scipy.optimize import linear_sum_assignment  # type: ignore
except Exception:  # pragma: no cover - SciPy may be unavailable
    linear_sum_assignment = None  # type: ignore


@dataclass
class _FormatSpec:
    """Binary layout information inferred from the file header."""

    endian: str  # '<' or '>'
    int_size: int  # in bytes
    float_size: int  # in bytes (always 8 for CalculiX .eig files)

    @property
    def int_fmt(self) -> str:
        return self.endian + ("q" if self.int_size == 8 else "i")

    @property
    def float_fmt(self) -> str:
        return self.endian + "d"

    @property
    def float_dtype(self) -> np.dtype:
        return np.dtype(self.endian + "f8")


def _infer_precision_and_endianness(data: bytes) -> _FormatSpec:
    """Infer integer width and endianness from the leading records.

    The first three integers correspond to boolean-like flags (cyclic symmetry,
    Hermitian, perturbation). We test 32-bit and 64-bit integer layouts under
    both little and big endian conventions and select the combination that
    yields plausible flag values. If multiple candidates remain, preference is
    given to little-endian storage (CalculiX defaults on Linux)."""

    # Ensure we have enough bytes for initial detection.
    if len(data) < 24:
        raise ValueError("File too small to contain .eig header")

    candidates: list[_FormatSpec] = []
    head = memoryview(data)
    for int_size in (4, 8):
        if len(head) < int_size * 4:
            continue
        for endian in ("<", ">"):
            fmt_char = "q" if int_size == 8 else "i"
            try:
                flags = struct.unpack_from(endian + "3" + fmt_char, head, 0)
            except struct.error:
                continue
            cyclic, nherm, iperturb = flags
            if cyclic not in (0, 1):
                continue
            if nherm not in (0, 1):
                continue
            if iperturb not in (0, 1):
                continue
            # Read the next integer for additional validation.
            try:
                nev = struct.unpack_from(endian + fmt_char, head, int_size * 3)[0]
            except struct.error:
                continue
            if nev <= 0 or nev > 10_000_000:
                # nev might not be present yet when iperturb=1. We'll accept
                # large values for now and disambiguate later.
                pass
            candidates.append(_FormatSpec(endian, int_size, 8))

    if not candidates:
        raise ValueError("Unable to determine integer size/endianness from header")

    # Prefer little-endian if multiple candidates remain.
    candidates.sort(key=lambda spec: (spec.endian != "<", spec.int_size))
    return candidates[0]


def _unpack_int(data: memoryview, offset: int, spec: _FormatSpec) -> Tuple[int, int]:
    value = struct.unpack_from(spec.int_fmt, data, offset)[0]
    return value, offset + spec.int_size


def _try_parse_body(
    data: bytes,
    float_offset: int,
    nev: int,
    spec: _FormatSpec,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    """Attempt to parse the floating-point payload for a given eigenvalue count.

    Returns the eigenvalues, eigenvectors, stiffness diagonal, mass diagonal,
    stiffness off-diagonals, mass off-diagonals, and metadata describing the
    inferred block sizes. Raises ``ValueError`` if the layout is inconsistent."""

    if nev <= 0:
        raise ValueError("Eigenvalue count must be positive")

    floats = np.frombuffer(data, dtype=spec.float_dtype, offset=float_offset)
    if floats.size < nev:
        raise ValueError("File truncated before eigenvalue block")

    eigenvalues = np.array(floats[:nev], copy=True)
    remainder = floats[nev:]
    total = remainder.size
    if total == 0:
        raise ValueError("Missing stiffness/mass/eigenvector data")

    ndof, nstiff_off, nmass_off = _infer_matrix_dimensions(remainder, nev)
    if ndof is None:
        raise ValueError("Could not infer matrix dimensions from payload")

    # Slice the remainder according to the inferred sizes.
    cursor = 0
    ad = np.array(remainder[cursor : cursor + ndof], copy=True)
    cursor += ndof
    au = np.array(remainder[cursor : cursor + nstiff_off], copy=True)
    cursor += nstiff_off
    adb = np.array(remainder[cursor : cursor + ndof], copy=True)
    cursor += ndof
    aub = np.array(remainder[cursor : cursor + nmass_off], copy=True)
    cursor += nmass_off

    eigenvector_count = ndof * nev
    if remainder.size < cursor + eigenvector_count:
        raise ValueError("Insufficient data for eigenvectors")

    eigvec_flat = remainder[cursor : cursor + eigenvector_count]
    eigenvectors = np.array(eigvec_flat, copy=True).reshape(nev, ndof).T

    meta = {
        "dof_count": ndof,
        "stiffness_offdiag_count": nstiff_off,
        "mass_offdiag_count": nmass_off,
    }
    return eigenvalues, eigenvectors, ad, adb, au, aub, meta


def _infer_matrix_dimensions(remainder: np.ndarray, nev: int) -> Tuple[Optional[int], int, int]:
    """Infer the degrees of freedom and sparse storage lengths.

    The layout after the eigenvalue block is
    ``[ad | au | adb | aub | eigenvectors]``. We search for a positive diagonal
    segment ``ad`` followed by another positive diagonal ``adb`` separated by
    two unknown off-diagonal blocks. Two passes are used: a parity-based check
    assuming equal off-diagonal lengths, and a more general scan for the second
    positive block."""

    total = remainder.size
    if total == 0:
        return None, 0, 0

    max_ndof = total // max(nev + 2, 1)
    positive_mask = np.isfinite(remainder) & (remainder > 0)

    # Pass 1: assume identical sparsity for stiffness and mass matrices.
    candidates: list[Tuple[int, int, int]] = []
    for ndof in range(1, max_ndof + 1):
        rem = total - ndof * (nev + 2)
        if rem < 0:
            break
        if rem % 2 != 0:
            continue
        off = rem // 2
        if off < 0:
            continue
        if not positive_mask[:ndof].all():
            continue
        mass_diag_start = ndof + off
        mass_diag_end = mass_diag_start + ndof
        if mass_diag_end > total:
            continue
        if not positive_mask[mass_diag_start:mass_diag_end].all():
            continue
        candidates.append((ndof, off, off))

    if candidates:
        # Prefer the largest ndof (unique in practice) for robustness.
        return max(candidates, key=lambda item: item[0])

    # Pass 2: detect the second positive block directly.
    if not positive_mask[0]:
        return None, 0, 0

    nonpos_indices = np.where(~positive_mask)[0]
    if nonpos_indices.size == 0:
        return None, 0, 0
    ndof = int(nonpos_indices[0])
    if ndof <= 0:
        return None, 0, 0

    # Locate the next full block of positive entries of length ndof.
    window = np.convolve(positive_mask.astype(np.int64), np.ones(ndof, dtype=np.int64), mode="valid")
    possible_starts = np.where(window == ndof)[0]
    possible_starts = possible_starts[possible_starts >= ndof]
    for start in possible_starts:
        mass_diag_start = int(start)
        mass_diag_end = mass_diag_start + ndof
        remainder_tail = total - mass_diag_end
        if remainder_tail < 0:
            continue
        if remainder_tail < ndof * nev:
            continue
        nstiff_off = mass_diag_start - ndof
        if nstiff_off < 0:
            continue
        nmass_off = remainder_tail - ndof * nev
        if nmass_off < 0:
            continue
        return ndof, int(nstiff_off), int(nmass_off)

    return None, 0, 0


def read_eig(filepath: str) -> dict:
    """Parse a CalculiX ``.eig`` file produced by a frequency step.

    Parameters
    ----------
    filepath:
        Path to the binary ``.eig`` file.

    Returns
    -------
    dict
        A dictionary containing

        ``eigenvalues`` : ``np.ndarray``
            Stored eigenvalues (``\lambda = \omega^2``).
        ``frequencies_hz`` : ``np.ndarray``
            Derived natural frequencies ``f = sqrt(|\lambda|)/(2\pi)``.
        ``eigenvectors`` : ``np.ndarray``
            Mode shapes with shape ``(ndof, nev)``.
        ``meta`` : ``dict``
            Metadata including storage details and flags from the header."""

    path = Path(filepath)
    data = path.read_bytes()
    spec = _infer_precision_and_endianness(data)
    view = memoryview(data)
    offset = 0

    cyclicsym, offset = _unpack_int(view, offset, spec)
    nherm, offset = _unpack_int(view, offset, spec)
    iperturb, offset = _unpack_int(view, offset, spec)

    meta = {
        "endianness": "little" if spec.endian == "<" else "big",
        "int_size": spec.int_size,
        "float_size": spec.float_size,
        "cyclic_symmetry": bool(cyclicsym),
        "hermitian": bool(nherm),
        "iperturb": int(iperturb),
    }

    if cyclicsym:
        raise NotImplementedError("Cyclic symmetry .eig files are not supported")
    if nherm != 1:
        raise NotImplementedError("Non-Hermitian eigenproblems are not supported")

    ref_displacements: Optional[np.ndarray] = None

    if iperturb not in (0, 1):
        raise ValueError(f"Unexpected iperturb flag {iperturb}")

    if iperturb == 0:
        nev, offset = _unpack_int(view, offset, spec)
        float_offset = offset
        eigenvalues, eigenvectors, ad, adb, au, aub, matrix_meta = _try_parse_body(data, float_offset, nev, spec)
    else:
        # Scan possible reference-displacement lengths (stored as doubles)
        base = offset
        bytes_remaining = len(data) - base
        max_ref = max((bytes_remaining - spec.int_size) // 8, 0)
        parsed = None
        nev = None
        ref_count_found = None
        for ref_count in range(max_ref + 1):
            pos = base + ref_count * 8
            if pos + spec.int_size > len(data):
                break
            candidate_nev = struct.unpack_from(spec.int_fmt, view, pos)[0]
            if candidate_nev <= 0 or candidate_nev > 1_000_000:
                continue
            try:
                parsed = _try_parse_body(data, pos + spec.int_size, candidate_nev, spec)
            except ValueError:
                continue
            nev = candidate_nev
            ref_count_found = ref_count
            offset = pos + spec.int_size
            break
        if parsed is None or nev is None or ref_count_found is None:
            raise ValueError("Unable to locate eigenvalue block after reference displacements")
        eigenvalues, eigenvectors, ad, adb, au, aub, matrix_meta = parsed
        ref_displacements = np.frombuffer(data, dtype=spec.float_dtype, offset=base, count=ref_count_found).copy()
        meta["reference_displacements_count"] = int(ref_count_found)

    meta.update(matrix_meta)
    meta["stiffness_diagonal"] = ad
    meta["mass_diagonal"] = adb
    meta["stiffness_offdiag"] = au
    meta["mass_offdiag"] = aub

    # Convert eigenvalues to frequencies (Hz). Negative values indicate
    # numerically small rigid-body modes; use absolute value before the square
    # root to retain their magnitudes for comparison with reference data.
    frequencies = np.sqrt(np.abs(eigenvalues)) / (2.0 * math.pi)

    if ref_displacements is not None:
        meta["reference_displacements"] = ref_displacements

    return {
        "eigenvalues": eigenvalues,
        "frequencies_hz": frequencies,
        "eigenvectors": eigenvectors,
        "meta": meta,
    }


def _relative_error(measured: np.ndarray, reference: np.ndarray) -> np.ndarray:
    denom = np.where(np.abs(reference) > 0, np.abs(reference), 1.0)
    return np.abs(measured - reference) / denom


def _mac_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a_c = np.asarray(a, dtype=np.complex128)
    b_c = np.asarray(b, dtype=np.complex128)
    num = np.abs(a_c.conj().T @ b_c) ** 2
    den = np.outer(np.sum(np.abs(a_c) ** 2, axis=0), np.sum(np.abs(b_c) ** 2, axis=0))
    with np.errstate(divide="ignore", invalid="ignore"):
        mac = np.divide(num, den, out=np.zeros_like(num, dtype=np.float64), where=den != 0)
    return mac


def _format_stats(values: np.ndarray) -> str:
    return f"min={values.min():.3e}, mean={values.mean():.3e}, max={values.max():.3e}"


def _orthonormal_basis(vectors: np.ndarray) -> np.ndarray:
    if vectors.size == 0:
        return vectors
    q, _ = np.linalg.qr(vectors, mode="reduced")
    return q


def _validate_against_reference(
    result: dict,
    ref_dir: Path,
    rtol: float,
    atol: float,
    mac_threshold: float,
    use_assignment: bool,
    rigid_threshold: float,
) -> dict:
    eigenvalues = result["eigenvalues"]
    frequencies = result["frequencies_hz"]
    eigenvectors = result["eigenvectors"]

    ref_eigenvalues = np.load(ref_dir / "eigenvalues.npy")
    ref_frequencies = np.load(ref_dir / "frequencies.npy")
    ref_eigenvectors = np.load(ref_dir / "eigenvectors.npy")

    if eigenvalues.shape != ref_eigenvalues.shape:
        raise ValueError("Eigenvalue count mismatch with reference")
    if frequencies.shape != ref_frequencies.shape:
        raise ValueError("Frequency count mismatch with reference")
    if eigenvectors.shape != ref_eigenvectors.shape:
        raise ValueError("Eigenvector shape mismatch with reference")

    rigid_mask = (np.abs(frequencies) < rigid_threshold) & (np.abs(ref_frequencies) < rigid_threshold)
    flex_mask = ~rigid_mask

    print(f"Identified {rigid_mask.sum()} rigid-body candidate modes (threshold {rigid_threshold} Hz)")

    validation: dict = {
        "rigid_mask": rigid_mask,
        "flex_mask": flex_mask,
    }

    eig_pass = True
    freq_pass = True

    if np.any(flex_mask):
        eig_rel = _relative_error(eigenvalues[flex_mask], ref_eigenvalues[flex_mask])
        freq_rel = _relative_error(frequencies[flex_mask], ref_frequencies[flex_mask])
        eig_pass = np.allclose(eigenvalues[flex_mask], ref_eigenvalues[flex_mask], rtol=rtol, atol=atol)
        freq_pass = np.allclose(frequencies[flex_mask], ref_frequencies[flex_mask], rtol=rtol, atol=atol)

        print("Eigenvalue relative errors (flexible modes):")
        for idx, err in zip(np.nonzero(flex_mask)[0], eig_rel):
            print(f"  mode {idx+1:3d}: {err:.3e}")
        print(f"Eigenvalue error summary: {_format_stats(eig_rel)}; pass={eig_pass}")

        print("Frequency relative errors (flexible modes):")
        for idx, err in zip(np.nonzero(flex_mask)[0], freq_rel):
            print(f"  mode {idx+1:3d}: {err:.3e}")
        print(f"Frequency error summary: {_format_stats(freq_rel)}; pass={freq_pass}")

        validation.update(
            {
                "eigenvalues_rel_error": eig_rel,
                "frequencies_rel_error": freq_rel,
            }
        )
    else:
        print("No flexible modes left after rigid-body filtering; skipping scalar comparisons.")
        validation.update(
            {
                "eigenvalues_rel_error": np.array([]),
                "frequencies_rel_error": np.array([]),
            }
        )

    mac_pass = True
    mac_diag = np.array([])
    permutation = np.arange(eigenvectors.shape[1])

    if np.any(flex_mask):
        mac_flex = _mac_matrix(eigenvectors[:, flex_mask], ref_eigenvectors[:, flex_mask])
        if use_assignment:
            if linear_sum_assignment is None:
                raise RuntimeError("scipy is required for --use-assignment but is not available")
            cost = -mac_flex
            row_ind, col_ind = linear_sum_assignment(cost)
            mac_diag = mac_flex[row_ind, col_ind]
            flex_indices = np.nonzero(flex_mask)[0]
            permutation = flex_indices[col_ind]
            print("Applied mode pairing via Hungarian assignment (flexible modes):")
            print("  mapping parsed mode -> reference mode:", permutation + 1)
        else:
            mac_diag = np.diag(mac_flex)
            permutation = np.nonzero(flex_mask)[0]

        mac_pass = bool(np.all(mac_diag >= mac_threshold))
        print("MAC diagonal values (flexible modes):")
        for idx, val in zip(permutation, mac_diag):
            print(f"  mode {idx+1:3d}: {val:.6f}")
        print(f"MAC summary: {_format_stats(mac_diag)}; pass={mac_pass}")
    else:
        print("No flexible modes available for MAC evaluation.")

    rigid_subspace_pass = True
    rigid_subspace_values = np.array([])
    if np.any(rigid_mask):
        basis_a = _orthonormal_basis(eigenvectors[:, rigid_mask])
        basis_b = _orthonormal_basis(ref_eigenvectors[:, rigid_mask])
        if basis_a.size and basis_b.size:
            singular_vals = np.linalg.svd(basis_a.T @ basis_b, compute_uv=False)
            rigid_subspace_values = singular_vals**2
            rigid_subspace_pass = bool(np.min(rigid_subspace_values) >= mac_threshold)
            print("Rigid-body subspace MAC (principal angles squared):", rigid_subspace_values)
            print(f"Rigid-body subspace pass={rigid_subspace_pass}")
        else:
            print("Rigid-body subspace insufficient to compute MAC (degenerate vectors).")

    overall_mac_pass = mac_pass and rigid_subspace_pass

    validation.update(
        {
            "eig_pass": bool(eig_pass),
            "freq_pass": bool(freq_pass),
            "mac_pass": overall_mac_pass,
            "mac_diag": mac_diag,
            "mac_permutation": permutation,
            "rigid_subspace_mac": rigid_subspace_values,
            "rigid_pass": rigid_subspace_pass,
        }
    )

    return validation


def _write_report(report_path: Path, result: dict, validation: Optional[dict]) -> None:
    output = {
        "dofs": int(result["meta"].get("dof_count", result["eigenvectors"].shape[0])),
        "modes": int(result["eigenvectors"].shape[1]),
    }
    if validation is not None:
        mac_diag = validation.get("mac_diag", np.array([]))
        output.update(
            {
                "eigenvalues_pass": validation["eig_pass"],
                "frequencies_pass": validation["freq_pass"],
                "mac_pass": validation["mac_pass"],
                "mac_threshold": float(np.min(mac_diag)) if mac_diag.size else None,
                "mac_diag": mac_diag.tolist(),
                "rigid_subspace_mac": validation.get("rigid_subspace_mac", np.array([])).tolist(),
            }
        )
    report_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"Wrote validation report to {report_path}")


def main(argv: Optional[Iterable[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Parse CalculiX .eig files and validate against references.")
    parser.add_argument("--eig", required=True, help="Path to the .eig file")
    parser.add_argument("--ref-dir", help="Directory containing reference NumPy arrays")
    parser.add_argument("--rtol", type=float, default=1e-6, help="Relative tolerance for eigenvalue/frequency comparisons")
    parser.add_argument("--atol", type=float, default=1e-12, help="Absolute tolerance for eigenvalue/frequency comparisons")
    parser.add_argument("--mac-threshold", type=float, default=0.99, help="Minimum acceptable MAC value")
    parser.add_argument("--use-assignment", action="store_true", help="Use Hungarian assignment to pair modes before MAC reporting")
    parser.add_argument("--write-report", action="store_true", help="Emit a JSON summary next to the .eig file")
    parser.add_argument("--rigid-threshold", type=float, default=1.0, help="Frequency threshold (Hz) below which modes are treated as rigid-body subspaces")

    args = parser.parse_args(list(argv) if argv is not None else None)

    result = read_eig(args.eig)

    ndof = result["eigenvectors"].shape[0]
    nev = result["eigenvectors"].shape[1]
    print(f"Parsed {ndof} DOFs and {nev} modes from {args.eig}")
    print(f"Endianness: {result['meta']['endianness']}, int size: {result['meta']['int_size']} bytes")

    validation = None
    if args.ref_dir:
        ref_dir = Path(args.ref_dir)
        validation = _validate_against_reference(
            result,
            ref_dir,
            rtol=args.rtol,
            atol=args.atol,
            mac_threshold=args.mac_threshold,
            use_assignment=args.use_assignment,
            rigid_threshold=args.rigid_threshold,
        )
        mac_pass = validation["mac_pass"]
        eig_pass = validation["eig_pass"]
        freq_pass = validation["freq_pass"]
        if eig_pass and freq_pass and mac_pass:
            print("All validation checks passed.")
        else:
            print("Validation failed: eigenvalues pass=%s, frequencies pass=%s, MAC pass=%s" % (eig_pass, freq_pass, mac_pass))

    if args.write_report:
        eig_path = Path(args.eig)
        report_path = eig_path.with_name(eig_path.name + "_validation.json")
        _write_report(report_path, result, validation)


if __name__ == "__main__":
    main()
