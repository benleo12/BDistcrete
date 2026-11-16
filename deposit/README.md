# The deposit

- `reference_MIX17.npz.part-aa` to `part-as`: the reference sample of `MIX17`, 865 MB, in 19 parts,
  since GitHub refuses files over 100 MB.
- `targets_thrust_rate_shape.npz`: the current NNLL+NLO thrust moments and paired window rates.
- `targets_thrust_nlo.npz`: the archived conditional-moment target file.
- `build_reference.json`: the record of the build of the reference sample.
- `SHA256SUMS`: the sha256 of the joined sample and of the targets and build record.

From `analysis/release`, join the parts, copy the targets files and check all checksums (three OK lines):

```bash
cat ../../deposit/reference_MIX17.npz.part-* > reference_MIX17.npz
cp ../../deposit/targets_thrust_*.npz .
grep -v build_reference ../../deposit/SHA256SUMS | shasum -a 256 -c
```

`analysis/release/REFERENCE.md` describes the sample, how it was built and how it was checked.
