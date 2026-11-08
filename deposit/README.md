# The deposit

- `reference_MIX17.npz.part-aa` to `part-as`: the reference sample of `MIX17`, 865 MB, in 19 parts,
  since GitHub refuses files over 100 MB.
- `targets_thrust_nlo.npz`: the paper's NNLL+NLO thrust calculation.
- `build_reference.json`: the record of the build of the reference sample.
- `SHA256SUMS`: the sha256 of the joined sample and of the two files above.

From `analysis/release`, join the parts, copy the targets file and check both (two OK lines):

```bash
cat ../../deposit/reference_MIX17.npz.part-* > reference_MIX17.npz
cp ../../deposit/targets_thrust_nlo.npz .
grep -v build_reference ../../deposit/SHA256SUMS | shasum -a 256 -c
```

`analysis/release/REFERENCE.md` describes the sample, how it was built and how it was checked.
