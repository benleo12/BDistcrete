#!/bin/bash
# Fetch the Section 6 results of the Perlmutter jobs into output/. Files not yet written are reported
# and skipped. The shrunk box and the two variant fits are taken from their separate jobs
# (rows_shrink, rows_variants: *_sep), which run all three passes, and copied to the plain names the
# paper scripts read, replacing any older local copy. rows_main reaches its time limit before its own
# stages 3 and 4 can finish, so its versions of these files are never used.
cd .
P="-i $HOME/.ssh/nersc -o CertificateFile=$HOME/.ssh/nersc-cert.pub"
R=${NERSC_USER}@perlmutter.nersc.gov:$SCRATCH/BDistcrete/analysis/output
for f in profile_MIX17ext_central_rows_var.json profile_MIX17ext_central_rows_shrink0.90sep.json \
         profile_MIX17geo_central_rows_sep.json profile_MIX17win_central_rows_sep.json \
         pseudo_rows_profile_MIX17ext_central_summary.json \
         profile_MIX17ext_central_exchange.json profile_MIX17ext_central_exchange_dist.npz \
         profile_MIX17ext_central_surface.json; do
  scp -q $P $R/$f output/ 2>/dev/null && echo "fetched $f" || echo "not yet: $f"
done
mkdir -p output/pseudo_rows_sets
scp -q $P "$R/pseudo_rows_profile_MIX17ext_central_set*.json" output/pseudo_rows_sets/ 2>/dev/null && echo "fetched $(ls output/pseudo_rows_sets | wc -l) pseudo-data sets"
for pair in "profile_MIX17ext_central_rows_shrink0.90sep.json profile_MIX17ext_central_rows_shrink0.90.json" \
            "profile_MIX17geo_central_rows_sep.json profile_MIX17geo_central_rows.json" \
            "profile_MIX17win_central_rows_sep.json profile_MIX17win_central_rows.json"; do
  set -- $pair; [ -f output/$1 ] && cp output/$1 output/$2 && echo "  $2 <- $1"
done
